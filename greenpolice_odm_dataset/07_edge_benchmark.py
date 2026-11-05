"""
E-Waste Detection Pipeline — Phase 7: Edge Benchmark & Deployment Validation
=============================================================================
Runs on the Jetson Orin NX (or any TensorRT-capable device).
Measures real-world latency, throughput, and accuracy of the deployed model.

This script does four things:
  1. Latency benchmark  — P50/P95/P99 inference time across input sizes
  2. Accuracy benchmark — mAP on a held-out test set (never seen during training)
  3. Confidence calibration — tune threshold for recall ≥ 0.87 at lowest FP
  4. Forensic output validation — verify GPS tagging, timestamp, audit trail

TILED INFERENCE (2026-10-02): the model takes ONE TILE (e.g. 768x768), not a whole frame.
  A 4K/20 MP frame is cut into overlapping native-resolution tiles plus a few coarser copies
  (ewads_tiling.py), the model runs once per tile, and the detections are merged.  Previously
  this script squashed the full frame to a square 640x640 (aspect ratio distorted, 8.5x
  downscale on a 20 MP frame), so the deployed model never saw a phone or a chip.
    - latency is now reported per TILE and per FRAME (tiles/frame x per-tile time);
    - accuracy/calibration/forensic modes all use the tiled path (--no_tiled = old behaviour);
    - the audit log records the tiling parameters next to the model version, so a detection
      can be reproduced bit-for-bit.
  For paper-grade mAP use 08_tiled_eval.py; the TP/FP/FN table here is IoU 0.5 only.

SURVEY MODE + CAPTURE CONTEXT (2026-10-02, second pass):
  * --video FLIGHT.MP4 [--srt_file FLIGHT.SRT]: the detector runs only on frames that add new
    ground (ewads_survey.CoverageScheduler): every object up to --max_object_m is fully inside
    --views processed frames.  Motion from SRT speed / GPS, else image phase correlation; an
    unreliable estimate means "process" (fails towards recall).  ~0.4-1.3 Hz instead of 30 fps.
  * every audit record now carries a `context`: height above take-off (DJI XMP RelativeAltitude
    or SRT 'H' -- NOT EXIF GPSAltitude, which is above sea level), GSD, gimbal pitch / oblique
    flag, exposure and motion blur, and which classes were ASSESSABLE / marginal / NOT assessable
    at that GSD.  A missing detection of a not-assessable class is not evidence of absence.
  * --auto_levels: per-frame pyramid levels from GSD (skips coarse levels at altitude).
  * TensorRT backend rewritten: TensorRT 8.5+/10 tensor API (+ legacy fallback), Ultralytics
    .engine metadata header, static or dynamic batch, buffers allocated once, no pycuda.
    Tiles are batched (FrameDetector), and --benchmark reports per_tile_ms_effective, which
    09_mission_planner.py reads.

Run on Jetson (engines are built there by 10_jetson_build_engine.py):
  python 07_edge_benchmark.py --mode trt --engine_path runs/jetson/ewads_fp16_b8.engine --benchmark
  python 07_edge_benchmark.py --mode trt --engine_path runs/jetson/ewads_fp16_b8.engine --conf 0.31 \
    --video DJI_0042.MP4 --srt_file DJI_0042.SRT --auto_levels --output_dir ./benchmark_results

Or run ONNX inference on laptop (no TensorRT needed) for pre-Jetson validation:
  python 07_edge_benchmark.py \
    --onnx_path  ./runs/export/ewads_yolov10n_fp32.onnx \
    --test_dir   ./data/test \
    --output_dir ./benchmark_results \
    --mode onnx

GPS Tagging (DJI Phantom 4 Pro):
  DJI Phantom 4 Pro embeds GPS EXIF data in every captured image.
  For live drone inference, telemetry is read from DJI's SRT subtitle file
  or via DJI SDK (if integrated). Pixhawk integration uses MAVLink telemetry.
  See _extract_gps_from_exif() and _extract_gps_from_mavlink() below.

Immutable Audit Trail:
  Every detection event is written as a SHA-256 hashed JSON record to an
  append-only JSONL file. The hash chain links each record to the previous,
  creating a tamper-evident log. See ForensicAuditLogger class.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ewads_tiling as et
import ewads_survey as es
import ewads_taxonomy as tx

try:
    import cv2
except ImportError:
    print("[ERROR] opencv not found: pip install opencv-python-headless")
    sys.exit(1)

# ─── Constants ────────────────────────────────────────────────────────────────
EWASTE_CLASSES = [
    "crt_monitor", "flat_screen", "circuit_board", "battery",
    "cable_bundle", "single_cable", "mobile_phone", "keyboard",
    "printer", "small_appliance", "large_appliance", "cpu_chip",
    "hard_drive", "power_supply", "ewaste_pile", "plastic_casing",
    "metal_component", "lighting",
]

# Forensic criticality tiers for reporting
# Single source of truth: ewads_taxonomy.TIER_1 (also used by 08 / 11 and targets E3).
# Change tier-1 there, before freezing targets_v1.json.
TIER_1_CLASSES = set(tx.TIER_1)  # Toxic materials

# Target metrics
TARGET_MAP5095 = 0.68
TARGET_RECALL  = 0.87
TARGET_FP_RATE = 0.05


# ─── Forensic Audit Logger ────────────────────────────────────────────────────
class ForensicAuditLogger:
    """
    Append-only, hash-chained detection log for forensic admissibility.

    Each record contains:
      - Detection payload (image hash, GPS, timestamp, detections)
      - SHA-256 hash of current record content
      - SHA-256 hash of previous record (chain link)
      - Model version string

    The chain can be verified by replaying all hashes forward.
    Tampering with any record breaks all subsequent hashes.

    This is NOT a substitute for a proper WORM (Write-Once Read-Many)
    storage system in production. Use this log as the cryptographic layer
    on top of whatever write-once storage your deployment uses.
    """

    # Field meanings (2026-10-08):
    #   timestamp_utc        PROCESSING time: when this record was written (not when the image was taken)
    #   capture_time_utc     when the pixels were captured, in UTC -- only if the source carries a UTC
    #                        offset or the operator asserted one (see capture_time_source); else null
    #   capture_time_local   camera-local capture time as recorded (zone may be unknown)
    #   capture_time_source  where the capture time came from, or "unavailable"
    #   model_sha256         SHA-256 of the weights / ONNX / engine file actually loaded
    #                        (ensembles in 11: list of the members' SHA-256)

    def __init__(self, log_path: Path, model_version: str, model_sha256=None,
                 verify_on_open: bool = True):
        self.log_path      = log_path
        self.model_version = model_version
        self.model_sha256  = model_sha256
        self.prev_hash     = "GENESIS"   # Chain starts with sentinel
        self._record_count = 0

        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Resuming an existing log: verify the WHOLE chain before appending to it.  Appending to a
        # broken chain would bury the break under valid-looking new records.
        if log_path.exists() and log_path.stat().st_size > 0:
            if verify_on_open:
                v = self.verify_chain()
                if not v.get("valid"):
                    raise RuntimeError(
                        f"[AuditLog] REFUSING to append to {log_path}: existing chain is broken "
                        f"({v.get('reason', v.get('error'))} at record {v.get('broken_at')}). Preserve this file "
                        f"as evidence and start a new log; do not repair it in place.")
            with open(log_path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.readlines() if ln.strip()]
            if lines:
                try:
                    self.prev_hash = json.loads(lines[-1])["record_hash"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    # only reachable with verify_on_open=False (a verified chain has a parseable tail):
                    # let verify_chain() report the damage instead of crashing here
                    self.prev_hash = None
                self._record_count = len(lines)
                print(f"  [AuditLog] Resuming chain at record {self._record_count}"
                      f"{' (verified)' if verify_on_open else ' (NOT verified: verify-only open)'}")
                print(f"  [AuditLog] Chain tail: {(self.prev_hash or 'UNREADABLE')[:16]}...")

    def log_detection(self, image_path: str, image_hash: str,
                      detections: list, gps: Optional[dict],
                      altitude_m: Optional[float],
                      confidence_threshold: float,
                      context: Optional[dict] = None,
                      capture_time: Optional[dict] = None) -> dict:
        """
        Write one forensic detection record.

        detections: list of dicts with keys:
          class_id, class_name, confidence, bbox_xyxy, tier
        capture_time: ewads_survey.capture_time_from_image / _from_srt output (None -> "unavailable")

        Returns the written record dict.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()      # PROCESSING time
        ct = {**es.CAPTURE_TIME_UNAVAILABLE, **(capture_time or {})}

        record_payload = {
            "record_index":   self._record_count,
            "timestamp_utc":  timestamp_utc,
            "capture_time_utc":    ct["capture_time_utc"],
            "capture_time_local":  ct["capture_time_local"],
            "capture_time_source": ct["capture_time_source"],
            "model_version":  self.model_version,
            "model_sha256":   self.model_sha256,
            "image_path":     str(image_path),
            "image_sha256":   image_hash,
            "gps":            gps,
            "altitude_m":     altitude_m,
            "conf_threshold": confidence_threshold,
            "detections":     detections,
            "n_detections":   len(detections),
            "tier1_detected": any(d["class_name"] in TIER_1_CLASSES
                                  for d in detections),
            "prev_hash":      self.prev_hash,
        }
        if context:                      # capture conditions + what this frame can / cannot prove
            record_payload["context"] = context

        # Hash the payload
        payload_str  = json.dumps(record_payload, sort_keys=True, default=str)
        record_hash  = hashlib.sha256(payload_str.encode()).hexdigest()

        record = {**record_payload, "record_hash": record_hash}

        # Append to log (never overwrite)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

        self.prev_hash      = record_hash
        self._record_count += 1

        return record

    def verify_chain(self) -> dict:
        """
        Replay the entire chain and verify hash integrity.
        Returns dict with valid=True if chain is unbroken, else first broken link.
        """
        if not self.log_path.exists():
            return {"valid": False, "error": "Log file not found"}

        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.readlines() if ln.strip()]

        if not lines:
            return {"valid": True, "records": 0}

        prev_hash = "GENESIS"
        for i, line in enumerate(lines):
            try:
                record = json.loads(line)
                stored_hash = record.pop("record_hash")
            except (json.JSONDecodeError, KeyError, AttributeError):
                return {"valid": False, "broken_at": i,
                        "reason": "unparseable record or missing record_hash (truncated write?)"}

            # Recompute hash
            payload_str    = json.dumps(record, sort_keys=True, default=str)
            expected_hash  = hashlib.sha256(payload_str.encode()).hexdigest()

            if record.get("prev_hash") != prev_hash:
                return {
                    "valid":        False,
                    "broken_at":    i,
                    "reason":       "prev_hash mismatch",
                    "expected":     prev_hash,
                    "found":        record.get("prev_hash"),
                }

            if stored_hash != expected_hash:
                return {
                    "valid":        False,
                    "broken_at":    i,
                    "reason":       "record_hash mismatch — record may be tampered",
                    "expected":     expected_hash,
                    "found":        stored_hash,
                }

            prev_hash = stored_hash

        return {"valid": True, "records": len(lines), "chain_tail": prev_hash}


# ─── GPS Extraction ───────────────────────────────────────────────────────────
def _extract_gps_from_exif(image_path: str) -> Optional[dict]:
    """
    Extract GPS coordinates from DJI image EXIF data.
    DJI Phantom 4 Pro embeds GPS, altitude, and gimbal angles in EXIF.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS

        img  = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None

        # Decode EXIF tags
        decoded = {TAGS.get(k, k): v for k, v in exif.items()}
        gps_raw = decoded.get("GPSInfo")
        if not gps_raw:
            return None

        gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
        if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
            return None                      # no fix: never report (0, 0) as a position

        def _dms_to_decimal(dms, ref):
            d, m, s = [float(x) for x in dms]
            decimal  = d + m / 60 + s / 3600
            return -decimal if ref in ["S", "W"] else decimal

        lat = _dms_to_decimal(gps.get("GPSLatitude", [0, 0, 0]),
                               gps.get("GPSLatitudeRef", "N"))
        lon = _dms_to_decimal(gps.get("GPSLongitude", [0, 0, 0]),
                               gps.get("GPSLongitudeRef", "E"))
        alt = float(gps.get("GPSAltitude", 0))

        return {
            "latitude":  lat,
            "longitude": lon,
            "altitude_m": alt,
            "source":    "exif_dji",
        }

    except Exception:
        return None


def _extract_gps_from_srt(srt_path: str, frame_number: int) -> Optional[dict]:
    """
    Extract GPS from DJI SRT (subtitle) telemetry file.
    DJI creates a .SRT file alongside each video with per-frame GPS/IMU data.
    Format varies by DJI firmware — this handles Phantom 4 Pro format.

    srt_path: path to .SRT file (same name as video, .srt extension)
    frame_number: video frame index to extract
    """
    try:
        with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # DJI SRT format: blocks separated by double newline
        # Each block: index, timestamp, telemetry line
        blocks = content.strip().split("\n\n")

        if frame_number >= len(blocks):
            frame_number = len(blocks) - 1

        block = blocks[frame_number]
        lines = block.strip().split("\n")

        # Find GPS line (contains [latitude: ... longitude: ...] or similar)
        import re
        for line in lines:
            lat_match = re.search(r"latitude\s*[:=]\s*([-\d.]+)", line, re.I)
            lon_match = re.search(r"longitude\s*[:=]\s*([-\d.]+)", line, re.I)
            alt_match = re.search(r"altitude\s*[:=]\s*([-\d.]+)", line, re.I)

            if lat_match and lon_match:
                return {
                    "latitude":   float(lat_match.group(1)),
                    "longitude":  float(lon_match.group(1)),
                    "altitude_m": float(alt_match.group(1)) if alt_match else None,
                    "source":     "dji_srt",
                    "frame":      frame_number,
                }

        return None

    except Exception:
        return None


def _extract_gps_from_mavlink(mavlink_log_path: str,
                               timestamp: float) -> Optional[dict]:
    """
    Extract GPS from Pixhawk MAVLink telemetry log at a given timestamp.
    Uses pymavlink if available.

    mavlink_log_path: path to .tlog or .bin file from Pixhawk
    timestamp: Unix timestamp to match
    """
    try:
        from pymavlink import mavutil

        mav = mavutil.mavlink_connection(mavlink_log_path)
        best_msg = None
        best_dt  = float("inf")

        while True:
            msg = mav.recv_match(type="GPS_RAW_INT", blocking=False)
            if msg is None:
                break
            msg_ts = msg._timestamp
            dt     = abs(msg_ts - timestamp)
            if dt < best_dt:
                best_dt  = dt
                best_msg = msg

        if best_msg and best_dt < 1.0:  # Within 1 second
            return {
                "latitude":   best_msg.lat / 1e7,
                "longitude":  best_msg.lon / 1e7,
                "altitude_m": best_msg.alt / 1e3,
                "fix_type":   best_msg.fix_type,
                "source":     "pixhawk_mavlink",
                "time_delta_s": best_dt,
            }
        return None

    except ImportError:
        return None
    except Exception:
        return None


# ─── Capture context: altitude, GSD, and what the frame can prove ───────────
OBLIQUE_LIMIT_DEG = 10.0     # gimbal more than this off nadir -> GSD varies across the frame


def capture_context(frame_wh: tuple, altitude_agl: Optional[float], altitude_source: str,
                    gimbal_pitch: Optional[float] = None, exposure_s: Optional[float] = None,
                    speed_mps: Optional[float] = None, camera: str = "phantom4pro",
                    mode: Optional[str] = None) -> dict:
    """
    Everything a reviewer needs to interpret a NEGATIVE: at this GSD, which classes could the
    detector have seen at all?  'not_assessable' classes are not evidence of absence.
    GSD uses the height above take-off (DJI RelativeAltitude / SRT 'H') -- never EXIF GPSAltitude,
    which is above sea level.
    """
    cam = es.CAMERAS[camera]
    W, H = frame_wh
    gsd = es.gsd_m(altitude_agl, W, cam, mode) if altitude_agl and altitude_agl > 0 else None
    oblique = gimbal_pitch is not None and abs(gimbal_pitch + 90.0) > OBLIQUE_LIMIT_DEG
    ctx = {
        "frame_wh": [W, H],
        "altitude_agl_m": round(altitude_agl, 2) if altitude_agl else None,
        "altitude_source": altitude_source if gsd else "unknown",
        "gsd_cm_per_px": round(gsd * 100, 3) if gsd else None,
        "gimbal_pitch_deg": gimbal_pitch,
        "oblique": bool(oblique),
        "exposure_s": exposure_s,
        "motion_blur_px": round(es.motion_blur_px(speed_mps, exposure_s, gsd), 2)
                          if (gsd and exposure_s and speed_mps is not None) else None,
        "resolvability_rule": f"smallest realistic instance >= {es.ASSESS_PX}px assessable, "
                              f">= {es.MARGINAL_PX}px marginal",
        "classes": es.resolvability_summary(gsd),
    }
    if oblique:
        ctx["warning"] = "oblique view: GSD and assessability are approximate"
    if gsd is None:
        ctx["warning"] = "no above-ground altitude: assessability unknown, negatives prove nothing"
    return ctx


def image_capture_meta(image_path: str, altitude_override: Optional[float]) -> dict:
    """DJI XMP (RelativeAltitude, gimbal, speed) + EXIF exposure for one still."""
    xmp = es.read_dji_xmp(image_path)
    alt, src = None, "unknown"
    if altitude_override:
        alt, src = altitude_override, "operator (--altitude_agl)"
    elif "rel_alt_m" in xmp:
        alt, src = xmp["rel_alt_m"], "dji_xmp RelativeAltitude (above take-off)"
    spd = None
    if "vx_mps" in xmp and "vy_mps" in xmp:
        spd = float(np.hypot(xmp["vx_mps"], xmp["vy_mps"]))
    return {"altitude_agl": alt, "altitude_source": src, "gimbal_pitch": xmp.get("gimbal_pitch_deg"),
            "exposure_s": es.read_exposure_s(image_path), "speed_mps": spd}


def get_image_sha256(image_path: str) -> str:
    """Compute SHA-256 hash of raw image file bytes."""
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Inference Backends ───────────────────────────────────────────────────────
class ONNXInferencer:
    """ONNX Runtime inference backend (CPU or CUDA)."""

    def __init__(self, onnx_path: Path):
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if self._cuda_available() else ["CPUExecutionProvider"])
        self.session    = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_size  = self.input_shape[2] if isinstance(self.input_shape[2], int) else None
        b0 = self.input_shape[0]
        self.dynamic   = not isinstance(b0, int)
        self.max_batch = 16 if self.dynamic else int(b0)      # static N>1 exports are padded
        print(f"  [ONNX] Loaded {onnx_path.name}  input {self.input_shape}")
        print(f"  [ONNX] Provider: {self.session.get_providers()[0]}")

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import onnxruntime as ort
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    def preprocess(self, img: np.ndarray, imgsz: int = 640) -> np.ndarray:
        # LEGACY full-frame path (squashes the frame to a square). Use the tiled path instead.
        img_resized = cv2.resize(img, (imgsz, imgsz))
        img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm    = img_rgb.astype(np.float32) / 255.0
        return np.transpose(img_norm, (2, 0, 1))[np.newaxis]  # 1xCxHxW

    @staticmethod
    def preprocess_tile(tile_rgb: np.ndarray) -> np.ndarray:
        """A T x T RGB tile -> 1x3xTxT float32 in [0,1].  No resizing."""
        return np.ascontiguousarray((tile_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])

    def infer(self, img_tensor: np.ndarray) -> list:
        b = img_tensor.shape[0]
        if not self.dynamic and b < self.max_batch:            # static batch: pad, then trim
            pad = np.zeros((self.max_batch - b,) + img_tensor.shape[1:], img_tensor.dtype)
            outs = self.session.run(None, {self.input_name: np.concatenate([img_tensor, pad])})
            return [o[:b] for o in outs]
        return self.session.run(None, {self.input_name: img_tensor})

    def infer_timed(self, img_tensor: np.ndarray) -> tuple:
        t0 = time.perf_counter()
        outputs = self.infer(img_tensor)
        t1 = time.perf_counter()
        return outputs, (t1 - t0) * 1000  # ms

    def parse_detections(self, outputs: list, conf_thresh: float,
                         orig_h: int, orig_w: int, imgsz: int = 640) -> list:
        """
        Parse ONNX output into detection dicts.
        Handles both YOLOv10 and RT-DETR output formats.
        Output format varies — adjust if your model differs.
        """
        detections = []

        # YOLO-family: output shape (1, num_boxes, 4+num_classes)
        # RT-DETR: output shape (1, num_queries, 4+num_classes)
        if len(outputs) == 1:
            preds = outputs[0][0]  # Remove batch dim → (N, 4+C)

            for pred in preds:
                if len(pred) < 5:
                    continue
                bbox   = pred[:4]
                scores = pred[4:]

                cls_id = int(np.argmax(scores))
                conf   = float(scores[cls_id])

                if conf < conf_thresh:
                    continue

                # Denormalize bbox (cx, cy, w, h) → (x1, y1, x2, y2)
                cx, cy, w, h = bbox
                x1 = int((cx - w / 2) * orig_w)
                y1 = int((cy - h / 2) * orig_h)
                x2 = int((cx + w / 2) * orig_w)
                y2 = int((cy + h / 2) * orig_h)

                cls_name = EWASTE_CLASSES[cls_id] if cls_id < len(EWASTE_CLASSES) else "unknown"

                detections.append({
                    "class_id":    cls_id,
                    "class_name":  cls_name,
                    "confidence":  round(conf, 4),
                    "bbox_xyxy":   [x1, y1, x2, y2],
                    "tier":        1 if cls_name in TIER_1_CLASSES else 2,
                })

        return detections


def read_engine_bytes(engine_path: Path) -> tuple:
    """
    (serialized_engine_bytes, metadata).  Ultralytics' .engine export writes a 4-byte little-endian
    JSON length + JSON metadata BEFORE the serialized engine; trtexec writes the bare engine.
    Feeding the Ultralytics file straight to deserialize_cuda_engine() fails, so strip the header
    (same rule as ultralytics/nn/backends/base.py: a header is a length that fits and parses as JSON).
    """
    data = Path(engine_path).read_bytes()
    n = int.from_bytes(data[:4], "little")
    if 0 < n <= len(data) - 4:
        try:
            meta = json.loads(data[4:4 + n].decode("utf-8"))
            if isinstance(meta, dict):
                return data[4 + n:], meta
        except (UnicodeDecodeError, ValueError):
            pass
    return data, {}


class TensorRTInferencer:
    """
    TensorRT backend for the Jetson.

    * TensorRT 8.5+ / 10.x name-based tensor API (JetPack 6.x ships TRT 8.6 or 10.x; the old
      binding API -- get_binding_shape / execute_async_v2 -- was removed in TRT 10), with a
      fallback to the binding API for older TensorRT.
    * Reads both bare trtexec engines and Ultralytics .engine files (metadata header).
    * Static OR dynamic batch: tiles are batched up to `max_batch` (static engines are padded).
    * Buffers are allocated ONCE (the previous version allocated + freed GPU memory on every call,
      which inflated the measured latency).
    * Device memory via torch (already required on the Jetson for Ultralytics); no pycuda.
    """

    def __init__(self, engine_path: Path):
        try:
            import tensorrt as trt
            import torch
        except ImportError as e:
            raise ImportError(
                f"{e}\nTensorRT backend needs the JetPack TensorRT Python bindings and torch "
                "(NVIDIA's Jetson wheel). Or use --mode onnx.")
        self.trt, self.torch = trt, torch
        self._check_cuda()
        self.logger = trt.Logger(trt.Logger.WARNING)
        blob, self.metadata = read_engine_bytes(engine_path)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize {engine_path} (TensorRT version mismatch? engines "
                               "must be built on the same JetPack/TensorRT that runs them)")
        self.context = self.engine.create_execution_context()
        self.device = self._device()
        self.stream = self._make_stream()
        self.new_api = hasattr(self.engine, "num_io_tensors")

        if self.new_api:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            is_in = {n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in names}
            shape = lambda n: tuple(self.engine.get_tensor_shape(n))          # noqa: E731
            dtype = lambda n: trt.nptype(self.engine.get_tensor_dtype(n))     # noqa: E731
        else:
            names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
            is_in = {n: self.engine.binding_is_input(i) for i, n in enumerate(names)}
            shape = lambda n: tuple(self.engine.get_binding_shape(self.engine.get_binding_index(n)))  # noqa: E731
            dtype = lambda n: trt.nptype(self.engine.get_binding_dtype(self.engine.get_binding_index(n)))  # noqa: E731
        self.names = names
        self.input_name = next(n for n in names if is_in[n])
        self.output_names = [n for n in names if not is_in[n]]
        in_shape = shape(self.input_name)                                   # (B|-1, 3, T, T)
        self.dynamic = in_shape[0] == -1
        if self.dynamic:
            if self.new_api:
                _mn, opt, mx = self.engine.get_tensor_profile_shape(self.input_name, 0)
            else:
                _mn, opt, mx = self.engine.get_profile_shape(0, self.engine.get_binding_index(self.input_name))
            self.max_batch = int(mx[0])
            # Ultralytics' dynamic engines are dynamic in H/W too (32 .. 2x imgsz): the tile size is
            # the OPT shape (= the export imgsz), never the profile max.
            hw = tuple(int(x) for x in opt[2:])
            imgsz = self.metadata.get("imgsz") if isinstance(self.metadata, dict) else None
            if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
                hw = (int(imgsz[0]), int(imgsz[1]))
            in_shape = (self.max_batch, int(opt[1])) + hw
        else:
            self.max_batch = int(in_shape[0])
        self.input_size = int(in_shape[2]) if in_shape[2] > 0 else None
        self._np_dtype = {n: dtype(n) for n in names}
        tmap = {np.float32: torch.float32, np.float16: torch.float16, np.int32: torch.int32,
                np.int64: torch.int64, np.int8: torch.int8, np.bool_: torch.bool}
        self._tdt = {n: tmap[np.dtype(self._np_dtype[n]).type] for n in names}

        # allocate once at max batch
        self._set_batch(self.max_batch, force=True)
        self.buffers = {}
        for n in names:
            shp = in_shape if n == self.input_name else self._out_shape(n)
            self.buffers[n] = torch.empty(tuple(int(s) for s in shp), dtype=self._tdt[n], device=self.device)
        self._cur_batch = self.max_batch
        print(f"  [TRT] Loaded {Path(engine_path).name}  (TensorRT {trt.__version__}, "
              f"{'tensor' if self.new_api else 'binding'} API)")
        print(f"  [TRT] Input {self.input_name} {in_shape} {np.dtype(self._np_dtype[self.input_name]).name}, "
              f"{'dynamic' if self.dynamic else 'static'} batch, max {self.max_batch}")
        if self.metadata:
            print(f"  [TRT] Ultralytics metadata: imgsz={self.metadata.get('imgsz')} "
                  f"batch={self.metadata.get('batch')} names={len(self.metadata.get('names', {}))}")

    # ---- device hooks (overridden by tests/test_trt_backend.py to run on CPU) ------------------
    def _check_cuda(self):
        if not self.torch.cuda.is_available():
            raise RuntimeError("TensorRT backend needs CUDA (torch.cuda.is_available() is False)")

    def _device(self):
        return self.torch.device("cuda:0")

    def _make_stream(self):
        return self.torch.cuda.Stream()

    def _stream_ctx(self):
        return self.torch.cuda.stream(self.stream)

    # ---- helpers -------------------------------------------------------------------------------
    def _set_batch(self, b: int, force: bool = False):
        if not self.dynamic or (not force and b == getattr(self, "_cur_batch", None)):
            return
        shp = (b, 3, self.input_size or 0, self.input_size or 0) if self.input_size else None
        if shp is None:
            raise RuntimeError("dynamic spatial size is not supported; export with a fixed imgsz")
        if self.new_api:
            self.context.set_input_shape(self.input_name, shp)
        else:
            self.context.set_binding_shape(self.engine.get_binding_index(self.input_name), shp)
        self._cur_batch = b

    def _out_shape(self, n):
        if self.new_api:
            return tuple(self.context.get_tensor_shape(n))
        return tuple(self.context.get_binding_shape(self.engine.get_binding_index(n)))

    @staticmethod
    def preprocess(img: np.ndarray, imgsz: int = 640) -> np.ndarray:
        # LEGACY full-frame path (squashes the frame to a square). Use the tiled path instead.
        img_resized = cv2.resize(img, (imgsz, imgsz))
        img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm    = img_rgb.astype(np.float32) / 255.0
        return np.ascontiguousarray(np.transpose(img_norm, (2, 0, 1))[np.newaxis])

    @staticmethod
    def preprocess_tile(tile_rgb: np.ndarray) -> np.ndarray:
        """A T x T RGB tile -> 1x3xTxT float32 in [0,1].  No resizing."""
        return np.ascontiguousarray((tile_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])

    def infer(self, img_tensor: np.ndarray) -> list:
        """(B,3,T,T) float32 -> list of output arrays with leading dim B (B <= max_batch)."""
        torch = self.torch
        b = int(img_tensor.shape[0])
        if b > self.max_batch:
            raise ValueError(f"batch {b} > engine max batch {self.max_batch}")
        run_b = b if self.dynamic else self.max_batch
        self._set_batch(run_b)
        inp = self.buffers[self.input_name]
        with self._stream_ctx():
            x = torch.from_numpy(np.ascontiguousarray(img_tensor)).to(self.device, non_blocking=True)
            inp[:b].copy_(x.to(self._tdt[self.input_name]))
            if run_b > b:
                inp[b:run_b].zero_()
            if self.new_api:
                for n in self.names:
                    self.context.set_tensor_address(n, self.buffers[n].data_ptr())
                ok = self.context.execute_async_v3(self.stream.cuda_stream)
            else:
                bindings = [0] * len(self.names)
                for n in self.names:
                    bindings[self.engine.get_binding_index(n)] = self.buffers[n].data_ptr()
                ok = self.context.execute_async_v2(bindings, self.stream.cuda_stream)
            outs = []
            for n in self.output_names:
                shp = self._out_shape(n)                         # current (possibly smaller) batch
                o = self.buffers[n].view(-1)[:int(np.prod(shp))].view(*shp) if self.dynamic else self.buffers[n]
                outs.append(o[:b].float().cpu().numpy())
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        return outs

    def infer_timed(self, img_tensor: np.ndarray) -> tuple:
        t0 = time.perf_counter()
        outputs = self.infer(img_tensor)
        return outputs, (time.perf_counter() - t0) * 1000


# ─── Tiled detection (the deployed inference path) ───────────────────────────
class FrameDetector:
    """
    Full-frame -> list of detection dicts, via tiles (default) or the legacy squash.

    Tiles are sent to the model in batches of up to `inferencer.max_batch` (static-batch engines
    are padded), which is where most of the Jetson's throughput comes from.

    auto_levels: with a known GSD (altitude), run only the pyramid levels the LARGEST expected
    object needs (ewads_survey.levels_needed) -- e.g. at 60 m a 3.5 m pile is ~150 px, so the
    1/4 and 1/8 levels are skipped.  The final whole-frame level is always kept as insurance.
    """

    def __init__(self, inferencer, tile: Optional[int], overlap: int = et.DEFAULT_OVERLAP,
                 tiled: bool = True, max_levels: Optional[int] = None, nms_iou: float = 0.5,
                 auto_levels: bool = False, largest_object_m: float = 3.5):
        self.inf, self.tiled, self.tile, self.overlap = inferencer, tiled, tile, overlap
        self.max_levels, self.auto_levels, self.largest_object_m = max_levels, auto_levels, largest_object_m
        self.batch = max(1, int(getattr(inferencer, "max_batch", 1) or 1))
        self.last_ms = 0.0
        self.last_tiles = 1
        self.last_levels = max_levels
        if tiled:
            def predict(tiles):
                x = np.concatenate([inferencer.preprocess_tile(t) for t in tiles])
                raw = inferencer.infer(x)[0]
                return [et.parse_raw_output(raw[i:i + 1], tile, 0.001) for i in range(len(tiles))]
            self.det = et.TiledDetector(predict, tile=tile, overlap=overlap, batch=self.batch, conf=0.001,
                                        nms_iou=nms_iou, max_levels=max_levels)

    def __call__(self, frame_bgr: np.ndarray, conf: float, gsd: Optional[float] = None) -> list:
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        if not self.tiled:
            tensor = self.inf.preprocess(frame_bgr)
            outs = self.inf.infer(tensor)
            self.last_ms = (time.perf_counter() - t0) * 1000
            return self.inf.parse_detections(outs, conf, h, w)
        lv = self.max_levels
        if self.auto_levels and gsd:
            need = es.levels_needed(gsd, (w, h), self.tile, self.overlap, self.largest_object_m)
            lv = min(need, lv) if lv else need
        self.det.max_levels = lv
        self.last_levels = lv
        d = self.det(frame_bgr)
        self.last_ms = (time.perf_counter() - t0) * 1000
        self.last_tiles = self.det.last_n_tiles
        out = []
        for x1, y1, x2, y2, cf, c in d[d[:, 4] >= conf] if len(d) else []:
            c = int(c)
            name = EWASTE_CLASSES[c] if c < len(EWASTE_CLASSES) else "unknown"
            out.append({"class_id": c, "class_name": name, "confidence": round(float(cf), 4),
                        "bbox_xyxy": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                        "tier": 1 if name in TIER_1_CLASSES else 2})
        return out


# ─── Latency Benchmark ────────────────────────────────────────────────────────
def run_latency_benchmark(inferencer, n_iterations: int = 300,
                           warmup: int = 30, imgsz: int = 640, overlap: int = et.DEFAULT_OVERLAP,
                           frame_budget_ms: float = 100.0) -> dict:
    """
    Measure inference latency distribution on the target device.
    Reports P50, P95, P99, mean, and throughput (FPS).
    """
    print(f"\n{'='*60}")
    print(f"  Latency Benchmark")
    print(f"  Iterations : {n_iterations}")
    print(f"  Warmup     : {warmup}")
    print(f"  Tile size  : {imgsz}x{imgsz} (one model call)")
    print(f"{'='*60}\n")

    dummy_img  = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    dummy_tens = inferencer.preprocess_tile(dummy_img)

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        inferencer.infer(dummy_tens)

    # Benchmark
    latencies = []
    print(f"  Benchmarking ({n_iterations} iterations)...")
    for _ in range(n_iterations):
        _, lat_ms = inferencer.infer_timed(dummy_tens)
        latencies.append(lat_ms)

    latencies = np.array(latencies)
    results = {
        "n_iterations":  n_iterations,
        "mean_ms":       float(np.mean(latencies)),
        "median_ms":     float(np.median(latencies)),
        "p95_ms":        float(np.percentile(latencies, 95)),
        "p99_ms":        float(np.percentile(latencies, 99)),
        "min_ms":        float(np.min(latencies)),
        "max_ms":        float(np.max(latencies)),
        "fps":           float(1000 / np.mean(latencies)),
        "imgsz":         imgsz,
    }

    print(f"\n  Results:")
    print(f"    Mean     : {results['mean_ms']:.2f} ms")
    print(f"    Median   : {results['median_ms']:.2f} ms")
    print(f"    P95      : {results['p95_ms']:.2f} ms")
    print(f"    P99      : {results['p99_ms']:.2f} ms")
    print(f"    FPS      : {results['fps']:.1f}")

    # Batched throughput: the deployed path sends up to max_batch tiles per call
    mb = int(getattr(inferencer, "max_batch", 1) or 1)
    results["max_batch"] = mb
    eff = results["median_ms"]
    if mb > 1:
        xb = np.concatenate([dummy_tens] * mb)
        for _ in range(max(3, warmup // mb)):
            inferencer.infer(xb)
        lb = np.array([inferencer.infer_timed(xb)[1] for _ in range(max(10, n_iterations // mb))])
        results["batch_median_ms"] = float(np.median(lb))
        results["batch_p95_ms"] = float(np.percentile(lb, 95))
        eff = min(eff, results["batch_median_ms"] / mb)
        print(f"    Batch {mb:<3}: {results['batch_median_ms']:.2f} ms/call  ->  {results['batch_median_ms'] / mb:.2f} ms/tile")
    results["per_tile_ms_effective"] = float(eff)
    print(f"    Effective: {eff:.2f} ms/tile  (09_mission_planner.py --bench_report reads this)")

    # Per-FRAME cost: the model runs once per tile, so a frame costs tiles x per-tile latency.
    results["per_frame"] = {}
    print(f"\n  Per-frame cost = tiles/frame x {eff:.1f} ms:")
    print(f"    {'frame':<12}{'native tiles':>13}{'all levels':>12}{'ms (all)':>10}{'ms (native+final)':>19}")
    for (fw, fh) in [(1920, 1080), (3840, 2160), (5472, 3648)]:
        allp = et.plan_tiles(fw, fh, imgsz, overlap)
        nat = [s_ for s_ in allp if s_.level == 0]
        fin = [s_ for s_ in allp if s_.final and s_.level != 0]
        ms_all, ms_nat = len(allp) * eff, (len(nat) + len(fin)) * eff
        results["per_frame"][f"{fw}x{fh}"] = {"tiles_all": len(allp), "tiles_native_plus_final": len(nat) + len(fin),
                                               "ms_all": ms_all, "ms_native_plus_final": ms_nat}
        print(f"    {fw}x{fh:<7}{len(nat):>13}{len(allp):>12}{ms_all:>10.0f}{ms_nat:>19.0f}")
    print("    (model time only; --accuracy measures end-to-end incl. decode + merge)")

    # What frame rate is actually needed is a flight-geometry question, not a camera one
    f4k = results["per_frame"]["3840x2160"]["ms_all"]
    print(f"\n  A 4K frame costs ~{f4k:.0f} ms -> max {1000 / f4k:.2f} frames/s of fully tiled 4K.")
    print("  A survey does not need video rate: at 25 m / 8 m/s a 4K frame must be processed ~0.8 Hz so")
    print("  every object is seen twice (--video does that scheduling).  Check your legs with:")
    print(f"    python 09_mission_planner.py --tile_ms {eff:.1f} --legs <alt:speed> ...")
    if frame_budget_ms and f4k > frame_budget_ms:
        print(f"  (--frame_budget_ms {frame_budget_ms:.0f}: a fully tiled 4K frame does not fit; with --video")
        print("   the budget is per PROCESSED frame, i.e. 1/process_hz, not 1/30 s)")

    return results


# ─── Accuracy Benchmark ───────────────────────────────────────────────────────
def run_accuracy_benchmark(inferencer, test_dir: Path,
                            conf_threshold: float = 0.25, detector: Optional["FrameDetector"] = None) -> dict:
    """
    Compute detection accuracy on the held-out test set.
    Uses COCO-style mAP computation via Ultralytics metrics.

    NOTE: This test set must be geographically diverse and must NOT have
    been seen during training, augmentation, or confidence calibration.
    """
    print(f"\n{'='*60}")
    print(f"  Accuracy Benchmark on Test Set")
    print(f"  Test dir  : {test_dir}")
    print(f"  Conf      : {conf_threshold}")
    print(f"{'='*60}\n")

    img_dir = test_dir / "images"
    lbl_dir = test_dir / "labels"

    # case-insensitive: real DJI frames are .JPG, and glob is case-sensitive on Linux
    image_files = sorted(p for p in img_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_files:
        print(f"  [ERROR] No images found in {img_dir}")
        return {}

    print(f"  Test images: {len(image_files)}")

    # Per-class accumulators
    all_tp = {cls: 0 for cls in EWASTE_CLASSES}
    all_fp = {cls: 0 for cls in EWASTE_CLASSES}
    all_fn = {cls: 0 for cls in EWASTE_CLASSES}

    n_processed  = 0
    total_det_ms = 0

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        detections = detector(img, conf_threshold)
        total_det_ms += detector.last_ms

        # Load ground truth
        lbl_path  = lbl_dir / (img_path.stem + ".txt")
        gt_cls    = []
        gt_bboxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_cls.append(int(parts[0]))
                        gt_bboxes.append([float(x) for x in parts[1:]])

        # Simple per-class TP/FP/FN (IoU > 0.5)
        matched_gt = set()
        for det in detections:
            cls_id   = det["class_id"]
            cls_name = det["class_name"]
            det_box  = det["bbox_xyxy"]

            matched = False
            for gi, (gcls, gbox) in enumerate(zip(gt_cls, gt_bboxes)):
                if gcls != cls_id or gi in matched_gt:
                    continue
                # Convert GT from YOLO to xyxy
                cx, cy, bw, bh = gbox
                gx1 = (cx - bw/2) * w
                gy1 = (cy - bh/2) * h
                gx2 = (cx + bw/2) * w
                gy2 = (cy + bh/2) * h

                iou = _compute_iou(det_box, [gx1, gy1, gx2, gy2])
                if iou >= 0.5:
                    all_tp[cls_name] += 1
                    matched_gt.add(gi)
                    matched = True
                    break

            if not matched:
                if cls_name in all_fp:
                    all_fp[cls_name] += 1

        # FN: GT boxes not matched
        for gi, gcls in enumerate(gt_cls):
            if gi not in matched_gt and 0 <= gcls < len(EWASTE_CLASSES):
                all_fn[EWASTE_CLASSES[gcls]] += 1

        n_processed += 1

    # Compute per-class precision, recall, F1
    results = {"per_class": {}, "overall": {}}
    all_prec, all_rec = [], []

    print(f"\n  Per-class results (IoU@0.5):")
    print(f"  {'Class':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>6} {'Rec':>6}")
    print(f"  {'-'*56}")

    for cls_name in EWASTE_CLASSES:
        tp = all_tp[cls_name]
        fp = all_fp[cls_name]
        fn = all_fn[cls_name]

        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        f1   = 2 * prec * rec / (prec + rec + 1e-6)

        tier = "T1" if cls_name in TIER_1_CLASSES else "  "
        flag = " ⚠" if rec < 0.80 else ""

        print(f"  {tier} {cls_name:<20} {tp:>6} {fp:>6} {fn:>6} "
              f"{prec:>6.3f} {rec:>6.3f}{flag}")

        results["per_class"][cls_name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
        }
        all_prec.append(prec)
        all_rec.append(rec)

    mean_prec = float(np.mean(all_prec))
    mean_rec  = float(np.mean(all_rec))
    mean_f1   = 2 * mean_prec * mean_rec / (mean_prec + mean_rec + 1e-6)
    avg_lat   = total_det_ms / max(n_processed, 1)

    results["overall"] = {
        "mean_precision":  round(mean_prec, 4),
        "mean_recall":     round(mean_rec, 4),
        "mean_f1":         round(mean_f1, 4),
        "n_images":        n_processed,
        "avg_latency_ms":  round(avg_lat, 2),
    }

    print(f"\n  {'Overall':<20} {'':>6} {'':>6} {'':>6} "
          f"{mean_prec:>6.3f} {mean_rec:>6.3f}")
    print(f"\n  Forensic Target Check:")
    print(f"    Recall ≥ {TARGET_RECALL}: "
          f"{'✓ PASS' if mean_rec >= TARGET_RECALL else '✗ FAIL'} ({mean_rec:.4f})")

    return results


def _compute_iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1    = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2    = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


# ─── Confidence Threshold Calibration ─────────────────────────────────────────
def calibrate_confidence_threshold(inferencer, val_dir: Path,
                                    target_recall: float = 0.87,
                                    search_range: tuple = (0.10, 0.70),
                                    detector: Optional["FrameDetector"] = None) -> dict:
    """
    Binary search for the confidence threshold that achieves target_recall
    at the lowest false positive rate.

    For forensic use: we tune for RECALL, not mAP.
    A missed dump is worse than a false alarm.
    The calibrated threshold is what you deploy — NOT the default 0.25.

    Returns dict with optimal threshold and achieved metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Confidence Threshold Calibration")
    print(f"  Target recall : {target_recall}")
    print(f"  Search range  : {search_range}")
    print(f"{'='*60}\n")

    img_dir    = val_dir / "images"
    lbl_dir    = val_dir / "labels"
    img_files  = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    # Pre-run inference at conf=0.001 to get all detections, then filter by threshold
    print(f"  Pre-running inference on {len(img_files)} validation images...")
    all_pred_scores = []   # [(conf, is_tp), ...]
    n_ground_truth  = 0

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w    = img.shape[:2]
        dets    = detector(img, 0.001)

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt_cls, gt_bboxes = [], []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_cls.append(int(parts[0]))
                        gt_bboxes.append([float(x) for x in parts[1:]])
        n_ground_truth += len(gt_cls)

        matched_gt = set()
        for det in sorted(dets, key=lambda x: -x["confidence"]):
            is_tp = False
            for gi, (gcls, gbox) in enumerate(zip(gt_cls, gt_bboxes)):
                if gcls != det["class_id"] or gi in matched_gt:
                    continue
                cx, cy, bw, bh = gbox
                gx1 = (cx - bw/2) * w;  gy1 = (cy - bh/2) * h
                gx2 = (cx + bw/2) * w;  gy2 = (cy + bh/2) * h
                if _compute_iou(det["bbox_xyxy"], [gx1, gy1, gx2, gy2]) >= 0.5:
                    is_tp = True
                    matched_gt.add(gi)
                    break
            all_pred_scores.append((det["confidence"], is_tp))

    # Sort by confidence descending
    all_pred_scores.sort(key=lambda x: -x[0])

    # Sweep thresholds
    print(f"\n  Sweeping thresholds:")
    print(f"  {'Threshold':>10} {'Recall':>8} {'Precision':>10} {'FP Rate':>8}")
    print(f"  {'-'*42}")

    best_threshold = search_range[1]
    best_fp_rate   = 1.0
    sweep_results  = []

    thresholds = np.linspace(search_range[0], search_range[1], 25)
    for thresh in thresholds:
        tp = sum(1 for conf, is_tp in all_pred_scores if conf >= thresh and is_tp)
        fp = sum(1 for conf, is_tp in all_pred_scores if conf >= thresh and not is_tp)
        fn = n_ground_truth - tp

        recall    = tp / (tp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        fp_rate   = fp / (tp + fp + 1e-6)

        sweep_results.append({
            "threshold": round(float(thresh), 3),
            "recall":    round(recall, 4),
            "precision": round(precision, 4),
            "fp_rate":   round(fp_rate, 4),
            "tp": tp, "fp": fp, "fn": fn,
        })

        meets_recall = recall >= target_recall
        flag = " ← candidate" if meets_recall else ""
        print(f"  {thresh:>10.3f} {recall:>8.4f} {precision:>10.4f} {fp_rate:>8.4f}{flag}")

        # Find highest threshold that still meets recall target (lowest FP)
        if meets_recall and fp_rate < best_fp_rate:
            best_threshold = thresh
            best_fp_rate   = fp_rate

    # Result
    best_result = next((r for r in sweep_results
                        if abs(r["threshold"] - best_threshold) < 0.001), sweep_results[0])

    print(f"\n  ✓ Calibrated threshold : {best_threshold:.3f}")
    print(f"    Achieved recall       : {best_result['recall']:.4f}")
    print(f"    Achieved precision    : {best_result['precision']:.4f}")
    print(f"    FP rate               : {best_result['fp_rate']:.4f}")

    return {
        "calibrated_threshold": float(best_threshold),
        "achieved_recall":      best_result["recall"],
        "achieved_precision":   best_result["precision"],
        "fp_rate":              best_result["fp_rate"],
        "target_recall":        target_recall,
        "sweep":                sweep_results,
    }


# ─── Forensic Inference Pipeline ──────────────────────────────────────────────
def run_forensic_inference(inferencer, image_dir: Path, output_dir: Path,
                            conf_threshold: float, model_version: str,
                            mavlink_log: Optional[str] = None,
                            srt_file: Optional[str] = None,
                            detector: Optional["FrameDetector"] = None,
                            altitude_agl: Optional[float] = None,
                            model_sha256: Optional[str] = None,
                            capture_utc_offset: Optional[str] = None):
    """
    Run inference on a directory of drone captures with full forensic output:
      - GPS tagged detections (EXIF / SRT / MAVLink)
      - NTP-synchronized timestamps (UTC)
      - Confidence-scored outputs
      - Immutable hash-chained audit log
      - Per-image annotated visualization
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    log_path = output_dir / "forensic_audit.jsonl"
    logger   = ForensicAuditLogger(log_path, model_version, model_sha256=model_sha256)

    # case-insensitive: DJI writes .JPG and glob is case-sensitive on Linux (2026-10-08)
    image_files = sorted(p for p in image_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"\n  Forensic inference on {len(image_files)} images")
    print(f"  Conf threshold : {conf_threshold}")
    print(f"  Audit log      : {log_path}")

    n_with_detections = 0
    n_tier1_alerts    = 0

    for frame_idx, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w    = img.shape[:2]
        meta    = image_capture_meta(str(img_path), altitude_agl)
        ctx     = capture_context((w, h), meta["altitude_agl"], meta["altitude_source"],
                                  meta["gimbal_pitch"], meta["exposure_s"], meta["speed_mps"])
        gsd     = ctx["gsd_cm_per_px"] / 100 if ctx["gsd_cm_per_px"] else None
        dets    = detector(img, conf_threshold, gsd=gsd)
        ctx.update(tiles=detector.last_tiles, pyramid_levels=detector.last_levels,
                   inference_ms=round(detector.last_ms, 1))

        # GPS extraction — try all sources in priority order
        gps = _extract_gps_from_exif(str(img_path))
        if gps is None and srt_file:
            gps = _extract_gps_from_srt(srt_file, frame_idx)
        if gps is None and mavlink_log:
            gps = _extract_gps_from_mavlink(mavlink_log, time.time())

        # Image hash
        img_hash = get_image_sha256(str(img_path))

        # Log to audit trail
        record = logger.log_detection(
            image_path=str(img_path),
            image_hash=img_hash,
            detections=dets,
            gps=gps,
            altitude_m=gps["altitude_m"] if gps else None,   # GPS altitude as recorded (MSL)
            confidence_threshold=conf_threshold,
            context=ctx,
            capture_time=es.capture_time_from_image(img_path, capture_utc_offset),
        )

        if dets:
            n_with_detections += 1

        tier1 = [d for d in dets if d["class_name"] in TIER_1_CLASSES]
        if tier1:
            n_tier1_alerts += 1
            print(f"  ⚠  TIER-1 ALERT [{img_path.name}]: "
                  f"{[d['class_name'] for d in tier1]}")

        # Draw visualization
        viz = img.copy()
        for det in dets:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            cls_name = det["class_name"]
            conf     = det["confidence"]

            color = (0, 0, 255) if cls_name in TIER_1_CLASSES else (0, 200, 50)
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(viz, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
            cv2.putText(viz, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Overlay GPS + timestamp
        if gps:
            gps_text = f"GPS: {gps['latitude']:.6f}, {gps['longitude']:.6f}"
            if gps.get("altitude_m"):
                gps_text += f" | Alt: {gps['altitude_m']:.1f}m"
            cv2.putText(viz, gps_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        ts_text = f"UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(viz, ts_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        _draw_assessability(viz, ctx)

        cv2.imwrite(str(viz_dir / img_path.name), viz,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Verify chain integrity
    print(f"\n  Verifying audit chain integrity...")
    chain_result = logger.verify_chain()
    if chain_result["valid"]:
        print(f"  ✓ Audit chain valid ({chain_result['records']} records)")
    else:
        print(f"  ✗ CHAIN INTEGRITY FAILURE at record {chain_result.get('broken_at')}")
        print(f"    Reason: {chain_result.get('reason')}")

    print(f"\n  Summary:")
    print(f"    Images processed    : {len(image_files)}")
    print(f"    With detections     : {n_with_detections}")
    print(f"    Tier-1 alerts       : {n_tier1_alerts}")
    print(f"    Audit log           : {log_path}")
    print(f"    Visualizations      : {viz_dir}")

    return {"chain_valid": chain_result["valid"],
            "n_images": len(image_files),
            "n_tier1_alerts": n_tier1_alerts}


def _draw_assessability(viz: np.ndarray, ctx: dict) -> None:
    na = ctx["classes"]["not_assessable"] + ctx["classes"]["unknown"]
    g = ctx["gsd_cm_per_px"]
    txt = (f"GSD {g:.2f} cm/px" if g else "GSD unknown") + \
          (f" | NOT assessable: {', '.join(na)}" if na else " | all classes assessable")
    cv2.putText(viz, txt[:180], (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def run_survey_video(video_path: Path, output_dir: Path, conf_threshold: float, model_version: str,
                     detector: "FrameDetector", srt_file: Optional[str] = None,
                     altitude_agl: Optional[float] = None, views: int = 2, max_object_m: float = 1.0,
                     spacing_frac: Optional[float] = None, max_gap_s: float = 2.0,
                     camera: str = "phantom4pro", save_every: int = 1,
                     model_sha256: Optional[str] = None, capture_utc_offset: Optional[str] = None) -> dict:
    """
    Survey a DJI video: run the tiled detector only on frames that add new ground, so every object
    up to `max_object_m` is fully inside `views` processed frames (ewads_survey.CoverageScheduler).

    Motion source per frame, in priority order: SRT horizontal speed, SRT GPS, image phase
    correlation against the last processed frame.  Unreliable motion -> the frame is processed
    (the scheduler fails towards recall).  Every processed frame gets a hash-chained record with
    the video file's SHA-256, the frame index/time, the SHA-256 of the decoded pixels, why it was
    scheduled, how many frames were skipped before it, and what classes it could assess.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    logger = ForensicAuditLogger(output_dir / "forensic_audit.jsonl", model_version, model_sha256=model_sha256)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_sha = _sha256_file(str(video_path))
    tele = es.parse_dji_srt(srt_file) if srt_file else []
    t_starts = np.array([r.get("t0_s", i / fps) for i, r in enumerate(tele)]) if tele else None
    cam = es.CAMERAS[camera]
    decoder = f"opencv {cv2.__version__} {cap.getBackendName() if hasattr(cap, 'getBackendName') else ''}".strip()

    def telemetry(t):
        if not tele:
            return {}
        return tele[max(0, int(np.searchsorted(t_starts, t, side="right")) - 1)]

    # spacing: from the plan if the altitude is known up front, else a conservative frame fraction
    if spacing_frac is None:
        alt0 = altitude_agl or telemetry(0.0).get("rel_alt_m")
        if alt0:
            fh = H * es.gsd_m(alt0, W, cam)
            spacing_frac = max(0.05, es.capture_spacing_m(fh, max_object_m, views) / fh)
        else:
            spacing_frac = 0.4 / max(views, 1)
    sched = es.CoverageScheduler(H, spacing_frac, max_gap_s=max_gap_s)
    print(f"\n  Survey video  : {video_path.name}  {W}x{H} @ {fps:.2f} fps, {n_total} frames")
    print(f"  SHA-256       : {video_sha[:16]}...")
    print(f"  Telemetry     : {len(tele)} SRT records" if tele else "  Telemetry     : none (image motion)")
    print(f"  Scheduling    : process when ground moved >= {spacing_frac:.0%} of frame height "
          f"(views={views}, objects <= {max_object_m} m), at least every {max_gap_s:.1f} s")

    idx, prev_t, skipped, n_proc, n_tier1, proc_ms = -1, None, 0, 0, 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        t = idx / fps
        tm = telemetry(t)
        alt = altitude_agl or tm.get("rel_alt_m")
        gsd = es.gsd_m(alt, W, cam) if alt else None
        dt = None if prev_t is None else t - prev_t
        prev_t = t
        gps = (tm["lat"], tm["lon"]) if "lat" in tm else None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ("speed_mps" not in tm and gps is None) else None
        go, why = sched.update(t, gray=gray, gsd=gsd, speed_mps=tm.get("speed_mps"), gps=gps, dt_s=dt)
        if not go:
            skipped += 1
            continue
        if sched.last_gray is None:            # keep a reference so image motion can take over
            sched.last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # if telemetry drops out
        ctx = capture_context((W, H), alt, "operator (--altitude_agl)" if altitude_agl else "dji_srt H (above take-off)",
                              None, tm.get("exposure_s"), tm.get("speed_mps"), camera)
        dets = detector(frame, conf_threshold, gsd=gsd)
        proc_ms.append(detector.last_ms)
        ctx.update(video_sha256=video_sha, frame_index=idx, frame_time_s=round(t, 3), decoder=decoder,
                   frame_pixels_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                   scheduled_because=why, frames_skipped_before=skipped,
                   schedule={"views": views, "max_object_m": max_object_m, "spacing_frac": round(spacing_frac, 4)},
                   tiles=detector.last_tiles, pyramid_levels=detector.last_levels,
                   inference_ms=round(detector.last_ms, 1))
        logger.log_detection(image_path=f"{video_path}#frame={idx}", image_hash=video_sha, detections=dets,
                             gps={"latitude": tm["lat"], "longitude": tm["lon"], "source": "dji_srt"} if gps else None,
                             altitude_m=alt, confidence_threshold=conf_threshold, context=ctx,
                             capture_time=es.capture_time_from_srt(tm, capture_utc_offset))
        skipped = 0
        n_proc += 1
        if any(d["class_name"] in TIER_1_CLASSES for d in dets):
            n_tier1 += 1
        if dets or n_proc % max(1, save_every) == 0:
            viz = frame.copy()
            for d in dets:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                col = (0, 0, 255) if d["class_name"] in TIER_1_CLASSES else (0, 200, 50)
                cv2.rectangle(viz, (x1, y1), (x2, y2), col, 2)
                cv2.putText(viz, f"{d['class_name']} {d['confidence']:.2f}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            _draw_assessability(viz, ctx)
            cv2.imwrite(str(viz_dir / f"frame_{idx:06d}.jpg"), viz, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cap.release()
    chain = logger.verify_chain()
    dur = (idx + 1) / fps if idx >= 0 else 0
    res = {"frames_total": idx + 1, "frames_processed": n_proc, "processed_hz": n_proc / dur if dur else 0,
           "mean_ms_per_processed_frame": float(np.mean(proc_ms)) if proc_ms else None,
           "realtime_duty": (float(np.sum(proc_ms)) / 1000 / dur) if dur and proc_ms else None,
           "tier1_frames": n_tier1, "chain_valid": chain["valid"], "video_sha256": video_sha}
    print(f"\n  Processed {n_proc}/{idx + 1} frames ({res['processed_hz']:.2f} Hz of {dur:.1f} s video)")
    if res["realtime_duty"] is not None:
        print(f"  Compute duty  : {res['realtime_duty']:.0%} of real time "
              f"({'keeps up' if res['realtime_duty'] < 1 else 'FALLS BEHIND -- slow down or views=1'})")
    print(f"  Audit chain   : {'valid' if chain['valid'] else 'BROKEN'}  ({output_dir / 'forensic_audit.jsonl'})")
    return res


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="EWADS edge benchmark and forensic inference pipeline"
    )
    parser.add_argument("--mode",          type=str, default="onnx",
                        choices=["onnx", "trt"],
                        help="Inference backend")
    parser.add_argument("--onnx_path",     type=str, default=None)
    parser.add_argument("--engine_path",   type=str, default=None,
                        help="TensorRT .trt engine path")
    parser.add_argument("--test_dir",      type=str,
                        default="./data/test",
                        help="Held-out test set (never seen during training)")
    parser.add_argument("--val_dir",       type=str,
                        default="./data/augmented/val",
                        help="Validation set for threshold calibration")
    parser.add_argument("--output_dir",    type=str,
                        default="./benchmark_results")
    parser.add_argument("--conf",          type=float, default=None,
                        help="Override confidence threshold (uses calibrated if None)")
    parser.add_argument("--target_recall", type=float, default=0.87)
    parser.add_argument("--benchmark",     action="store_true",
                        help="Run latency benchmark")
    parser.add_argument("--calibrate",     action="store_true",
                        help="Run confidence threshold calibration")
    parser.add_argument("--accuracy",      action="store_true",
                        help="Run accuracy benchmark on test set")
    parser.add_argument("--forensic",      action="store_true",
                        help="Run forensic inference pipeline on test_dir")
    parser.add_argument("--verify_chain",  action="store_true",
                        help="Verify existing audit log chain integrity")
    parser.add_argument("--audit_log",     type=str, default=None)
    parser.add_argument("--model_version", type=str, default="ewads-v1.0.0")
    parser.add_argument("--mavlink_log",   type=str, default=None)
    parser.add_argument("--srt_file",      type=str, default=None)
    parser.add_argument("--tile",          type=int, default=None,
                        help="tile size = model input size (default: read from the model)")
    parser.add_argument("--overlap",       type=int, default=et.DEFAULT_OVERLAP)
    parser.add_argument("--no_tiled",      action="store_true",
                        help="legacy: squash the whole frame to one square (loses small objects)")
    parser.add_argument("--max_levels",    type=int, default=None,
                        help="cheaper: only the first N pyramid levels + the final whole-frame one")
    parser.add_argument("--frame_budget_ms", type=float, default=100.0)
    parser.add_argument("--all",           action="store_true",
                        help="Run all benchmarks in sequence")
    # survey / capture context (2026-10-02)
    parser.add_argument("--altitude_agl",  type=float, default=None,
                        help="height above ground (m) if XMP/SRT altitude is missing or wrong (sloping site)")
    parser.add_argument("--auto_levels",   action="store_true",
                        help="per-frame pyramid levels from GSD (skips coarse levels at altitude)")
    parser.add_argument("--largest_object_m", type=float, default=3.5)
    parser.add_argument("--video",         type=str, default=None,
                        help="survey a video: process only frames that add ground (needs --srt_file or image motion)")
    parser.add_argument("--views",         type=int, default=2)
    parser.add_argument("--max_object_m",  type=float, default=1.0)
    parser.add_argument("--spacing_frac",  type=float, default=None,
                        help="override: process when ground moved this fraction of frame height")
    parser.add_argument("--max_gap_s",     type=float, default=2.0)
    parser.add_argument("--capture_utc_offset", type=str, default=None,
                        help="UTC offset of the camera clock, e.g. +04:00, used ONLY when the file has no "
                             "offset of its own; every such record says the offset was operator-asserted")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify chain only
    if args.verify_chain:
        log_path = Path(args.audit_log or "./benchmark_results/forensic_audit.jsonl")
        logger   = ForensicAuditLogger(log_path, "verify-only", verify_on_open=False)   # report, don't refuse
        result   = logger.verify_chain()
        status   = "✓ VALID" if result["valid"] else "✗ INVALID"
        print(f"\n  Chain status: {status}")
        print(json.dumps(result, indent=2))
        return

    # Load inferencer
    if args.mode == "trt":
        if not args.engine_path:
            print("[ERROR] --engine_path required for TRT mode")
            sys.exit(1)
        inferencer = TensorRTInferencer(Path(args.engine_path))
    else:
        if not args.onnx_path:
            print("[ERROR] --onnx_path required for ONNX mode")
            sys.exit(1)
        inferencer = ONNXInferencer(Path(args.onnx_path))

    all_results = {}
    model_file = Path(args.engine_path if args.mode == "trt" else args.onnx_path)
    model_sha256 = _sha256_file(str(model_file))        # the file actually loaded -> every audit record
    print(f"  Model file    : {model_file}  sha256 {model_sha256[:16]}...")
    all_results["model"] = {"path": str(model_file), "sha256": model_sha256}

    tile = args.tile or getattr(inferencer, "input_size", None) or 640
    tiled = not args.no_tiled
    if tiled and getattr(inferencer, "input_size", None) and tile != inferencer.input_size:
        print(f"[ERROR] --tile {tile} != model input {inferencer.input_size}"); sys.exit(1)
    detector = FrameDetector(inferencer, tile, args.overlap, tiled, args.max_levels,
                             auto_levels=args.auto_levels, largest_object_m=args.largest_object_m)
    print(f"\n  Inference path: {'TILED  tile %d  overlap %d' % (tile, args.overlap) if tiled else 'LEGACY full-frame squash (small objects will be lost)'}")
    all_results["inference"] = {"tiled": tiled, "tile": tile, "overlap": args.overlap,
                                "max_levels": args.max_levels}
    audit_version = f"{args.model_version}|tile{tile}|overlap{args.overlap}|levels{args.max_levels or ('auto' if args.auto_levels else 'all')}" \
        if tiled else f"{args.model_version}|fullframe-legacy"

    # Latency benchmark
    if args.benchmark or args.all:
        lat_results = run_latency_benchmark(inferencer, imgsz=tile, overlap=args.overlap,
                                            frame_budget_ms=args.frame_budget_ms)
        all_results["latency"] = lat_results

    # Confidence calibration
    conf_threshold = args.conf
    if args.calibrate or args.all or conf_threshold is None:
        val_dir = Path(args.val_dir)
        if val_dir.exists():
            cal_results    = calibrate_confidence_threshold(
                inferencer, val_dir, target_recall=args.target_recall, detector=detector
            )
            all_results["calibration"] = cal_results
            if conf_threshold is None:
                conf_threshold = cal_results["calibrated_threshold"]
                print(f"\n  Using calibrated threshold: {conf_threshold:.3f}")
        else:
            print(f"  [SKIP] Val dir not found: {val_dir}")
            conf_threshold = conf_threshold or 0.25

    # Accuracy benchmark
    if args.accuracy or args.all:
        test_dir = Path(args.test_dir)
        if test_dir.exists():
            acc_results = run_accuracy_benchmark(inferencer, test_dir, conf_threshold, detector=detector)
            all_results["accuracy"] = acc_results
        else:
            print(f"  [SKIP] Test dir not found: {args.test_dir}")

    # Forensic inference
    if args.forensic or args.all:
        test_dir = Path(args.test_dir)
        if test_dir.exists():
            for_results = run_forensic_inference(
                inferencer=inferencer,
                image_dir=test_dir / "images",
                output_dir=output_dir / "forensic_output",
                conf_threshold=conf_threshold or 0.25,
                model_version=audit_version,
                mavlink_log=args.mavlink_log,
                srt_file=args.srt_file,
                detector=detector,
                altitude_agl=args.altitude_agl,
                model_sha256=model_sha256,
                capture_utc_offset=args.capture_utc_offset,
            )
            all_results["forensic"] = for_results

    # Survey a video with coverage scheduling
    if args.video:
        all_results["survey"] = run_survey_video(
            Path(args.video), output_dir / "survey_output", conf_threshold or 0.25, audit_version + "|survey",
            detector, srt_file=args.srt_file, altitude_agl=args.altitude_agl, views=args.views,
            max_object_m=args.max_object_m, spacing_frac=args.spacing_frac, max_gap_s=args.max_gap_s,
            model_sha256=model_sha256, capture_utc_offset=args.capture_utc_offset)

    # Save full results
    results_path = output_dir / "benchmark_report.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Benchmark complete. Report: {results_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
