"""
EWADS survey physics: what the camera can resolve, how fast the drone may fly, and how often a
frame actually has to be processed.

Why this exists
  Tiling made small objects detectable but costs ~33 model calls per 4K frame (74 per 20 MP).
  Running that at video rate is impossible on an Orin NX, and it is also pointless: at survey
  speed consecutive video frames overlap by >95%.  What enforcement needs is that every patch of
  ground is SEEN (fully, k times) at a GSD where the target class is resolvable.  That sets the
  processing rate, not the camera's frame rate.

  The same geometry answers the tiny-object problem honestly: a 3 cm chip cannot be detected from
  20 m, so the system must (a) plan altitude per target class and (b) record, in the forensic log,
  which classes were and were not assessable in every frame.  A "no detection" for a class that
  was not assessable is NOT evidence of absence.

Everything here is pure numpy / OpenCV, no torch, so it runs on the Jetson and in tests.

Numbers that must stay consistent with the renderer (02_blenderproc_synthetic_v2.py) are checked
by tests/test_survey.py (it parses 02 with `ast`, without importing blenderproc).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ───────────────────────────── camera model ──────────────────────────────────
@dataclass(frozen=True)
class Camera:
    name: str
    sensor_w_mm: float
    sensor_h_mm: float
    focal_mm: float
    modes: Dict[str, Tuple[int, int]]          # name -> (W, H) in px
    # fraction of the sensor WIDTH a mode reads out (1.0 = full width).  The renderer assumes
    # 1.0 for every mode; if your firmware crops 4K video, measure it (see verify_gsd in HANDOFF)
    # and set it here, because GSD scales with it.
    width_crop: Dict[str, float] = field(default_factory=dict)

    def hfov_deg(self, mode: str) -> float:
        w = self.sensor_w_mm * self.width_crop.get(mode, 1.0)
        return math.degrees(2 * math.atan(w / 2 / self.focal_mm))


CAMERAS = {
    # DJI Phantom 4 Pro: 1" CMOS 13.2 x 8.8 mm, fixed 8.8 mm lens (same numbers as 02's CAMERA_PROFILES)
    "phantom4pro": Camera(
        "phantom4pro", 13.2, 8.8, 8.8,
        modes={"stills_3x2": (5472, 3648), "stills_16x9": (5472, 3078),
               "video_4k": (3840, 2160), "video_c4k": (4096, 2160), "video_1080p": (1920, 1080)}),
}


def gsd_m(altitude_m: float, width_px: int, cam: Camera, mode: Optional[str] = None) -> float:
    """Ground sample distance (m/px) for a NADIR pinhole camera over flat ground."""
    w_mm = cam.sensor_w_mm * (cam.width_crop.get(mode, 1.0) if mode else 1.0)
    return altitude_m * w_mm / (cam.focal_mm * width_px)


def footprint_m(altitude_m: float, mode: str, cam: Camera) -> Tuple[float, float]:
    W, H = cam.modes[mode]
    g = gsd_m(altitude_m, W, cam, mode)
    return W * g, H * g


# ───────────────────────────── class sizes ───────────────────────────────────
# Mirrors 02's CLASS_REAL_SIZE_M (nominal longest side, m) -- checked by tests/test_survey.py.
CLASS_REAL_SIZE_M = {
    "crt_monitor": 0.45, "flat_screen": 0.60, "circuit_board": 0.25,
    "battery": 0.25, "cable_bundle": 0.50, "single_cable": 1.00,
    "mobile_phone": 0.15, "keyboard": 0.45, "printer": 0.50,
    "small_appliance": 0.35, "large_appliance": 1.70, "cpu_chip": 0.04,
    "hard_drive": 0.15, "power_supply": 0.16, "ewaste_pile": 2.50,
    "plastic_casing": 0.40, "metal_component": 0.30, "lighting": 1.20,
}
EWASTE_CLASSES = list(CLASS_REAL_SIZE_M)          # same order as 02/03/05/07
SCALE_JITTER_LO = 0.7                              # 02's DEFAULT_SCALE_JITTER[0]
# The SMALLEST realistic instance the renderer produces (02's CLASS_SIZE_BY_SHAPE small branch
# x its jitter low end).  Assessability is judged on these: claiming "no batteries here" while an
# AA cell would be 5 px is exactly the error we must not make in court.
SMALL_VARIANT_M = {"battery": 0.055 * 0.8, "lighting": 0.12 * 0.8}
CLASS_MIN_FEATURE_M = {"single_cable": 0.005}      # 02: thin-part width that must reach MIN_FEATURE_PX

ASSESS_PX = 16       # = 02's SPAWN_MIN_PX: the renderer never shows the model anything smaller
MARGINAL_PX = 8      # = ewads_tiling.MIN_OBJ_PX: labelled in training, but AP there is low
MIN_FEATURE_PX = 1.0


def class_small_size_m(cls: str) -> float:
    return SMALL_VARIANT_M.get(cls, CLASS_REAL_SIZE_M[cls] * SCALE_JITTER_LO)


def class_nominal_size_m(cls: str) -> float:
    return CLASS_REAL_SIZE_M[cls]


def resolvability(gsd: Optional[float], classes: Sequence[str] = EWASTE_CLASSES,
                  assess_px: float = ASSESS_PX, marginal_px: float = MARGINAL_PX) -> dict:
    """
    Per class at this GSD: px of the smallest realistic instance and a status
      assessable      smallest instance >= assess_px (and thin parts >= 1 px)
      marginal        >= marginal_px: detections are meaningful, absence is not
      not_assessable  below that: the class cannot be judged in this frame
    gsd None -> every class 'unknown' (no altitude: never claim assessability without it).
    """
    out = {}
    for c in classes:
        if gsd is None or gsd <= 0:
            out[c] = {"status": "unknown", "px_small": None}
            continue
        px = class_small_size_m(c) / gsd
        feat = CLASS_MIN_FEATURE_M.get(c)
        feat_ok = feat is None or feat / gsd >= MIN_FEATURE_PX
        if px >= assess_px and feat_ok:
            st = "assessable"
        elif px >= marginal_px and feat_ok:
            st = "marginal"
        else:
            st = "not_assessable"
        out[c] = {"status": st, "px_small": round(px, 1)}
    return out


def resolvability_summary(gsd: Optional[float]) -> dict:
    r = resolvability(gsd)
    by = {"assessable": [], "marginal": [], "not_assessable": [], "unknown": []}
    for c, v in r.items():
        by[v["status"]].append(c)
    return by


def max_altitude_m(cls: str, mode: str, cam: Camera, px: float = ASSESS_PX,
                   size: str = "small") -> float:
    """Highest altitude at which `cls` (smallest or nominal instance) still spans `px` pixels
    (and its thin parts >= 1 px)."""
    W, _ = cam.modes[mode]
    per_m = gsd_m(1.0, W, cam, mode)                 # gsd per metre of altitude
    s = class_small_size_m(cls) if size == "small" else class_nominal_size_m(cls)
    alt = s / (px * per_m)
    feat = CLASS_MIN_FEATURE_M.get(cls)
    if feat:
        alt = min(alt, feat / (MIN_FEATURE_PX * per_m))
    return alt


# ───────────────────────────── motion & coverage ─────────────────────────────
def motion_blur_px(speed_mps: float, exposure_s: float, gsd: float) -> float:
    return speed_mps * exposure_s / gsd


def max_speed_mps(gsd: float, exposure_s: float, blur_px: float = 1.0) -> float:
    return blur_px * gsd / exposure_s


def capture_spacing_m(along_track_m: float, max_object_m: float, views: int) -> float:
    """Ground distance between processed frames so that every object up to `max_object_m`
    lies COMPLETELY inside at least `views` processed frames.  <=0 means impossible."""
    return (along_track_m - max_object_m) / max(views, 1)


def sequence_offsets_m(n_frames: int, along_track_m: float, cross_track_m: float, yaw_deg: float,
                       views: int = 3, max_object_m: float = 1.0,
                       terrain_half_side_m: Optional[float] = None) -> dict:
    """Camera positions for a short overlapping synthetic SEQUENCE (added 2026-10-08), so that
    11_evidence_pipeline.py can be validated and its stage thresholds fixed on synthetic data.

    Frames are spaced the way 11's CoverageScheduler would process them: spacing =
    capture_spacing_m(along_track, max_object_m, views), floored at 5% of the along-track footprint
    (same floor as 11's iter_video).  Offsets are centred on the scene origin (the middle frame is the
    scene's own view) and run along the image's vertical axis rotated by the camera yaw: a nadir
    camera with yaw psi has its image 'up' along world (-sin psi, cos psi).
    -> {"offsets": [(dx, dy), ...] metres, "spacing_m", "overlap_frac", "fits": bool, "max_reach_m"}
    `fits` is False if a frame corner would leave a square terrain of half-side terrain_half_side_m."""
    fh = float(along_track_m)
    spacing = max(0.05 * fh, capture_spacing_m(fh, max_object_m, views))
    ux, uy = -math.sin(math.radians(yaw_deg)), math.cos(math.radians(yaw_deg))
    centre = (n_frames - 1) / 2.0
    offs = [((k - centre) * spacing * ux, (k - centre) * spacing * uy) for k in range(n_frames)]
    reach = centre * spacing + 0.5 * math.hypot(along_track_m, cross_track_m)   # worst case: any yaw
    fits = True if terrain_half_side_m is None else reach <= terrain_half_side_m
    return {"offsets": offs, "spacing_m": spacing, "overlap_frac": max(0.0, 1.0 - spacing / fh),
            "fits": fits, "max_reach_m": reach}


def levels_needed(gsd: float, frame_wh: Tuple[int, int], tile: int, overlap: int,
                  max_object_m: float) -> int:
    """
    How many leading pyramid levels the tiler must run so the largest expected object is whole in
    some tile (an object whose size at level l is <= overlap is always whole in a level-l tile).
    Returned value plugs into TiledDetector(max_levels=...), which always adds the final
    (whole-frame) level as cheap insurance against anything bigger than expected.
    """
    from ewads_tiling import pyramid_scales
    W, H = frame_wh
    scales = pyramid_scales(W, H, tile)
    px = max_object_m / gsd
    for i, s in enumerate(scales):
        if px * s <= overlap:
            return i + 1
    return len(scales)


def tiles_per_frame(frame_wh: Tuple[int, int], tile: int, overlap: int,
                    max_levels: Optional[int] = None) -> int:
    from ewads_tiling import plan_tiles
    specs = plan_tiles(frame_wh[0], frame_wh[1], tile, overlap)
    if max_levels:
        keep = {s.level for s in specs if s.level < max_levels} | {s.level for s in specs if s.final}
        specs = [s for s in specs if s.level in keep]
    return len(specs)


@dataclass
class MissionPlan:
    mode: str
    altitude_m: float
    gsd_cm: float
    footprint_m: Tuple[float, float]
    speed_mps: float
    speed_limit_blur_mps: float
    blur_px: float
    spacing_m: float
    process_hz: float
    max_levels: int
    tiles_per_frame: int
    tiles_per_s: float
    tile_ms: Optional[float]
    duty_cycle: Optional[float]
    area_rate_ha_per_h: float
    resolvability: dict
    warnings: List[str]


def plan_mission(cam: Camera, mode: str, altitude_m: float, speed_mps: float,
                 exposure_s: float = 1 / 1000, views: int = 2, max_object_m: float = 1.0,
                 side_overlap: float = 0.2, tile: int = 768, overlap: int = 128,
                 tile_ms: Optional[float] = None, blur_limit_px: float = 1.0,
                 largest_object_m: float = 3.5) -> MissionPlan:
    """
    One flight leg.  Assumes nadir camera, image TOP = flight direction (DJI default), so the
    along-track footprint is the image HEIGHT.
      views          processed frames that must fully contain each object (2 = one retry against
                     blur / glare / a missed detection; 1 = cheapest)
      max_object_m   largest object that must be FULLY inside `views` frames (piles are surveyed
                     from higher up, so 1 m covers every item class except piles/large appliances)
      tile_ms        measured per-tile latency on the target (07 --benchmark); None = unknown
    """
    W, H = cam.modes[mode]
    g = gsd_m(altitude_m, W, cam, mode)
    fw, fh = W * g, H * g
    warn = []
    vmax = max_speed_mps(g, exposure_s, blur_limit_px)
    blur = motion_blur_px(speed_mps, exposure_s, g)
    if blur > blur_limit_px:
        warn.append(f"motion blur {blur:.1f} px > {blur_limit_px} px: fly <= {vmax:.1f} m/s or use a "
                    f"shorter shutter (blur smears exactly the small objects this altitude is for)")
    sp = capture_spacing_m(fh, max_object_m, views)
    if sp <= 0:
        warn.append(f"footprint along track ({fh:.1f} m) too short to contain a {max_object_m} m object "
                    f"{views}x: fly higher or lower max_object_m")
        hz = float("inf")
    else:
        hz = speed_mps / sp
    lv = levels_needed(g, (W, H), tile, overlap, largest_object_m)
    tpf = tiles_per_frame((W, H), tile, overlap, lv)
    tps = hz * tpf if math.isfinite(hz) else float("inf")
    duty = tps * tile_ms / 1000 if (tile_ms and math.isfinite(tps)) else None
    if duty is not None and duty > 0.8:
        warn.append(f"compute duty {duty:.0%} > 80%: slow to "
                    f"{speed_mps * 0.8 / duty:.1f} m/s, use views=1, or a smaller/INT8 model")
    swath = fw * (1 - side_overlap)
    return MissionPlan(mode, altitude_m, g * 100, (fw, fh), speed_mps, vmax, blur, sp, hz, lv, tpf, tps,
                       tile_ms, duty, speed_mps * swath * 3600 / 1e4, resolvability_summary(g), warn)


# ───────────────────────────── DJI metadata ──────────────────────────────────
_XMP_KEYS = {
    "RelativeAltitude": "rel_alt_m", "AbsoluteAltitude": "abs_alt_m",
    "GimbalPitchDegree": "gimbal_pitch_deg", "GimbalYawDegree": "gimbal_yaw_deg",
    "GimbalRollDegree": "gimbal_roll_deg", "FlightYawDegree": "flight_yaw_deg",
    "FlightXSpeed": "vx_mps", "FlightYSpeed": "vy_mps", "FlightZSpeed": "vz_mps",
}


def read_dji_xmp(path: str | Path, max_bytes: int = 256 * 1024) -> dict:
    """
    DJI writes flight data as XMP (drone-dji:RelativeAltitude="+12.30" ...) near the start of the
    JPEG.  RelativeAltitude is height above the TAKE-OFF point (barometric), which is the right
    input for GSD over flat ground; EXIF GPSAltitude is above sea level / ellipsoid and is NOT
    (using it would give a nonsense GSD anywhere above sea level).
    Returns {} when absent.  Both attribute (k="v") and element (<k>v</k>) forms are handled.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read(max_bytes).decode("latin-1", "ignore")
    except OSError:
        return {}
    out = {}
    for k, name in _XMP_KEYS.items():
        m = re.search(r"drone-dji:%s\s*=\s*\"([+\-]?[\d.]+)\"" % k, raw) or \
            re.search(r"<drone-dji:%s>\s*([+\-]?[\d.]+)\s*<" % k, raw)
        if m:
            out[name] = float(m.group(1))
    return out


def read_exposure_s(path: str | Path) -> Optional[float]:
    try:
        from PIL import Image
        ex = Image.open(path).getexif()
        v = ex.get_ifd(0x8769).get(0x829A) if hasattr(ex, "get_ifd") else None   # ExposureTime
        return float(v) if v else None
    except Exception:
        return None


# ─── capture time (forensic hardening, 2026-10-08) ──────────────────────────────
# Audit records carry WHEN THE PIXELS WERE CAPTURED, separately from timestamp_utc (= when the record
# was processed).  A UTC capture time is only claimed when the source carries a UTC offset (or the
# operator asserts the camera clock's offset, which the source string then says).  Camera-local
# times without a zone are kept in capture_time_local, never silently converted.
CAPTURE_TIME_UNAVAILABLE = {"capture_time_utc": None, "capture_time_local": None,
                            "capture_time_source": "unavailable"}
_DT_RE = re.compile(r"(\d{4})[-:](\d{2})[-:](\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:[.,](\d{1,6}))?"
                    r"\s*(Z|[+\-]\d{2}:?\d{2})?")
_XMP_DATE_RE = re.compile(r"(?:xmp:CreateDate|exif:DateTimeOriginal|photoshop:DateCreated)"
                          r"\s*(?:=\s*\"([^\"]+)\"|>\s*([^<]+)<)")


def _parse_offset(off: Optional[str]):
    """'+04:00' / '+0400' / '-05:30' / 'Z' -> datetime.timezone, None if absent/invalid."""
    from datetime import timedelta, timezone
    if not off:
        return None
    off = off.strip()
    if off.upper() == "Z":
        return timezone.utc
    m = re.fullmatch(r"([+\-])(\d{2}):?(\d{2})", off)
    if not m:
        return None
    sign = 1 if m.group(1) == "+" else -1
    return timezone(sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3))))


def _capture_record(text: str, source: str, offset: Optional[str] = None,
                    assume_offset: Optional[str] = None) -> Optional[dict]:
    """Date-time text (+ optional separate offset) -> capture-time fields, or None if unparseable."""
    from datetime import datetime, timezone
    m = _DT_RE.search(text or "")
    if not m:
        return None
    y, mo, d, h, mi, s_, frac, inline_off = m.groups()
    us = int((frac or "0").ljust(6, "0")[:6])
    try:
        local = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s_), us)
    except ValueError:
        return None
    tz = _parse_offset(inline_off) or _parse_offset(offset)
    src = source
    if tz is None and assume_offset:
        tz = _parse_offset(assume_offset)
        if tz is not None:
            src += f"; UTC offset {assume_offset} ASSERTED by operator (not in the file)"
    if tz is None:
        return {"capture_time_utc": None, "capture_time_local": local.isoformat(),
                "capture_time_source": src + "; no UTC offset in source: camera-local time, zone unknown"}
    return {"capture_time_utc": local.replace(tzinfo=tz).astimezone(timezone.utc).isoformat(),
            "capture_time_local": local.isoformat(), "capture_time_source": src}


def capture_time_from_image(path: str | Path, assume_offset: Optional[str] = None) -> dict:
    """EXIF DateTimeOriginal (+SubSecTimeOriginal, +OffsetTimeOriginal) or DJI/XMP CreateDate.
    Preference: a source WITH a UTC offset, then EXIF local, then XMP local, else 'unavailable'."""
    cands = []
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif().get_ifd(0x8769)
        dt, ss, off = ex.get(0x9003), ex.get(0x9291), ex.get(0x9011)
        if dt:
            txt = str(dt).strip() + (f".{str(ss).strip()}" if ss else "")
            r = _capture_record(txt, "exif DateTimeOriginal" + ("+OffsetTimeOriginal" if off else ""),
                                offset=str(off) if off else None, assume_offset=assume_offset)
            if r:
                cands.append(r)
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            raw = f.read(256 * 1024).decode("latin-1", "ignore")
        m = _XMP_DATE_RE.search(raw)
        if m:
            r = _capture_record(m.group(1) or m.group(2), "xmp CreateDate", assume_offset=assume_offset)
            if r:
                cands.append(r)
    except OSError:
        pass
    for r in cands:
        if r["capture_time_utc"]:
            return r
    return cands[0] if cands else dict(CAPTURE_TIME_UNAVAILABLE)


def capture_time_from_srt(rec: Optional[dict], assume_offset: Optional[str] = None) -> dict:
    """Per-frame date-time from a parse_dji_srt() record (newer DJI SRT lines carry
    'YYYY-MM-DD hh:mm:ss,mmm'; the Phantom 4 Pro SRT does not)."""
    if rec and rec.get("datetime_text"):
        r = _capture_record(rec["datetime_text"], "dji_srt per-frame datetime", assume_offset=assume_offset)
        if r:
            return r
    return dict(CAPTURE_TIME_UNAVAILABLE)


_SRT_NUM = r"([+\-]?\d+(?:\.\d+)?)"


def parse_dji_srt(path: str | Path) -> List[dict]:
    """
    Per-subtitle telemetry from a DJI .SRT.  Formats differ by aircraft/firmware; this accepts the
    common ones and records what it found:
      Phantom 4 Pro:  'F/2.8, SS 320, ISO 100, EV 0, GPS (lon, lat, 19), D 25.88m, H 12.40m,
                       H.S 3.20m/s, V.S 0.00m/s'
      newer:          '[latitude: 25.2] [longitude: 55.3] [rel_alt: 12.4 abs_alt: 30.1] [shutter: 1/1000.0]'
    Returns [{"idx", "t0_s", "t1_s", "lat", "lon", "rel_alt_m", "speed_mps", "exposure_s"}...]
    (missing fields omitted).
    """
    txt = Path(path).read_text(encoding="utf-8", errors="ignore").replace("\r", "")
    out = []
    for blk in re.split(r"\n\s*\n", txt.strip()):
        lines = blk.strip().split("\n")
        if len(lines) < 2:
            continue
        rec: dict = {}
        try:
            rec["idx"] = int(lines[0].strip())
        except ValueError:
            pass
        tm = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", blk)
        if tm:
            a = [int(x) for x in tm.groups()]
            rec["t0_s"] = a[0] * 3600 + a[1] * 60 + a[2] + a[3] / 1000
            rec["t1_s"] = a[4] * 3600 + a[5] * 60 + a[6] + a[7] / 1000
        body = " ".join(lines[1:])
        dtm = _DT_RE.search(body)                         # newer firmware: '2023-05-01 10:20:30,123,456'
        if dtm:
            rec["datetime_text"] = dtm.group(0).strip()
        m = re.search(r"GPS\s*\(\s*%s\s*,\s*%s\s*,\s*%s" % (_SRT_NUM, _SRT_NUM, _SRT_NUM), body)
        if m:                                            # P4P order: lon, lat, (gps alt)
            rec["lon"], rec["lat"] = float(m.group(1)), float(m.group(2))
        la = re.search(r"latitude\s*[:=]\s*%s" % _SRT_NUM, body, re.I)
        lo = re.search(r"longitude\s*[:=]\s*%s" % _SRT_NUM, body, re.I)
        if la and lo:
            rec["lat"], rec["lon"] = float(la.group(1)), float(lo.group(1))
        h = re.search(r"(?:^|[\s,\]])H\s+%s\s*m" % _SRT_NUM, body) or \
            re.search(r"rel_alt\s*[:=]\s*%s" % _SRT_NUM, body, re.I)
        if h:
            rec["rel_alt_m"] = float(h.group(1))
        hs = re.search(r"H\.S\s+%s\s*m/s" % _SRT_NUM, body)
        if hs:
            rec["speed_mps"] = abs(float(hs.group(1)))
        ss = re.search(r"shutter\s*[:=]\s*1/%s" % _SRT_NUM, body, re.I) or \
            re.search(r"\bSS\s+%s" % _SRT_NUM, body)
        if ss and float(ss.group(1)) > 0:
            rec["exposure_s"] = 1.0 / float(ss.group(1))      # P4P 'SS 320' means 1/320 s
        out.append(rec)
    return out


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ───────────────────────────── image-based motion ────────────────────────────
def _thumb(gray: np.ndarray, width: int = 320) -> Tuple[np.ndarray, float]:
    import cv2
    s = width / gray.shape[1]
    t = cv2.resize(gray, (width, max(8, int(round(gray.shape[0] * s)))), interpolation=cv2.INTER_AREA)
    return t.astype(np.float32), s


def estimate_shift_px(ref_gray: np.ndarray, cur_gray: np.ndarray, width: int = 320) -> Tuple[float, float, float]:
    """
    Translation of `cur` relative to `ref` in FULL-RES pixels via phase correlation on thumbnails
    (nadir survey motion is near-pure translation).  Returns (dx, dy, response); response < ~0.05
    means the match is unreliable (low texture, too little overlap).
    """
    import cv2
    a, s = _thumb(ref_gray, width)
    b, _ = _thumb(cur_gray, width)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return dx / s, dy / s, float(resp)


# ───────────────────────────── frame scheduler ───────────────────────────────
class CoverageScheduler:
    """
    Decide which video frames to run the (expensive) tiled detector on.

    A frame is processed when the ground moved >= `spacing_px` since the last processed frame, so
    every object up to the planned size is fully inside `views` processed frames.  Motion comes
    from (priority) telemetry speed x dt, GPS displacement, or image phase correlation against the
    last PROCESSED frame (no drift accumulation).  Safety nets:
      * max_gap_s     always process at least this often (hover, unreliable motion estimate)
      * min_gap_s     never faster than this (protects the compute budget)
      * unreliable correlation -> process (fail towards recall, never towards skipping ground)
    Every decision is returned with a reason so it can go into the forensic log.
    """

    def __init__(self, frame_h_px: int, spacing_frac: float, max_gap_s: float = 2.0,
                 min_gap_s: float = 0.0, min_response: float = 0.05):
        self.H = frame_h_px
        self.spacing_px = max(1.0, spacing_frac * frame_h_px)
        self.max_gap_s, self.min_gap_s, self.min_resp = max_gap_s, min_gap_s, min_response
        self.last_t: Optional[float] = None
        self.last_gray: Optional[np.ndarray] = None
        self.last_gps: Optional[Tuple[float, float]] = None
        self.moved_px = 0.0
        self._step_px = 0.0          # motion between the last two frames (to avoid overshooting)
        self.n_seen = self.n_processed = 0

    @classmethod
    def from_plan(cls, plan: MissionPlan, frame_h_px: int, **kw) -> "CoverageScheduler":
        frac = plan.spacing_m / plan.footprint_m[1] if plan.spacing_m > 0 else 0.0
        return cls(frame_h_px, max(frac, 1e-3), **kw)

    def _accept(self, t, gray, gps, reason):
        self.last_t, self.last_gray, self.last_gps = t, gray, gps
        self.moved_px = 0.0
        self._step_px = 0.0
        self.n_processed += 1
        return True, reason

    def update(self, t_s: float, gray: Optional[np.ndarray] = None, gsd: Optional[float] = None,
               speed_mps: Optional[float] = None, gps: Optional[Tuple[float, float]] = None,
               dt_s: Optional[float] = None) -> Tuple[bool, str]:
        self.n_seen += 1
        if self.last_t is None:
            return self._accept(t_s, gray, gps, "first frame")
        gap = t_s - self.last_t
        if gap < self.min_gap_s:
            return False, f"min_gap ({gap:.2f}s)"
        if gap >= self.max_gap_s:
            return self._accept(t_s, gray, gps, f"max_gap {gap:.2f}s")
        # --- motion since last processed frame, in px of this frame
        before = self.moved_px
        if speed_mps is not None and gsd and dt_s is not None:
            self.moved_px += speed_mps * dt_s / gsd
            src = "telemetry speed"
        elif gps is not None and self.last_gps is not None and gsd:
            self.moved_px = haversine_m(self.last_gps[0], self.last_gps[1], gps[0], gps[1]) / gsd
            src = "gps"
        elif gray is not None and self.last_gray is not None:
            dx, dy, resp = estimate_shift_px(self.last_gray, gray)
            if resp < self.min_resp:
                return self._accept(t_s, gray, gps, f"motion unreliable (resp {resp:.3f})")
            self.moved_px = math.hypot(dx, dy)
            src = "image correlation"
        else:
            return self._accept(t_s, gray, gps, "no motion source")
        self._step_px = max(0.0, self.moved_px - before)
        # process THIS frame if the next one would overshoot the spacing (frames are discrete;
        # overshooting by one step would leave a strip seen fewer than `views` times)
        if self.moved_px + self._step_px > self.spacing_px:
            return self._accept(t_s, gray, gps, f"moved {self.moved_px:.0f}px (+{self._step_px:.0f}/frame) "
                                                f"~ spacing {self.spacing_px:.0f}px ({src})")
        return False, f"moved {self.moved_px:.0f}px < {self.spacing_px:.0f}px ({src})"
