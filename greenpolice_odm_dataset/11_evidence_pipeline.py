"""
EWADS -- Step 11: OFFLINE EVIDENCE PIPELINE  (RTX 5090 workstation, after the flight)
=====================================================================================
The Jetson's job is triage: find the dump site in real time and tell the pilot to fly the low
evidence pass.  The evidence that goes into an enforcement file is produced HERE, overnight, with
everything the Jetson cannot afford:

    every evidence frame
      -> for each model in the ENSEMBLE (2-3 independently trained checkpoints)
         -> for each TEST-TIME VIEW (D4 flips/rotations, optional up-scaling)
            -> native-resolution pyramid tiling (ewads_tiling, same as 07/08)
      -> WEIGHTED BOX FUSION of all runs           (boxes averaged, "found in k of N runs")
    all frames of the flight
      -> MULTI-VIEW FUSION across overlapping frames (registration / GPS)
         -> one candidate OBJECT per real ground object, scored by how consistently it was seen
      -> review crops + review.csv for the human analyst   (machine output is a CANDIDATE, a person
                                                          confirms -- that is what makes it evidence)
      -> hash-chained audit log: model hashes, code hashes, config, every frame, every object

Also an ABLATION mode: give it a labelled set (--data_dir with images/ + labels/) and it scores each
stage (single model, +TTA, +ensemble, +multi-view, +fill) with the same COCO metrics as 08, so you
see which lever actually pays on YOUR data instead of trusting anyone's claims (including mine).

Usage
  # evidence run on a flight's stills (evidence pass, 20 MP, XMP altitude + EXIF GPS read automatically)
  python 11_evidence_pipeline.py --weights runs/a/best.pt runs/b/best.pt --images D:/flights/2026-10-03/low \\
         --tta flips --out_dir evidence/2026-10-03

  # same on a video + DJI subtitle file (frames chosen so every object is in >= 3 processed frames)
  python 11_evidence_pipeline.py --weights a.pt b.pt --video DJI_0042.MP4 --srt DJI_0042.SRT --out_dir evidence/v42

  # ABLATION on labelled VAL: pick per-stage thresholds there ...
  python 11_evidence_pipeline.py --weights a.pt b.pt --data_dir data/val_seq --tta d4 --out_dir abl/val
  # ... then report on TEST with those thresholds frozen (never calibrate on test)
  python 11_evidence_pipeline.py --weights a.pt b.pt --data_dir data/test_seq --tta d4 \\
         --stage_thresholds abl/val/<run>/stage_thresholds.json --out_dir abl/test

  # re-fuse with other settings without re-running the models
  python 11_evidence_pipeline.py --data_dir data/val_seq --out_dir abl/val --reuse_cache --wbf_iou 0.6

Outputs: every invocation writes to its OWN folder  <out_dir>/run_<UTC>_<manifest sha>/  (manifest,
audit log, review crops, review.csv, report, 08-compatible detections_<stage>.npz,
stage_thresholds.json); <out_dir>/LATEST_RUN.txt names the newest.  Only <out_dir>/cache is shared.

Notes
  * Multi-view fusion only helps when frames OVERLAP (a flight sequence).  On a set of unrelated
    frames it does nothing (every object is a one-view track) -- the ablation will show ~0 gain.
  * Thresholds are per stage: fused scores live on a different scale (a box found by 1 of 8 runs is
    scored ~1/8), so a threshold tuned for the single model is wrong for the fused output.
  * Cost: (#models x #TTA views x #scales) x the tiles of one frame.  2 models x 4 flips on a 20 MP
    still = 8 x 74 tiles.  --dry_run prints the estimate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ewads_fusion as F      # noqa: E402
import ewads_metrics as M     # noqa: E402
import ewads_survey as es     # noqa: E402
import ewads_taxonomy as tx   # noqa: E402
import ewads_tiling as et     # noqa: E402

STAGES = ["single", "tta", "ensemble", "tta+ensemble", "multiview", "multiview+fill"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha256_file(p, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


# ───────────────────────── inputs ─────────────────────────
def capture_time(p: Path) -> str:
    """EXIF DateTimeOriginal + SubSecTimeOriginal (sortable), '' if absent."""
    try:
        from PIL import Image
        ex = Image.open(p).getexif().get_ifd(0x8769)
        dt, ss = ex.get(0x9003), ex.get(0x9291)
        return f"{dt or ''}.{ss or ''}"
    except Exception:
        return ""


def list_stills(d: Path) -> List[Path]:
    files = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXT]
    return sorted(files, key=lambda p: (capture_time(p), p.name))


def still_meta(p: Path, w: int, cam, alt_override: Optional[float], b07) -> dict:
    x = es.read_dji_xmp(p)
    gps = b07._extract_gps_from_exif(str(p)) or {}
    alt = alt_override or x.get("rel_alt_m")
    yaw = x.get("gimbal_yaw_deg", x.get("flight_yaw_deg"))
    return {"lat": gps.get("latitude"), "lon": gps.get("longitude"), "yaw_deg": yaw,
            "rel_alt_m": alt, "gsd_m": es.gsd_m(alt, w, cam) if alt else None,
            "gimbal_pitch_deg": x.get("gimbal_pitch_deg"),
            "altitude_source": "operator" if alt_override else ("dji_xmp RelativeAltitude" if alt else None)}


def iter_stills(paths: List[Path], cam, alt_override, b07, capture_utc_offset: Optional[str] = None):
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] unreadable {p.name}")
            continue
        meta = still_meta(p, img.shape[1], cam, alt_override, b07)
        yield p.stem, img, meta, {"image_path": str(p), "image_sha256": sha256_file(p),
                                  "capture_time": es.capture_time_from_image(p, capture_utc_offset)}


def iter_video(video: Path, srt: Optional[str], cam, alt_override, views: int, max_object_m: float,
               max_gap_s: float, capture_utc_offset: Optional[str] = None):
    """Frames chosen by the CoverageScheduler so every object <= max_object_m is fully inside >= views
    processed frames (3 by default here: multi-view fusion needs views to agree)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"[ERROR] cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vsha = sha256_file(video)
    tele = es.parse_dji_srt(srt) if srt else []
    t0s = np.array([r.get("t0_s", i / fps) for i, r in enumerate(tele)]) if tele else None
    mode = next((m for m, wh in cam.modes.items() if wh == (W, H) and m.startswith("video")), None)

    def tel(t):
        return tele[max(0, int(np.searchsorted(t0s, t, side="right")) - 1)] if tele else {}

    alt0 = alt_override or tel(0.0).get("rel_alt_m")
    if alt0:
        fh = H * es.gsd_m(alt0, W, cam, mode)
        frac = max(0.05, es.capture_spacing_m(fh, max_object_m, views) / fh)
    else:
        frac = 0.4 / views
    sched = es.CoverageScheduler(H, frac, max_gap_s=max_gap_s)
    idx, prev_t = -1, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            t = idx / fps
            tm = tel(t)
            alt = alt_override or tm.get("rel_alt_m")
            gsd = es.gsd_m(alt, W, cam, mode) if alt else None
            gps = (tm["lat"], tm["lon"]) if "lat" in tm else None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            go, why = sched.update(t, gray=gray if (gps is None and "speed_mps" not in tm) else None, gsd=gsd,
                                   speed_mps=tm.get("speed_mps"), gps=gps, dt_s=None if prev_t is None else t - prev_t)
            prev_t = t
            if not go:
                continue
            if sched.last_gray is None:
                sched.last_gray = gray
            meta = {"lat": tm.get("lat"), "lon": tm.get("lon"), "yaw_deg": None,   # P4P SRT has no heading
                    "rel_alt_m": alt, "gsd_m": gsd, "altitude_source": "dji_srt H" if alt else None}
            yield f"f{idx:06d}", frame, meta, {
                "image_path": f"{video}#frame={idx}", "video_sha256": vsha, "frame_index": idx,
                "frame_time_s": round(t, 3), "schedule_reason": why,
                "image_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                "capture_time": es.capture_time_from_srt(tm, capture_utc_offset)}
    finally:
        cap.release()    # generator may stop early; Windows keeps the file locked until released


# ───────────────────────── models ─────────────────────────
class Ensemble:
    """Each model wrapped in its own TiledDetector (models may use different tile sizes)."""

    def __init__(self, paths: List[str], args, b08):
        self.paths, self.dets, self.names = paths, [], []
        for p in paths:
            ns = argparse.Namespace(weights=p if p.endswith(".pt") else None,
                                    onnx=p if p.endswith(".onnx") else None,
                                    engine=p if p.endswith(".engine") else None,
                                    tile=args.tile, min_conf=args.min_conf, device=args.device, batch=args.batch)
            fn, tile, batch = b08.build_predict_fn(ns, args.tile)
            self.dets.append(et.TiledDetector(fn, tile=tile, overlap=args.overlap, batch=batch,
                                              conf=args.min_conf, max_levels=args.max_levels))
            self.names.append(Path(p).name)

    def run(self, frame: np.ndarray, plan: List[F.TTARun]) -> List[np.ndarray]:
        out = []
        for d in self.dets:
            out += F.run_tta(frame, lambda f: d(f), plan)
        return out


def fuse_stages(runs: List[np.ndarray], plan: List[F.TTARun], n_models: int, args,
                model_w: np.ndarray) -> Dict[str, np.ndarray]:
    """runs are ordered model-major: runs[m * len(plan) + j]."""
    P = len(plan)
    id_j = next(j for j, r in enumerate(plan) if r.k == 0 and r.scale == 1.0)
    wbf = lambda rs, ws: F.weighted_boxes_fusion(rs, ws, args.wbf_iou, args.min_conf, args.wbf_conf,
                                                 one_per_run=not args.wbf_reference,
                                                 pad_px=0.0 if args.wbf_reference else args.wbf_pad_px)
    single = runs[id_j]
    st = {"single": np.column_stack([single, np.ones(len(single))]) if len(single) else np.zeros((0, 7))}
    st["tta"] = wbf(runs[:P], None) if P > 1 else st["single"]
    if n_models > 1:
        st["ensemble"] = wbf([runs[m * P + id_j] for m in range(n_models)], model_w)
        st["tta+ensemble"] = wbf(runs, np.repeat(model_w, P)) if P > 1 else st["ensemble"]
    else:
        st["ensemble"], st["tta+ensemble"] = st["single"], st["tta"]
    return st


# ───────────────────────── scoring (ablation) ─────────────────────────
def score_stage(gts, dets, conf: Optional[float], target_recall: float, assess_px: float) -> dict:
    nc = len(tx.CLASSES)
    d6 = {k: v[:, :6] for k, v in dets.items()}
    r = M.evaluate(gts, d6, nc)
    re_ = M.evaluate(gts, d6, nc, size_range=(assess_px, 1e9))
    cal = None
    if conf is None:
        cal = M.calibrate_threshold(gts, d6, nc, target_recall,
                                    thresholds=tuple(np.round(np.arange(0.005, 0.96, 0.005), 3)))
        conf = cal["threshold"]
    op = M.operating_point(gts, d6, nc, conf)
    ope = M.operating_point(gts, d6, nc, conf, min_px=assess_px)
    return {"map5095": r["map"], "map50": r["map50"], "map75": r["map75"], "env_map5095": re_["map"],
            "conf": conf, "calibrated_here": cal is not None,
            "reached_target": cal["reached_target"] if cal else None,
            "recall": op["recall"], "fp_share": op["fp_share"], "fp_per_image": op["fp_per_image"],
            "env_recall": ope["recall"], "env_fp_share": ope["fp_share"]}


# ───────────────────────── evidence objects ─────────────────────────
def save_crop(frame, box, out_dir: Path, name: str) -> dict:
    x1, y1, x2, y2 = box
    s = max(x2 - x1, y2 - y1)
    pad = max(64.0, s)
    h, w = frame.shape[:2]
    a, b = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    c, d = int(min(w, x2 + pad)), int(min(h, y2 + pad))
    raw = np.ascontiguousarray(frame[b:d, a:c])
    ann = raw.copy()
    cv2.rectangle(ann, (int(x1 - a), int(y1 - b)), (int(x2 - a), int(y2 - b)), (0, 0, 255), 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    pr, pa = out_dir / f"{name}_raw.png", out_dir / f"{name}_boxed.png"
    cv2.imwrite(str(pr), raw); cv2.imwrite(str(pa), ann)
    return {"crop_raw": pr.name, "crop_raw_sha256": sha256_file(pr),
            "crop_pixels_sha256": hashlib.sha256(raw.tobytes()).hexdigest(), "crop_shape": list(raw.shape),
            "crop_boxed": pa.name,
            "crop_offset_xy": [a, b],
            "crop_note": "raw = unmodified pixels of the analysed frame (source re-verified by SHA-256 before "
                         "cropping); crop_pixels_sha256 hashes the pixel array (PNG bytes vary by encoder)"}


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", default=[], help="1-3 models (.pt / .onnx / .engine): the ensemble")
    ap.add_argument("--model_weights", nargs="+", type=float, default=None, help="WBF weight per model")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="folder of evidence stills (one flight / pass, any order: sorted by EXIF time)")
    src.add_argument("--video", help="DJI video (use with --srt)")
    src.add_argument("--data_dir", help="LABELLED set (images/ + labels/) -> ablation table")
    ap.add_argument("--srt", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--camera", default="phantom4pro", choices=list(es.CAMERAS))
    ap.add_argument("--altitude_agl", type=float, default=None, help="override altitude above take-off (m)")
    # tiling
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--overlap", type=int, default=et.DEFAULT_OVERLAP)
    ap.add_argument("--max_levels", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--min_conf", type=float, default=0.001)
    # TTA + WBF
    ap.add_argument("--tta", default="flips", choices=list(F.TTA_SETS),
                    help="none | flips (4 views) | d4 (8 views: flips + 90 deg rotations; both are in 03's training augs)")
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0],
                    help="extra frame scales, e.g. 1.0 1.5 (up-scaling helps 8-20 px objects; bicubic, no AI super-resolution)")
    ap.add_argument("--wbf_iou", type=float, default=0.55)
    ap.add_argument("--wbf_conf", default="avg", choices=["avg", "max"])
    ap.add_argument("--wbf_pad_px", type=float, default=1.5,
                    help="compare boxes grown by this many px (absorbs pixel jitter on small objects; see ewads_fusion)")
    ap.add_argument("--wbf_reference", action="store_true",
                    help="reference WBF (a run may add several boxes to one cluster); default: one box per run")
    # multi-view
    ap.add_argument("--no_multiview", action="store_true")
    ap.add_argument("--min_views", type=int, default=2, help="detections needed to CONFIRM an object")
    ap.add_argument("--window", type=int, default=12, help="frames a track may go unseen before it closes")
    ap.add_argument("--gps_err_m", type=float, default=2.0)
    ap.add_argument("--fill_factor", type=float, default=0.9)
    ap.add_argument("--mv_score", default="geo", choices=["geo", "track"],
                    help="per-box score after multi-view: sqrt(own x track) or the track score")
    ap.add_argument("--views", type=int, default=3, help="video: each object fully inside >= this many processed frames")
    ap.add_argument("--max_object_m", type=float, default=1.0)
    ap.add_argument("--max_gap_s", type=float, default=2.0)
    # reporting
    ap.add_argument("--report_conf", type=float, default=None,
                    help="score for an object to be listed for review (default: the final stage's threshold "
                         "from --stage_thresholds, else 0.25)")
    ap.add_argument("--stage_thresholds", default=None, help="JSON from a VAL ablation run (frozen thresholds)")
    ap.add_argument("--allow_selection_on_heldout", action="store_true",
                    help="override the guard that refuses deriving stage thresholds on a held-out/test folder")
    ap.add_argument("--capture_utc_offset", default=None,
                    help="camera clock UTC offset (e.g. +04:00), used only when a file has none; recorded as asserted")
    ap.add_argument("--signing_pubkey", default=None,
                    help="PEM public key that will sign this run's audit head (tools/sign_chain_head.py); "
                         "its fingerprint goes into the run manifest")
    ap.add_argument("--target_recall", type=float, default=0.87)
    ap.add_argument("--assess_px", type=float, default=16)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--reuse_cache", action="store_true", help="reuse raw per-run detections from out_dir/cache")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.data_dir and not args.stage_thresholds:      # this run would DERIVE thresholds -> selection data
        import ewads_hier_eval as HE                     # checked BEFORE any inference
        why = HE.selection_data_problem(args.data_dir)
        if why and not args.allow_selection_on_heldout:
            sys.exit(f"[ERROR] deriving stage thresholds refused: {why}. Pass --stage_thresholds from the "
                     f"synthetic validation-sequence run (main/val_seq) to score this set.")
        if why:
            print(f"  ⚠  OVERRIDE (--allow_selection_on_heldout): {why}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "cache"; cache_dir.mkdir(exist_ok=True)
    b07, b08 = _load("bench07", "07_edge_benchmark.py"), _load("eval08", "08_tiled_eval.py")
    cam = es.CAMERAS[args.camera]
    plan = F.tta_plan(args.tta, args.scales)
    if not any(r.k == 0 and r.scale == 1.0 for r in plan):
        plan = [F.TTARun(0, 1.0)] + plan
    n_models = len(args.weights)
    missing = [p for p in args.weights if not Path(p).is_file()]
    if missing:
        sys.exit(f"[ERROR] model file(s) not found: {missing}")
    models = [{"file": p, "sha256": sha256_file(p)} for p in args.weights]
    models_given = bool(n_models)
    if not models_given:
        if not args.reuse_cache:
            sys.exit("[ERROR] --weights required (or --reuse_cache with an existing out_dir/cache)")
        anyc = next(iter(sorted(cache_dir.glob("*.npz"))), None)
        if anyc is None:
            sys.exit(f"[ERROR] --reuse_cache but {cache_dir} is empty")
        z = np.load(anyc, allow_pickle=False)
        if "settings" not in z.files:
            sys.exit(f"[ERROR] {anyc.name} predates cache settings; re-run with --weights")
        cs = json.loads(str(z["settings"]))
        n_models, models = len(cs["models"]), cs["models"]
        if [r.name for r in plan] != cs["plan"]:
            sys.exit(f"[ERROR] cache was made with TTA runs {cs['plan']}; pass the same --tta/--scales")
        for k in ("tile", "overlap", "max_levels", "min_conf"):       # the cached runs were made with these
            if getattr(args, k) != cs[k]:
                print(f"  (--{k} {getattr(args, k)} ignored: cached runs used {cs[k]})")
                setattr(args, k, cs[k])
        print(f"  (re-fusing cached runs of {n_models} model(s): {[m['file'] for m in models]})")
    model_w = np.asarray(args.model_weights if args.model_weights else [1.0] * max(n_models, 1), float)

    # ── provenance ──
    code = {f: sha256_file(HERE / f) for f in ("11_evidence_pipeline.py", "ewads_fusion.py", "ewads_tiling.py",
                                                "ewads_taxonomy.py", "ewads_metrics.py", "ewads_survey.py")}
    cfg = {k: v for k, v in vars(args).items()}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "models": models, "code_sha256": code,
                "config": cfg, "tta_runs": [r.name for r in plan], "taxonomy_sha256": tx.taxonomy_sha256(),
                "python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__}
    if args.signing_pubkey:
        sys.path.insert(0, str(HERE / "tools"))
        import sign_chain_head as SCH
        manifest["signing_public_key_fingerprint"] = SCH.public_key_fingerprint(args.signing_pubkey)
    else:
        manifest["signing_public_key_fingerprint"] = None
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True, default=_json).encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    # every run gets its OWN folder: re-fusing with other settings must never overwrite the crops,
    # review sheet or manifest that an earlier audit log refers to (the cache is shared)
    run_dir = out / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{manifest_sha[:10]}"
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "run_manifest.json").write_bytes(manifest_bytes)
        (out / "LATEST_RUN.txt").write_text(run_dir.name)
    cache_settings = {"models": models, "plan": [r.name for r in plan], "tile": args.tile, "overlap": args.overlap,
                      "max_levels": args.max_levels, "min_conf": args.min_conf}
    cache_key = hashlib.sha256(json.dumps(cache_settings, sort_keys=True).encode()).hexdigest()[:16]

    # ── frames ──
    gts = None
    if args.images or args.data_dir:
        d = Path(args.images or Path(args.data_dir) / "images")
        paths = list_stills(d)[:args.max_frames] if args.max_frames else list_stills(d)
        frames_iter = iter_stills(paths, cam, args.altitude_agl, b07, args.capture_utc_offset)
        n_expected = len(paths)
        if args.data_dir:
            from PIL import Image
            gts = {}
            for p in paths:
                with Image.open(p) as im:
                    w, h = im.size
                gts[p.stem] = et.yolo_to_xyxy(et.read_yolo(Path(args.data_dir) / "labels" / f"{p.stem}.txt"), w, h)
    else:
        frames_iter = iter_video(Path(args.video), args.srt, cam, args.altitude_agl, args.views,
                                 args.max_object_m, args.max_gap_s, args.capture_utc_offset)
        n_expected = None

    print(f"\n  EWADS evidence pipeline")
    print(f"  models   : {n_models}  {[Path(m['file']).name for m in models]}")
    print(f"  views    : {len(plan)} per model ({', '.join(r.name for r in plan)})  -> "
          f"{max(n_models, 1) * len(plan)} runs per frame, fused by WBF (iou {args.wbf_iou})")
    print(f"  multiview: {'off' if args.no_multiview else f'on (confirm at >= {args.min_views} views)'}")
    print(f"  run      : {run_dir}  (manifest sha256 {manifest_sha[:16]}...)")
    if args.dry_run:
        tile = args.tile or et.DEFAULT_TILE
        for wh in [(5472, 3648), (3840, 2160)]:
            n = sum(len(et.plan_tiles(int(wh[0] * s), int(wh[1] * s), tile, args.overlap)) for s in args.scales) \
                * len(F.TTA_SETS[args.tta]) * max(n_models, 1)
            print(f"  dry run  : {wh[0]}x{wh[1]} -> {n} tile inferences per frame "
                  f"(~{n * 0.01:.1f} s at 10 ms/tile; measure on your 5090)")
        if n_expected:
            print(f"  frames   : {n_expected}")
        return

    ensemble = Ensemble(args.weights, args, b08) if models_given else None
    fuser = None if args.no_multiview else F.MultiViewFuser(window=args.window, gps_err_m=args.gps_err_m,
                                                            min_views=args.min_views)
    stage_dets: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in STAGES[:4]}
    frame_info: Dict[str, dict] = {}
    t_all, n = time.time(), 0
    for fid, frame, meta, info in frames_iter:
        t0 = time.time()
        cf = cache_dir / f"{fid}.npz"
        runs = None
        if args.reuse_cache and cf.exists():
            z = np.load(cf, allow_pickle=False)
            if str(z["key"]) == cache_key:
                runs = [z[f"r{i}"] for i in range(int(z["n"]))]
            elif not models_given:
                sys.exit(f"[ERROR] {cf.name} was made with different models/settings than the rest of the cache; "
                         f"re-run with --weights")
        if runs is None:
            if ensemble is None:
                sys.exit(f"[ERROR] no cache for {fid} and no --weights")
            runs = ensemble.run(frame, plan)
            np.savez_compressed(cf, key=cache_key, n=len(runs), n_models=n_models,
                                settings=json.dumps(cache_settings),
                                **{f"r{i}": r.astype(np.float32) for i, r in enumerate(runs)})
        st = fuse_stages(runs, plan, n_models, args, model_w)
        for s, dd in st.items():
            stage_dets[s][fid] = dd
        if fuser is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            link = fuser.add_frame(fid, st["tta+ensemble"], (frame.shape[1], frame.shape[0]), gray, meta)
            info["registration"] = link["link"]
            info["segment"] = link["segment"]
        info.update(meta=meta, wh=[frame.shape[1], frame.shape[0]])
        frame_info[fid] = info
        n += 1
        print(f"    {n}/{n_expected or '?'}  {fid}  runs {len(runs)}  fused {len(st['tta+ensemble'])}"
              f"  {time.time() - t0:.1f}s" + (f"  link {info.get('registration', {}).get('method')}" if fuser else ""),
              end="\r")
    print(f"\n  processed {n} frames in {time.time() - t_all:.0f}s")
    if n == 0:
        sys.exit("[ERROR] no frames")

    final = "tta+ensemble"
    tracks = []
    if fuser is not None:
        tracks = fuser.finalize()
        stage_dets["multiview"] = fuser.rescored(fill_missed=False, mode=args.mv_score)
        stage_dets["multiview+fill"] = fuser.rescored(fill_missed=True, fill_factor=args.fill_factor,
                                                      mode=args.mv_score)
        final = "multiview"
        segs = len({fr.seg for fr in fuser.frames})
        methods = {}
        for fr in fuser.frames[1:]:
            mth = fr.link.get("method") or "none"
            methods[mth] = methods.get(mth, 0) + 1
        print(f"  multi-view: {segs} segment(s); frame links {methods}; "
              f"{sum(t.n_obs >= args.min_views for t in tracks)} confirmed tracks of {len(tracks)}")

    for s, dd in stage_dets.items():                       # 08-compatible caches
        if dd:
            np.savez_compressed(run_dir / f"detections_{s.replace('+', '_')}.npz",
                                **{k: v[:, :6].astype(np.float32) for k, v in dd.items()})

    # thresholds are only valid for the fusion settings (and models) they were calibrated with
    fusion_cfg = {"models": [m["sha256"] for m in models], "tta_runs": [r.name for r in plan],
                  "wbf_iou": args.wbf_iou, "wbf_pad_px": args.wbf_pad_px, "wbf_conf": args.wbf_conf,
                  "wbf_reference": args.wbf_reference, "mv_score": args.mv_score, "fill_factor": args.fill_factor,
                  "min_views": args.min_views, "window": args.window, "tile": args.tile, "overlap": args.overlap,
                  "max_levels": args.max_levels, "min_conf": args.min_conf}
    thr_in =json.loads(Path(args.stage_thresholds).read_text()) if args.stage_thresholds else {}
    if thr_in:
        diff = {k: (thr_in.get("_fusion_config", {}).get(k), v) for k, v in fusion_cfg.items()
                if thr_in.get("_fusion_config", {}).get(k) != v}
        if diff:
            print(f"\n  ⚠  --stage_thresholds were calibrated with DIFFERENT settings: {diff}\n"
                  f"     the operating points below are not valid; re-run the VAL ablation with these settings.")
        thr_in = {k: v for k, v in thr_in.items() if not k.startswith("_")}

    # ── ablation ──
    report = {"manifest_sha256": manifest_sha, "n_frames": n, "stages": {}}
    if gts is not None:
        n_gt = sum(len(c) for c, _ in gts.values())
        print(f"\n  ABLATION on {args.data_dir}: {n} frames, {n_gt} objects"
              f"  ({'thresholds frozen from ' + args.stage_thresholds if thr_in else 'thresholds calibrated HERE for recall >= %.2f -- use VAL' % args.target_recall})")
        hdr = f"  {'stage':<16}{'mAP':>7}{'d':>7}{'AP50':>7}{'AP75':>7}{'env mAP':>9}{'conf':>7}{'recall':>8}{'FP share':>9}{'env rec':>9}{'env FP':>8}"
        print(hdr)
        base = None
        thr_out = {}
        for s in STAGES:
            if not stage_dets.get(s):
                continue
            if s in ("ensemble", "tta+ensemble") and n_models < 2:
                continue
            if s == "tta" and len(plan) < 2:
                continue
            r = score_stage(gts, stage_dets[s], thr_in.get(s), args.target_recall, args.assess_px)
            base = r["map5095"] if base is None else base
            report["stages"][s] = r
            thr_out[s] = r["conf"]
            print(f"  {s:<16}{r['map5095']:>7.4f}{r['map5095'] - base:>+7.4f}{r['map50']:>7.3f}{r['map75']:>7.3f}"
                  f"{r['env_map5095']:>9.4f}{r['conf']:>7.3f}{r['recall']:>8.3f}{r['fp_share']:>9.3f}"
                  f"{r['env_recall']:>9.3f}{r['env_fp_share']:>8.3f}")
        if not thr_in:
            (run_dir / "stage_thresholds.json").write_text(json.dumps({**thr_out, "_fusion_config": fusion_cfg,
                                                                   "_target_recall": args.target_recall}, indent=2))
            print(f"  thresholds -> {run_dir / 'stage_thresholds.json'}  (pass to the TEST run with --stage_thresholds)")
        print("  d = change in mAP@0.5:0.95 vs the single model.  A gain is only real if it holds on TEST "
              "with the VAL thresholds.")

    # ── evidence objects + audit log ──
    rep_conf = args.report_conf if args.report_conf is not None else thr_in.get(final, 0.25)
    logger = b07.ForensicAuditLogger(run_dir / "evidence_audit.jsonl",
                                     model_version="ensemble:" + ",".join(m["sha256"][:12] for m in models),
                                     model_sha256=[m["sha256"] for m in models])
    final_dets = stage_dets[final]
    sup = {(o["f"], t.tid): o["support"] for t in tracks for o in t.obs}
    fidx = {fr.fid: i for i, fr in enumerate(fuser.frames)} if fuser else {}
    tmap = {t.tid: t for t in tracks}
    for fid, info in frame_info.items():
        d = final_dets.get(fid, np.zeros((0, 7)))
        dl = []
        for row in d[d[:, 4] >= rep_conf] if len(d) else []:
            if fuser is not None:
                # what THIS frame's detector output said (observed); the track's vote is logged beside it
                propagated = bool(row[7])
                c = int(row[5]) if propagated else int(row[8])
                name = tx.CLASSES[c]
                rec = {"class_id": c, "class_name": name, "group": tx.FINE_TO_GROUP[name],
                       "confidence": round(float(row[4]), 4),
                       "observed_conf": None if propagated else round(float(row[9]), 4),
                       "bbox_xyxy": [round(float(v), 1) for v in row[:4]], "tier": 1 if name in tx.TIER_1 else 2,
                       "propagated": propagated}
                if propagated:
                    rec["note"] = "NOT detected in this frame: projected from the other views of this object"
                if row[6] >= 0:
                    t = tmap[int(row[6])]
                    rec.update(track_id=int(row[6]), track_class=tx.CLASSES[t.cls], n_views=t.n_obs,
                               n_covering=len(t.covering), n_runs=int(sup.get((fidx[fid], t.tid), 0)))
            else:
                c = int(row[5]); name = tx.CLASSES[c]
                rec = {"class_id": c, "class_name": name, "group": tx.FINE_TO_GROUP[name],
                       "confidence": round(float(row[4]), 4), "bbox_xyxy": [round(float(v), 1) for v in row[:4]],
                       "tier": 1 if name in tx.TIER_1 else 2, "n_runs": int(row[6]) if len(row) > 6 else 1}
            dl.append(rec)
        meta = info.get("meta", {})
        ctx = {"record_type": "frame", "run_manifest_sha256": manifest_sha, "stage": final,
               "score_rule": ({"geo": "sqrt(WBF conf in this frame x track consistency), consistency = "
                                      "sum(conf over views) / frames whose footprint contained the object",
                               "track": "track consistency = sum(conf over views) / frames whose footprint "
                                        "contained the object"}[args.mv_score] if fuser else
                              f"WBF ({args.wbf_conf}) over {max(n_models, 1) * len(plan)} runs; absent runs count as 0"),
               **{k: v for k, v in info.items() if k not in ("image_path", "image_sha256", "meta")},
               "capture": meta}
        gps = {"latitude": meta["lat"], "longitude": meta["lon"]} if meta.get("lat") is not None else None
        logger.log_detection(info["image_path"], info["image_sha256"], dl, gps, meta.get("rel_alt_m"), rep_conf, ctx,
                             capture_time=info.get("capture_time"))

    objects = []
    if fuser is not None:
        crops = run_dir / "review_crops"
        cands = list(fuser.candidate_objects(rep_conf, mode=args.mv_score))
        # Pass 1: cut every crop, visiting source frames in file/frame order so a video is decoded
        # forward once (one reader per video, closed afterwards) instead of re-opened per candidate.
        best_of = {t.tid: max(t.obs, key=lambda o: o["conf"]) for t, _ in cands}
        by_frame = {}
        for t, _ in cands:
            by_frame.setdefault(fuser.frames[best_of[t.tid]["f"]].fid, []).append(t)
        crop_of = {}
        readers = {}
        try:
            for fid in sorted(by_frame, key=lambda f: _frame_sort_key(frame_info[f]["image_path"])):
                img, why = _reload(frame_info[fid], readers)   # re-read the source (frames are not kept in RAM)
                for t in by_frame[fid]:
                    name = tx.CLASSES[t.cls]
                    crop_of[t.tid] = save_crop(img, best_of[t.tid]["box"], crops, f"obj{t.tid:05d}_{name}") \
                        if img is not None else {"crop_error": why}
        finally:
            for r in readers.values():
                r.close()
        # Pass 2: records in candidate order (unchanged output order)
        for t, absorbed in cands:
            best = best_of[t.tid]
            fr = fuser.frames[best["f"]]
            finfo = frame_info[fr.fid]
            name = tx.CLASSES[t.cls]
            crop = crop_of[t.tid]
            pos = fuser.ground_position(t)
            obj = {"track_id": t.tid, "class_name": name, "group": tx.FINE_TO_GROUP[name],
                   "tier": 1 if name in tx.TIER_1 else 2,
                   "score": round(t.box_score if args.mv_score == "geo" else t.score, 4),
                   "consistency": round(t.score, 4),
                   "confirmed": t.n_obs >= args.min_views, "n_views": t.n_obs, "n_covering": len(t.covering),
                   "class_votes": {tx.CLASSES[k]: round(v, 4) for k, v in t.class_votes.items()},
                   "best_frame": fr.fid, "best_box_xyxy": [round(float(v), 1) for v in best["box"]],
                   "best_conf": round(best["conf"], 4), "best_n_runs": best["support"],
                   "best_box_long_side_px": round(float(max(best["box"][2] - best["box"][0],
                                                            best["box"][3] - best["box"][1])), 1),
                   "ground_position": pos, **crop,
                   "observations": [{"frame": fuser.frames[o["f"]].fid, "box": [round(float(v), 1) for v in o["box"]],
                                     "conf": round(o["conf"], 4), "class": tx.CLASSES[o["cls"]], "n_runs": o["support"]}
                                    for o in t.obs],
                   "covering_frames": [fuser.frames[i].fid for i in t.covering],
                   "merged_duplicate_tracks": [{"track_id": a.tid, "consistency": round(a.score, 4),
                                                "class": tx.CLASSES[a.cls], "n_views": a.n_obs} for a in absorbed]}
            objects.append(obj)
            ctx = {"record_type": "evidence_object", "run_manifest_sha256": manifest_sha,
                   **{k: v for k, v in obj.items() if k not in ("class_name",)},
                   "status": "CANDIDATE - requires human review"}
            det = [{"class_id": t.cls, "class_name": name, "confidence": obj["score"],
                    "bbox_xyxy": obj["best_box_xyxy"], "tier": obj["tier"]}]
            logger.log_detection(finfo["image_path"], finfo["image_sha256"], det,
                                 {"latitude": pos["lat"], "longitude": pos["lon"]} if pos else None,
                                 finfo.get("meta", {}).get("rel_alt_m"), rep_conf, ctx,
                                 capture_time=finfo.get("capture_time"))
        with open(run_dir / "review.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "class", "group", "tier", "score", "consistency", "confirmed", "n_views", "n_covering",
                        "best_frame", "best_box_px", "lat", "lon", "crop", "reviewer_decision", "reviewer_class",
                        "reviewer_name", "reviewer_note"])
            for o in objects:
                p = o["ground_position"] or {}
                w.writerow([o["track_id"], o["class_name"], o["group"], o["tier"], o["score"], o["consistency"], o["confirmed"],
                            o["n_views"], o["n_covering"], o["best_frame"], o["best_box_long_side_px"],
                            p.get("lat", ""), p.get("lon", ""), o.get("crop_boxed", ""), "", "", "", ""])
        n_conf = sum(o["confirmed"] for o in objects)
        print(f"\n  {len(objects)} candidate objects with score >= {rep_conf:.3f} ({n_conf} confirmed in "
              f">= {args.min_views} views, {sum(o['tier'] == 1 for o in objects)} tier-1)")
        print(f"  review sheet -> {run_dir / 'review.csv'}   crops -> {run_dir / 'review_crops'}")

    chain = logger.verify_chain()
    report.update(objects=objects, report_conf=rep_conf, final_stage=final, chain=chain)
    (run_dir / "evidence_report.json").write_text(json.dumps(report, indent=2, default=_json))
    print(f"  audit chain: {'valid' if chain['valid'] else 'BROKEN'} ({chain.get('records')} records) "
          f"-> {run_dir / 'evidence_audit.jsonl'}")
    print(f"  report     -> {run_dir / 'evidence_report.json'}")
    print(f"  sign the chain head now (key kept OUTSIDE this repository):\n"
          f"    python tools/sign_chain_head.py sign --log {run_dir / 'evidence_audit.jsonl'} --private_key <key.pem>"
          + ("" if manifest.get("signing_public_key_fingerprint") else
             "\n    (no --signing_pubkey given: the run manifest records no signing-key fingerprint)") + "\n")


class _SequentialVideoReader:
    """Decodes a video strictly forward (seeking can land on a different frame on some codecs) and
    keeps the capture open between requests. Asking for an earlier frame re-opens from the start."""

    def __init__(self, path: str):
        self.path, self.cap, self.next_idx = path, None, 0

    def get(self, idx: int):
        if self.cap is None or idx < self.next_idx:
            self.close()
            self.cap, self.next_idx = cv2.VideoCapture(self.path), 0
        ok, fr = False, None
        while self.next_idx <= idx:
            ok, fr = self.cap.read()
            self.next_idx += 1
            if not ok:
                break
        return (fr if ok else None)

    def close(self):
        if self.cap is not None:
            self.cap.release()    # Windows keeps the file locked until released
            self.cap = None


def _frame_sort_key(image_path: str):
    """(source file, frame index) so crops visit each video's frames in decode order."""
    if "#frame=" in image_path:
        v, i = image_path.split("#frame=")
        return (v, int(i))
    return (image_path, 0)


def _reload(info: dict, readers: Optional[dict] = None):
    """Re-read the analysed frame for a review crop and PROVE it is the same pixels/bytes that were
    analysed (hash recorded at analysis time).  -> (image, None) or (None, reason).
    `readers` ({video path: _SequentialVideoReader}) is owned and closed by the caller; without it
    a one-off reader is opened and closed here."""
    p = info["image_path"]
    if "#frame=" in p:
        v, i = p.split("#frame=")
        own = readers is None
        reader = _SequentialVideoReader(v) if own else readers.setdefault(v, _SequentialVideoReader(v))
        try:
            fr = reader.get(int(i))
        finally:
            if own:
                reader.close()
        if fr is None:
            return None, "frame could not be decoded again"
        if hashlib.sha256(fr.tobytes()).hexdigest() != info["image_sha256"]:
            return None, "re-decoded frame differs from the analysed frame (decoder not deterministic?)"
        return fr, None
    try:
        raw = Path(p).read_bytes()
    except OSError as e:
        return None, f"source unreadable: {e}"
    if hashlib.sha256(raw).hexdigest() != info["image_sha256"]:
        return None, "source file changed since analysis (SHA-256 mismatch)"
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), None


if __name__ == "__main__":
    main()
