#!/usr/bin/env python3
"""
H2: ground-sampling-distance / assessability geometry validation (ATHAR, added 2026-10-08).

The assessability gate ("is this object >= 16 px at this altitude?") rests on a pinhole model:
predicted pixel length = L_target / GSD, GSD = height * sensor_width / (focal * image_width)
(ewads_survey.gsd_m -- the SAME function the renderer, the planner and 07 use).  This tool checks that
model against photographs of a reference target of known length (default 1 m) taken from measured heights.

Inputs
  --photos DIR            reference-target photos
  --heights CSV           columns: stem,height_m,camera
                          camera = a profile name in ewads_survey.CAMERAS (e.g. phantom4pro) -> profile
                          intrinsics; anything else (e.g. 'exif', 'pixel7') -> EXIF intrinsics
  one annotation source of the target's two ENDPOINTS (pixels):
    --endpoints CSV       columns: stem,x1,y1,x2,y2
    --cvat_xml FILE       CVAT for images 1.1: a <polyline> (first and last point) or <points> with
                          >= 2 points on each image (label filter: --cvat_label)
    --yolo_pose_dir DIR   <stem>.txt  'cls cx cy w h x1 y1 [v1] x2 y2 [v2]' (normalised keypoints)
  --target_m 1.0          true target length
  --sensor_width_mm W     EXIF fallback when only FocalLength is present (phones rarely need it)

Intrinsics per image, in order (the source used is recorded per image):
  profile            focal_mm / sensor_w_mm (x mode width_crop) from ewads_survey.CAMERAS; the mode is
                     inferred from the image size (video modes matter: see crop factor below)
  exif focal plane   FocalLength x FocalPlaneXResolution (px per mm on the sensor)
  exif 35mm-equiv    FocalLengthIn35mmFilm, diagonal convention: f_px = f35 * diag_px / 43.27 mm
  exif + sensor      FocalLength / --sensor_width_mm x image width

Assumptions (recorded): nadir camera, target lying on flat ground at the measured height below the
camera.  For a nadir camera over flat ground the scale is the same everywhere in the frame (lens
distortion aside).  Oblique shots violate the model: DJI XMP gimbal pitch is recorded and
|pitch + 90| > 5 deg is flagged; the target's offset from the image centre is recorded too.

Outputs (default results_bundle/): h2_gsd.json (per image + summary by camera / height / mode +
implied video crop + H2 target check with the targets' freeze status), h2_gsd.csv, h2_gsd_scatter.png.

Usage
  python tools/gsd_validation.py --photos data/h2/photos --heights data/h2/heights.csv --cvat_xml data/h2/annotations.xml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ewads_survey as es  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DIAG_35MM = math.hypot(36.0, 24.0)          # 43.27 mm
PITCH_TOL_DEG = 5.0


# ───────────────────────── annotations ─────────────────────────
def endpoints_from_csv(p: Path) -> dict:
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        out[r["stem"]] = ((float(r["x1"]), float(r["y1"])), (float(r["x2"]), float(r["y2"])))
    return out


def endpoints_from_cvat(p: Path, label: Optional[str]) -> dict:
    out = {}
    for img in ET.parse(p).getroot().iter("image"):
        stem = Path(img.get("name", "")).stem
        for el in list(img):
            if el.tag not in ("polyline", "points") or (label and el.get("label") != label):
                continue
            pts = [tuple(float(v) for v in xy.split(",")) for xy in el.get("points", "").split(";") if xy]
            if len(pts) >= 2:
                out[stem] = (pts[0], pts[-1])
                break
    return out


def endpoints_from_yolo_pose(d: Path, sizes: dict) -> dict:
    out = {}
    for f in d.glob("*.txt"):
        if f.stem not in sizes:
            continue
        W, H = sizes[f.stem]
        vals = [float(v) for v in f.read_text(encoding="utf-8").split()]
        kp = vals[5:]
        if len(kp) >= 6 and len(kp) % 3 == 0:          # with visibility flags
            pts = [(kp[i], kp[i + 1]) for i in range(0, len(kp), 3)]
        elif len(kp) >= 4:
            pts = [(kp[i], kp[i + 1]) for i in range(0, len(kp) - 1, 2)]
        else:
            continue
        out[f.stem] = ((pts[0][0] * W, pts[0][1] * H), (pts[-1][0] * W, pts[-1][1] * H))
    return out


# ───────────────────────── intrinsics ─────────────────────────
def _exif(path: Path) -> dict:
    from PIL import Image
    with Image.open(path) as im:
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
    def f(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    unit = sub.get(0xA210)                            # FocalPlaneResolutionUnit: 2 inch, 3 cm, 4 mm
    return {"focal_mm": f(sub.get(0x920A)), "f35": f(sub.get(0xA405)),
            "fp_xres": f(sub.get(0xA20E)), "fp_unit": int(unit) if unit else None,
            "make": str(ex.get(0x010F) or "").strip(), "model": str(ex.get(0x0110) or "").strip()}


def mode_for(cam: es.Camera, W: int, H: int) -> Optional[str]:
    for m, (mw, mh) in cam.modes.items():
        if (W, H) in ((mw, mh), (mh, mw)):
            return m
    return None


def focal_px(path: Path, W: int, H: int, camera: str, sensor_width_mm: Optional[float]) -> tuple:
    """-> (f_px, source, mode, details) or (None, reason, None, {})."""
    if camera in es.CAMERAS:
        cam = es.CAMERAS[camera]
        mode = mode_for(cam, W, H)
        w_mm = cam.sensor_w_mm * (cam.width_crop.get(mode, 1.0) if mode else 1.0)
        # sensor width is the LONG side for every profile; portrait frames use the long side too
        f = cam.focal_mm / w_mm * max(W, H)
        src = f"profile {camera} ({cam.focal_mm} mm / {w_mm:.3f} mm sensor width" + \
              (f", mode {mode}" if mode else ", mode unknown: full sensor width assumed") + ")"
        return f, src, mode, {"focal_mm": cam.focal_mm, "sensor_w_mm": w_mm}
    e = _exif(path)
    if e["focal_mm"] and e["fp_xres"] and e["fp_unit"] in (2, 3, 4):
        px_per_mm = e["fp_xres"] / {2: 25.4, 3: 10.0, 4: 1.0}[e["fp_unit"]]
        return e["focal_mm"] * px_per_mm, "exif FocalLength x FocalPlaneXResolution", None, e
    if e["f35"]:
        return e["f35"] * math.hypot(W, H) / DIAG_35MM, \
            "exif FocalLengthIn35mmFilm (diagonal convention, 43.27 mm)", None, e
    if e["focal_mm"] and sensor_width_mm:
        return e["focal_mm"] / sensor_width_mm * max(W, H), \
            f"exif FocalLength + --sensor_width_mm {sensor_width_mm}", None, e
    return None, "no usable intrinsics (no profile, no EXIF focal plane / 35mm-equivalent, no --sensor_width_mm)", None, e


# ───────────────────────── evaluation ─────────────────────────
def _stats(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": float(np.median(v)), "mean": float(np.mean(v)),
            "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}


def evaluate(photos: Path, heights: Path, ends_src: dict, target_m: float,
             sensor_width_mm: Optional[float]) -> dict:
    from PIL import Image
    rows_h = {r["stem"]: r for r in csv.DictReader(open(heights, encoding="utf-8"))}
    files = {p.stem: p for p in photos.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT}
    sizes = {}
    for s, p in files.items():
        with Image.open(p) as im:
            sizes[s] = im.size
    ends = ends_src["fn"](sizes) if callable(ends_src.get("fn")) else ends_src["ends"]
    per, skipped = [], []
    for stem in sorted(files):
        p = files[stem]
        if stem not in rows_h:
            skipped.append({"stem": stem, "reason": "not in heights.csv"})
            continue
        if stem not in ends:
            skipped.append({"stem": stem, "reason": "no endpoint annotation"})
            continue
        h = float(rows_h[stem]["height_m"])
        camera = (rows_h[stem].get("camera") or "exif").strip()
        W, H = sizes[stem]
        f, src, mode, det = focal_px(p, W, H, camera, sensor_width_mm)
        (x1, y1), (x2, y2) = ends[stem]
        measured = math.hypot(x2 - x1, y2 - y1)
        xmp = es.read_dji_xmp(p)
        pitch = xmp.get("gimbal_pitch_deg")
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rec = {"stem": stem, "camera": camera, "image_w": W, "image_h": H, "mode": mode,
               "height_m": h, "intrinsics_source": src, "focal_px": f, "measured_px": measured,
               "gimbal_pitch_deg": pitch,
               "oblique_flag": (pitch is not None and abs(pitch + 90.0) > PITCH_TOL_DEG),
               "target_centre_offset_frac": math.hypot(cx - W / 2, cy - H / 2) / (0.5 * math.hypot(W, H))}
        if f is None or measured <= 0 or h <= 0:
            rec.update(predicted_px=None, rel_error=None, abs_rel_error=None)
            skipped.append({"stem": stem, "reason": src if f is None else "zero length / height"})
        else:
            pred = target_m * f / h
            rec.update(predicted_px=pred, rel_error=(pred - measured) / measured,
                       abs_rel_error=abs(pred - measured) / measured,
                       ratio_measured_over_predicted=measured / pred)
        per.append(rec)

    ok = [r for r in per if r.get("abs_rel_error") is not None]
    summary = {"all": _stats([r["abs_rel_error"] for r in ok]),
               "by_camera": {c: _stats([r["abs_rel_error"] for r in ok if r["camera"] == c])
                             for c in sorted({r["camera"] for r in ok})},
               "by_height_m": {f"{hh:g}": _stats([r["abs_rel_error"] for r in ok if r["height_m"] == hh])
                               for hh in sorted({r["height_m"] for r in ok})},
               "by_mode": {str(m): _stats([r["abs_rel_error"] for r in ok if r["mode"] == m])
                           for m in sorted({r["mode"] for r in ok}, key=str)},
               "signed_rel_error": _stats([r["rel_error"] for r in ok]),
               "n_oblique_flagged": sum(bool(r["oblique_flag"]) for r in ok),
               "n_skipped": len(skipped)}
    # implied video crop: predicted (full sensor width) / measured = fraction of sensor width read out
    crop = {}
    for m in sorted({r["mode"] for r in ok if r["mode"] and r["mode"].startswith("video")}):
        rs = [r for r in ok if r["mode"] == m and r["camera"] in es.CAMERAS]
        if rs:
            frac = [r["predicted_px"] / r["measured_px"] for r in rs]
            crop[m] = {"n": len(rs), "implied_width_fraction_read_out_median": float(np.median(frac)),
                       "implied_crop_factor_median": float(np.median([1 / x for x in frac])),
                       "configured_width_crop": es.CAMERAS[rs[0]["camera"]].width_crop.get(m, 1.0),
                       "note": "width fraction < 1 means the mode reads a cropped sensor area; set "
                               "ewads_survey.Camera.width_crop[mode] to it (re-run the H2 check afterwards)"}
    return {"per_image": per, "skipped": skipped, "summary": summary, "implied_video_crop": crop,
            "target_m": target_m}


def check_target(res: dict, targets: Optional[Path]) -> Optional[dict]:
    if not targets or not targets.exists():
        return None
    import ewads_hier_eval as HE
    tl = HE.load_targets(targets)
    sec = tl["spec"].get("sections", {}).get("h2_gsd")
    if not sec:
        return None
    out = []
    for t in sec["targets"]:
        agg = t.get("aggregate", "p95")
        v = res["summary"]["all"].get({"p95": "p95", "median": "median", "max": "max", "mean": "mean"}[agg])
        ok = None if v is None else (v <= t["value"] if t["op"] == "<=" else v < t["value"] if t["op"] == "<"
                                     else v >= t["value"] if t["op"] == ">=" else v > t["value"])
        out.append({"id": t["id"], "metric": t["metric"], "aggregate": agg, "op": t["op"], "target": t["value"],
                    "measured": v, "n": res["summary"]["all"].get("n", 0), "pass": ok})
    return {"file": targets.name, "sha256": HE.sha256_file(targets), "freeze_status": tl["status"],
            "lock": tl["lock"], "checks": out}


# ───────────────────────── figure ─────────────────────────
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # dataviz reference palette slots 1-3 (validated, light)
MARKERS = ["o", "s", "^"]                            # secondary encoding: identity never colour-alone
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def scatter(res: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in res["per_image"] if r.get("predicted_px") is not None]
    if not ok:
        return None
    def group(r):                                  # camera + mode: a video crop shows as its own series
        return r["camera"] + (f" · {r['mode']}" if r.get("mode") else "")
    cams = sorted({group(r) for r in ok})
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=200)
    ax.set_axisbelow(True)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    lo = 0.9 * min(min(r["predicted_px"], r["measured_px"]) for r in ok)
    hi = 1.1 * max(max(r["predicted_px"], r["measured_px"]) for r in ok)
    xs = np.array([lo, hi])
    ax.fill_between(xs, 0.90 * xs, 1.10 * xs, color=GRID, lw=0, zorder=1,   # |pred - meas| <= 0.1 meas
                    label="±10% (H2 tolerance band)")
    ax.plot(xs, xs, color=INK2, lw=1.2, zorder=2, label="identity (predicted = measured)")
    for i, c in enumerate(cams[:len(SERIES)]):
        rs = [r for r in ok if group(r) == c]
        ax.scatter([r["measured_px"] for r in rs], [r["predicted_px"] for r in rs], s=34,
                   marker=MARKERS[i], color=SERIES[i], edgecolor=SURFACE, linewidth=1.0, zorder=3,
                   label=f"{c} (n={len(rs)})")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color="#efeeea", lw=0.6, zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("measured length of the 1 m target (px)", color=INK, fontsize=9)
    ax.set_ylabel("predicted length from height + camera geometry (px)", color=INK, fontsize=9)
    s = res["summary"]["all"]
    ax.set_title(f"H2 — GSD model vs reference target  (n={s.get('n', 0)}, |rel. error| p95 = "
                 f"{s.get('p95', float('nan')):.3f})", color=INK, fontsize=9.5, loc="left")
    leg = ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    if len(cams) > len(SERIES):
        ax.text(0.99, 0.01, f"+{len(cams) - len(SERIES)} more camera/mode groups in h2_gsd.csv", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--heights", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--endpoints")
    g.add_argument("--cvat_xml")
    g.add_argument("--yolo_pose_dir")
    ap.add_argument("--cvat_label", default=None)
    ap.add_argument("--target_m", type=float, default=1.0)
    ap.add_argument("--sensor_width_mm", type=float, default=None)
    ap.add_argument("--out_dir", default="results_bundle")
    ap.add_argument("--targets", default=str(ROOT / "targets_v2.json"))
    a = ap.parse_args(argv)

    if a.endpoints:
        src = {"ends": endpoints_from_csv(Path(a.endpoints)), "kind": f"endpoints csv {a.endpoints}"}
    elif a.cvat_xml:
        src = {"ends": endpoints_from_cvat(Path(a.cvat_xml), a.cvat_label), "kind": f"cvat xml {a.cvat_xml}"}
    else:
        d = Path(a.yolo_pose_dir)
        src = {"fn": lambda sizes: endpoints_from_yolo_pose(d, sizes), "kind": f"yolo pose {d}"}
    res = evaluate(Path(a.photos), Path(a.heights), src, a.target_m, a.sensor_width_mm)
    res.update(created_utc=datetime.now(timezone.utc).isoformat(), annotation_source=src["kind"],
               assumptions="nadir pinhole camera, target on flat ground at the measured height; "
                           f"oblique if |gimbal pitch + 90| > {PITCH_TOL_DEG} deg (DJI XMP only)",
               tool_sha256=__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest())
    res["targets"] = check_target(res, Path(a.targets) if a.targets else None)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = scatter(res, out / "h2_gsd_scatter.png")
    res["figure"] = str(png) if png else None
    (out / "h2_gsd.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    cols = ["stem", "camera", "mode", "height_m", "image_w", "image_h", "intrinsics_source", "focal_px",
            "predicted_px", "measured_px", "rel_error", "abs_rel_error", "gimbal_pitch_deg", "oblique_flag",
            "target_centre_offset_frac"]
    with open(out / "h2_gsd.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(res["per_image"])

    print(f"\n  H2 GSD validation: {res['summary']['all'].get('n', 0)} images scored, "
          f"{len(res['skipped'])} skipped ({src['kind']})")
    for r in res["per_image"]:
        pred = "-" if r["predicted_px"] is None else f"{r['predicted_px']:.1f}"
        err = "-" if r["abs_rel_error"] is None else f"{r['abs_rel_error']:.3f}"
        print(f"    {r['stem']:<24} {r['camera']:<12} h={r['height_m']:<6g} pred {pred}  meas "
              f"{r['measured_px']:.1f}  |err| {err}  [{r['intrinsics_source']}]"
              + ("  OBLIQUE" if r["oblique_flag"] else ""))
    s = res["summary"]["all"]
    if s.get("n"):
        print(f"  |rel. error|: median {s['median']:.4f}  mean {s['mean']:.4f}  p95 {s['p95']:.4f}  max {s['max']:.4f}")
    for m, c in res["implied_video_crop"].items():
        print(f"  {m}: implied sensor-width fraction read out {c['implied_width_fraction_read_out_median']:.3f} "
              f"(crop factor {c['implied_crop_factor_median']:.3f}, n={c['n']}; configured {c['configured_width_crop']})")
    t = res["targets"]
    if t:
        print(f"  targets {t['file']}: {t['freeze_status']}")
        for c in t["checks"]:
            v = "no verdict" if c["pass"] is None else ("PASS" if c["pass"] else "FAIL")
            print(f"    {c['id']}  {c['metric']} {c['aggregate']} {c['op']} {c['target']}  measured "
                  f"{'-' if c['measured'] is None else round(c['measured'], 4)}  (n={c['n']})  {v}")
    print(f"  -> {out / 'h2_gsd.json'}, h2_gsd.csv, {res['figure']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
