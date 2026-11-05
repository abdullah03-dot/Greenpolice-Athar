"""
Tests for tools/gsd_validation.py (H2), Task 5 (added 2026-10-08).

Photos are synthesised with KNOWN geometry and the tool must recover exactly what was put in:
  - Phantom 4 Pro stills (profile intrinsics, mode stills_3x2) at 8 / 15 / 25 m, annotations off by a
    known relative error -> per-image errors, p95 and the H2 check reproduce those values;
  - Phantom 4 Pro 4K video frames rendered with a 0.8 sensor-width crop -> implied width fraction 0.8;
  - phone photos: EXIF FocalLength x FocalPlaneXResolution, and FocalLengthIn35mmFilm (diagonal);
  - an image with no usable intrinsics is skipped with a reason; an oblique DJI XMP pitch is flagged;
  - endpoints CSV, CVAT XML and YOLO-pose give identical results; JSON, CSV and PNG are written.

Run: python tests/test_gsd_validation.py
"""
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import gsd_validation as GV  # noqa: E402

F_P4P_STILLS = 8.8 / 13.2 * 5472          # px
F_P4P_4K_CROPPED = 8.8 / (13.2 * 0.8) * 3840   # what a cropped 4K mode really images


def _img(path, size, exif=None, xmp=None):
    im = Image.new("RGB", size, (100, 100, 100))
    ex = Image.Exif()
    if exif:
        ifd = ex.get_ifd(0x8769)
        for k, v in exif.items():
            ifd[k] = v
    kw = {"xmp": xmp.encode()} if xmp else {}          # embedded near the file start, as DJI does
    im.save(path, quality=60, exif=ex.tobytes() if exif else b"", **kw)


def build(td: Path):
    photos = td / "photos"
    photos.mkdir()
    rows, ends, truth = [], {}, {}
    def add(stem, size, h, camera, f_true_px, err=0.0, exif=None, xmp=None, f_tool_px=None):
        _img(photos / f"{stem}.jpg", size, exif, xmp)
        W, H = size
        L = 1.0 * f_true_px / h * (1 + err)           # measured length the annotation will show
        x1, y1 = W / 2 - L / 2, H / 2
        ends[stem] = ((x1, y1), (x1 + L, y1))
        rows.append({"stem": stem, "height_m": h, "camera": camera})
        pred = 1.0 * (f_tool_px or f_true_px) / h
        truth[stem] = abs(pred - L) / L
    for i, (h, e) in enumerate([(8, 0.02), (15, -0.03), (25, 0.01), (8, -0.01), (15, 0.03), (25, 0.0)]):
        add(f"p4p_still_{i}", (5472, 3648), h, "phantom4pro", F_P4P_STILLS, e)
    for i, h in enumerate([10, 20]):                   # 4K video, really cropped to 0.8 of the width
        add(f"p4p_4k_{i}", (3840, 2160), h, "phantom4pro", F_P4P_4K_CROPPED,
            f_tool_px=8.8 / 13.2 * 3840)
    # phone, focal plane resolution: 4.2 mm lens, 1000 px/mm
    add("phone_fp", (4032, 3024), 12, "pixel", 4.2 * 1000, 0.02,
        exif={0x920A: 4.2, 0xA20E: 1000.0, 0xA210: 4})
    # phone, 35 mm-equivalent 26 mm (diagonal convention)
    f35 = 26 * math.hypot(4032, 3024) / GV.DIAG_35MM
    add("phone_35", (4032, 3024), 18, "pixel", f35, -0.02, exif={0xA405: 26})
    # no intrinsics at all
    _img(photos / "nothing.jpg", (4032, 3024))
    rows.append({"stem": "nothing", "height_m": 10, "camera": "exif"})
    ends["nothing"] = ((10, 10), (500, 10))
    # oblique DJI shot
    add("p4p_oblique", (5472, 3648), 12, "phantom4pro", F_P4P_STILLS, 0.0,
        xmp='<x drone-dji:GimbalPitchDegree="-60.0" drone-dji:RelativeAltitude="+12.0"/>')
    with open(td / "heights.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "height_m", "camera"])
        w.writeheader(); w.writerows(rows)
    with open(td / "endpoints.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["stem", "x1", "y1", "x2", "y2"])
        for s, ((a, b), (c, d)) in ends.items():
            w.writerow([s, a, b, c, d])
    # same endpoints as CVAT XML (polyline) and YOLO pose
    xml = ['<annotations>']
    (td / "pose").mkdir()
    for s, ((a, b), (c, d)) in ends.items():
        W, H = Image.open(photos / f"{s}.jpg").size
        xml.append(f'<image name="{s}.jpg" width="{W}" height="{H}">'
                   f'<polyline label="target_1m" points="{a:.4f},{b:.4f};{c:.4f},{d:.4f}"/></image>')
        (td / "pose" / f"{s}.txt").write_text(
            f"0 0.5 0.5 0.1 0.1 {a / W:.9f} {b / H:.9f} 2 {c / W:.9f} {d / H:.9f} 2\n")
    xml.append("</annotations>")
    (td / "ann.xml").write_text("".join(xml), encoding="utf-8")
    return truth


def run(td, *ann):
    out = td / ("out_" + ann[0].strip("-"))
    GV.main(["--photos", str(td / "photos"), "--heights", str(td / "heights.csv"), *ann,
             "--out_dir", str(out), "--targets", str(ROOT / "targets_v2.json")])
    return json.loads((out / "h2_gsd.json").read_text(encoding="utf-8")), out


def test_gsd():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        truth = build(td)
        res, out = run(td, "--endpoints", str(td / "endpoints.csv"))
        per = {r["stem"]: r for r in res["per_image"]}
        for s, e in truth.items():
            assert math.isclose(per[s]["abs_rel_error"], e, rel_tol=1e-6, abs_tol=1e-9), (s, per[s]["abs_rel_error"], e)
        assert per["p4p_still_0"]["mode"] == "stills_3x2" and "profile phantom4pro" in per["p4p_still_0"]["intrinsics_source"]
        assert per["p4p_4k_0"]["mode"] == "video_4k"
        assert "FocalPlaneXResolution" in per["phone_fp"]["intrinsics_source"]
        assert "35mm" in per["phone_35"]["intrinsics_source"]
        assert any(s["stem"] == "nothing" and "no usable intrinsics" in s["reason"] for s in res["skipped"])
        assert per["p4p_oblique"]["oblique_flag"] is True and per["p4p_still_0"]["oblique_flag"] is False
        crop = res["implied_video_crop"]["video_4k"]
        assert crop["n"] == 2 and math.isclose(crop["implied_width_fraction_read_out_median"], 0.8, rel_tol=1e-6)
        vals = np.array(list(truth.values()))
        assert math.isclose(res["summary"]["all"]["p95"], float(np.percentile(vals, 95)), rel_tol=1e-9)
        h2 = res["targets"]["checks"][0]
        assert h2["id"] == "H2" and h2["measured"] == res["summary"]["all"]["p95"]
        assert h2["pass"] == (h2["measured"] <= 0.10) and res["targets"]["freeze_status"] in (
            "NOT FROZEN", "FROZEN", "MODIFIED AFTER FREEZE")
        assert (out / "h2_gsd.csv").exists() and (out / "h2_gsd_scatter.png").stat().st_size > 10_000
        rows = list(csv.DictReader(open(out / "h2_gsd.csv", encoding="utf-8")))
        assert len(rows) == len(res["per_image"])
        # the three annotation formats agree
        r2, _ = run(td, "--cvat_xml", str(td / "ann.xml"))
        r3, _ = run(td, "--yolo_pose_dir", str(td / "pose"))
        for other in (r2, r3):
            o = {r["stem"]: r for r in other["per_image"]}
            for s in truth:
                assert math.isclose(o[s]["measured_px"], per[s]["measured_px"], rel_tol=1e-6), (s, o[s]["measured_px"])
    print("  ok  H2: per-image errors exact, modes + EXIF intrinsics, crop 0.8 recovered, oblique flagged, "
          "3 annotation formats agree, outputs written")


if __name__ == "__main__":
    test_gsd()
    print("\ngsd-validation tests passed")
