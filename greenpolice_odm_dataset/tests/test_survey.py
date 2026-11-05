"""
Tests for ewads_survey.py (no GPU, no blenderproc).

  1. Constants mirror the renderer: CLASS_REAL_SIZE_M, SPAWN_MIN_PX, MIN_FEATURE_PX, the camera,
     the small-variant sizes -- parsed from 02 with `ast`, so a later edit of 02 that forgets
     ewads_survey fails here instead of silently mis-labelling assessability in court records.
  2. Geometry: GSD and max altitude round-trip; levels_needed really makes the largest object
     whole in some tile (checked against ewads_tiling's own tile plan).
  3. DJI SRT parsing (Phantom 4 Pro and newer formats) and XMP parsing.
  4. CoverageScheduler on a synthetic nadir flight: every ground row ends up fully inside >= views
     processed frames, with telemetry and with image correlation.
Run:  python tests/test_survey.py
"""
import ast
import math
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ewads_survey as S   # noqa: E402
import ewads_tiling as et  # noqa: E402


def _consts_from_02():
    tree = ast.parse((ROOT / "02_blenderproc_synthetic_v2.py").read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("CLASS_REAL_SIZE_M", "SPAWN_MIN_PX", "MIN_FEATURE_PX", "DEFAULT_SCALE_JITTER",
                        "CLASS_MIN_FEATURE_M", "EWASTE_CLASSES", "CAMERA_PROFILES"):
                try:
                    out[name] = ast.literal_eval(node.value)
                except ValueError:
                    pass
            if name == "CLASS_SIZE_BY_SHAPE":      # rules hold lambdas: pull every (size, jitter)
                d = {}
                for k, v in zip(node.value.keys, node.value.values):
                    d[ast.literal_eval(k)] = [(ast.literal_eval(r.elts[1]), ast.literal_eval(r.elts[2]))
                                              for r in v.elts]
                out[name] = d
    return out


def test_consistency_with_renderer():
    c = _consts_from_02()
    assert c["CLASS_REAL_SIZE_M"] == S.CLASS_REAL_SIZE_M, "class sizes drifted from 02"
    assert c["SPAWN_MIN_PX"] == S.ASSESS_PX
    assert c["MIN_FEATURE_PX"] == S.MIN_FEATURE_PX
    assert c["DEFAULT_SCALE_JITTER"][0] == S.SCALE_JITTER_LO
    assert c["CLASS_MIN_FEATURE_M"] == S.CLASS_MIN_FEATURE_M
    if "EWASTE_CLASSES" in c:
        assert c["EWASTE_CLASSES"] == S.EWASTE_CLASSES, "class order drifted from 02"
    for cls, rules in c["CLASS_SIZE_BY_SHAPE"].items():   # smallest instance any rule can produce
        smallest = min(size * jit[0] for size, jit in rules)
        assert abs(S.SMALL_VARIANT_M[cls] - smallest) < 1e-9, f"small variant of {cls} drifted"
    cam = c.get("CAMERA_PROFILES", {}).get("dji_phantom4_pro")
    if cam:
        p4 = S.CAMERAS["phantom4pro"]
        assert cam["sensor_width_mm"] == p4.sensor_w_mm and cam["focal_mm"] == p4.focal_mm
        for (wh, _w) in cam["resolutions"]:
            assert tuple(wh) in set(p4.modes.values()), f"renderer resolution {wh} missing from modes"
    import importlib.util
    spec = importlib.util.spec_from_file_location("b07", ROOT / "07_edge_benchmark.py")
    src = (ROOT / "07_edge_benchmark.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "EWASTE_CLASSES":
            assert ast.literal_eval(node.value) == S.EWASTE_CLASSES, "07 class order differs"
    print("  ok  constants match 02 (sizes, thresholds, camera, class order) and 07")


def test_geometry():
    cam = S.CAMERAS["phantom4pro"]
    for mode in cam.modes:
        for cls in S.EWASTE_CLASSES:
            alt = S.max_altitude_m(cls, mode, cam)
            g = S.gsd_m(alt, cam.modes[mode][0], cam, mode)
            st = S.resolvability(g * 0.999)[cls]          # float edge: exactly-at-limit may round down
            assert st["status"] == "assessable", (mode, cls, st)
            st2 = S.resolvability(g * 1.05)[cls]
            assert st2["status"] != "assessable", (mode, cls, st2)
    # P4P 4K at 10 m: 2*10*tan(HFOV/2)/3840 = 10*13.2/(8.8*3840) = 3.906 mm
    assert abs(S.gsd_m(10, 3840, cam) - 0.00390625) < 1e-9
    assert S.resolvability(None)["cpu_chip"]["status"] == "unknown"
    # levels_needed: largest object must be whole in some tile at a kept level
    rng = np.random.default_rng(0)
    for (W, H) in [(3840, 2160), (5472, 3648), (1920, 1080)]:
        for alt in (5, 12, 25, 45, 80):
            g = S.gsd_m(alt, W, cam)
            L = S.levels_needed(g, (W, H), 768, 128, 3.5)
            specs = et.plan_tiles(W, H, 768, 128)
            kept = [s for s in specs if s.level < L or s.final]
            px = 3.5 / g
            for _ in range(20):
                if px >= min(W, H):
                    break
                x0 = rng.uniform(0, W - px); y0 = rng.uniform(0, H - px)
                ok = False
                for s in kept:
                    bx0, by0, bx1, by1 = x0 * s.scale, y0 * s.scale, (x0 + px) * s.scale, (y0 + px) * s.scale
                    if bx0 >= s.x0 and by0 >= s.y0 and bx1 <= s.x0 + s.vw and by1 <= s.y0 + s.vh:
                        ok = True
                        break
                assert ok, (W, H, alt, L)
    print("  ok  GSD / max-altitude round trip, levels_needed keeps the largest object whole")


def test_srt_and_xmp():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.SRT"
        p.write_text("1\n00:00:00,000 --> 00:00:00,033\nF/2.8, SS 320, ISO 100, EV 0, GPS (55.300000, 25.200000, 19), "
                     "D 25.88m, H 12.40m, H.S 3.20m/s, V.S 0.00m/s\n\n"
                     "2\n00:00:00,033 --> 00:00:00,066\n<font size=\"28\">FrameCnt: 2 [iso: 100] [shutter: 1/1000.0] "
                     "[latitude: 25.200100] [longitude: 55.300200] [rel_alt: 30.500 abs_alt: 52.1]</font>\n")
        r = S.parse_dji_srt(p)
        assert r[0]["lat"] == 25.2 and r[0]["lon"] == 55.3 and r[0]["rel_alt_m"] == 12.4
        assert r[0]["speed_mps"] == 3.2 and abs(r[0]["exposure_s"] - 1 / 320) < 1e-12
        assert r[1]["lat"] == 25.2001 and r[1]["rel_alt_m"] == 30.5 and abs(r[1]["exposure_s"] - 1e-3) < 1e-12
        assert abs(r[1]["t0_s"] - 0.033) < 1e-9
        x = (b'<x:xmpmeta><rdf:Description drone-dji:RelativeAltitude="+12.30" '
             b'drone-dji:GimbalPitchDegree="-89.90" drone-dji:AbsoluteAltitude="+64.3"/>'
             b'<drone-dji:FlightXSpeed>-1.5</drone-dji:FlightXSpeed></x:xmpmeta>')
        j = Path(td) / "x.jpg"
        j.write_bytes(b"\xff\xd8\xff\xe1" + struct.pack(">H", len(x) + 2) + x + b"\xff\xd9")
        m = S.read_dji_xmp(j)
        assert m["rel_alt_m"] == 12.3 and m["gimbal_pitch_deg"] == -89.9 and m["vx_mps"] == -1.5
    print("  ok  DJI SRT (P4P + newer formats) and XMP parsing")


def _coverage_check(processed_offsets, H, obj_px, views):
    """every object of obj_px fully inside >= views processed frames (1-D along track)."""
    total = processed_offsets[-1] + H
    worst = 99
    for y in range(H, int(total - obj_px - H), 7):   # steady state: skip the first and last frame-length
        n = sum(1 for o in processed_offsets if o <= y and y + obj_px <= o + H)
        worst = min(worst, n)
    return worst


def test_scheduler():
    import cv2
    cam = S.CAMERAS["phantom4pro"]
    alt, v, fps = 20.0, 6.0, 30.0
    W, H = 960, 540
    g = S.gsd_m(alt, W, cam)
    # long textured ground strip
    rng = np.random.default_rng(1)
    strip = cv2.GaussianBlur((rng.random((H * 8, W)) * 255).astype(np.uint8), (0, 0), 2)
    step = v / fps / g                                  # px per frame
    obj_px = 1.0 / g
    for src in ("telemetry", "image"):
        sch = S.CoverageScheduler(H, (S.capture_spacing_m(H * g, 1.0, 2)) / (H * g), max_gap_s=10)
        offs, y, i = [], 0.0, 0
        while y + H < strip.shape[0]:
            fr = strip[int(round(y)):int(round(y)) + H]
            if src == "telemetry":
                go, why = sch.update(i / fps, gsd=g, speed_mps=v, dt_s=None if i == 0 else 1 / fps)
            else:
                go, why = sch.update(i / fps, gray=fr)
            if go:
                offs.append(int(round(y)))
            y += step; i += 1
        worst = _coverage_check(offs, H, obj_px, 2)
        assert worst >= 2, (src, worst, offs)
        print(f"  ok  scheduler ({src:9s}): {len(offs)}/{i} frames processed "
              f"({len(offs) / (i / fps):.2f} Hz), "
              f"every 1 m object in >= {worst} processed frames")


if __name__ == "__main__":
    test_consistency_with_renderer()
    test_geometry()
    test_srt_and_xmp()
    test_scheduler()
    print("\nsurvey tests passed")
