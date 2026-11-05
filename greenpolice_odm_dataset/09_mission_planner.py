#!/usr/bin/env python3
"""
09_mission_planner.py — what a flight can and cannot prove, before you fly it.

Turns camera geometry + the detector's resolvability rules + measured Jetson latency into:
  * per-class MAXIMUM altitude (smallest realistic instance >= 16 px, thin parts >= 1 px)
  * for each planned leg: GSD, footprint, motion-blur speed limit, how often a frame must be
    processed so every object is fully seen `views` times, tiles/s, Jetson duty cycle, ha/hour
  * which classes are assessable / marginal / NOT assessable on that leg -- the list that goes into
    the enforcement report next to every "nothing found"

Typical use
  # 1) measure per-tile latency on the Jetson (writes benchmark_report.json)
  python 07_edge_benchmark.py --mode trt --engine_path best_fp16.engine --benchmark
  # 2) plan
  python 09_mission_planner.py --mode video_4k --legs 60:10 25:8 8:3 \\
         --bench_report benchmark_results/benchmark_report.json --area_ha 2
  python 09_mission_planner.py --mode stills_3x2 --target_class cpu_chip --speed 1.5
  python 09_mission_planner.py --class_table            # max altitude per class, every mode

Assumptions (stated in the output too): nadir gimbal (-90 deg), flat ground, altitude = height
above the ground under the aircraft (DJI RelativeAltitude is above TAKE-OFF -- on sloping sites
correct for it), image top = flight direction, no sensor crop in video modes (verify once: photograph
a 1 m target at a known height and compare the measured px with this tool's GSD).
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ewads_survey as S  # noqa: E402

RENDER_PROFILES = {"high_surveillance": (40, 80), "mid_confirmation": (15, 40), "low_evidence": (3, 15)}


def tile_ms_from_report(path: str) -> float:
    rep = json.loads(Path(path).read_text())
    lat = rep.get("latency", rep)
    for k in ("per_tile_ms_effective", "median_ms"):
        if lat.get(k):
            return float(lat[k])
    raise SystemExit(f"[ERROR] no latency numbers in {path} (run 07 --benchmark first)")


def class_table(cam: S.Camera, px: float) -> None:
    modes = list(cam.modes)
    print(f"\n  Max altitude (m) at which the SMALLEST realistic instance spans >= {px:.0f} px "
          f"(thin parts >= 1 px)\n")
    print("  " + f"{'class':<17}{'small (m)':>10}" + "".join(f"{m:>13}" for m in modes))
    for c in S.EWASTE_CLASSES:
        row = "".join(f"{S.max_altitude_m(c, m, cam, px):>13.1f}" for m in modes)
        print(f"  {c:<17}{S.class_small_size_m(c):>10.3f}{row}")
    print("\n  Renderer altitude profiles (02): " +
          ", ".join(f"{k} {a}-{b} m" for k, (a, b) in RENDER_PROFILES.items()))


def print_plan(p: S.MissionPlan, area_ha=None) -> dict:
    hz = f"{p.process_hz:.2f}" if math.isfinite(p.process_hz) else "inf"
    print(f"\n  ── leg: {p.mode} @ {p.altitude_m:.1f} m, {p.speed_mps:.1f} m/s " + "─" * 30)
    print(f"    GSD {p.gsd_cm:.2f} cm/px   footprint {p.footprint_m[0]:.1f} x {p.footprint_m[1]:.1f} m "
          f"(across x along track)")
    print(f"    motion blur {p.blur_px:.2f} px   (speed limit for <=1 px: {p.speed_limit_blur_mps:.1f} m/s)")
    if p.mode.startswith("stills"):
        iv = 1 / p.process_hz if p.process_hz > 0 else float("inf")
        print(f"    one photo every {p.spacing_m:.1f} m  ->  every {iv:.1f} s "
              f"(check the camera's minimum interval-shot time can keep up)")
    else:
        print(f"    process a frame every {p.spacing_m:.1f} m  ->  {hz} Hz "
              f"(not the camera's 30 fps: the frames in between overlap)")
    print(f"    pyramid levels {p.max_levels} (+final)  ->  {p.tiles_per_frame} tiles/frame, "
          f"{p.tiles_per_s:.1f} tiles/s")
    if p.duty_cycle is not None:
        print(f"    Jetson duty cycle {p.duty_cycle:.0%}  (at {p.tile_ms:.1f} ms/tile)")
    else:
        print("    Jetson duty cycle: unknown -- pass --tile_ms or --bench_report (measure, don't guess)")
    print(f"    coverage {p.area_rate_ha_per_h:.1f} ha/h"
          + (f"  ->  {area_ha} ha takes {60 * area_ha / p.area_rate_ha_per_h:.1f} min on-line (+ turns)"
             if area_ha else ""))
    r = p.resolvability
    print(f"    ASSESSABLE      : {', '.join(r['assessable']) or '-'}")
    print(f"    marginal (8-16) : {', '.join(r['marginal']) or '-'}")
    print(f"    NOT assessable  : {', '.join(r['not_assessable']) or '-'}")
    for w in p.warnings:
        print(f"    ⚠  {w}")
    d = p.__dict__.copy()
    d["area_minutes"] = 60 * area_ha / p.area_rate_ha_per_h if area_ha else None
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="phantom4pro", choices=list(S.CAMERAS))
    ap.add_argument("--mode", default="video_4k")
    ap.add_argument("--legs", nargs="*", default=[], help="altitude_m:speed_mps pairs, e.g. 60:10 25:8 8:3")
    ap.add_argument("--target_class", default=None,
                    help="plan the highest leg at which this class is assessable (x --margin)")
    ap.add_argument("--margin", type=float, default=0.9, help="altitude safety factor for --target_class")
    ap.add_argument("--speed", type=float, default=None,
                    help="m/s for --target_class legs (default: the 1 px blur limit, capped at 10)")
    ap.add_argument("--exposure", type=float, default=1 / 1000, help="shutter time in s (1/1000 default)")
    ap.add_argument("--views", type=int, default=2, help="processed frames that must contain each object")
    ap.add_argument("--max_object_m", type=float, default=1.0)
    ap.add_argument("--largest_object_m", type=float, default=3.5, help="sizes the pyramid levels")
    ap.add_argument("--side_overlap", type=float, default=0.2)
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--tile_ms", type=float, default=None, help="measured per-tile latency on the Jetson")
    ap.add_argument("--bench_report", default=None, help="07's benchmark_report.json (reads the latency)")
    ap.add_argument("--assess_px", type=float, default=S.ASSESS_PX)
    ap.add_argument("--area_ha", type=float, default=None)
    ap.add_argument("--class_table", action="store_true")
    ap.add_argument("--json", default=None, help="write the plan here")
    a = ap.parse_args()

    cam = S.CAMERAS[a.camera]
    if a.mode not in cam.modes:
        raise SystemExit(f"[ERROR] mode {a.mode} not in {list(cam.modes)}")
    tile_ms = a.tile_ms or (tile_ms_from_report(a.bench_report) if a.bench_report else None)

    print(f"\n  EWADS mission planner — {cam.name}, {a.mode} {cam.modes[a.mode][0]}x{cam.modes[a.mode][1]}, "
          f"HFOV {cam.hfov_deg(a.mode):.1f}°, shutter 1/{1 / a.exposure:.0f} s, views {a.views}")
    print("  Assumes nadir gimbal, flat ground, altitude above ground under the aircraft.")

    if a.class_table:
        class_table(cam, a.assess_px)

    legs = []
    for s in a.legs:
        alt, spd = (float(x) for x in s.split(":"))
        legs.append((alt, spd))
    if a.target_class:
        if a.target_class not in S.CLASS_REAL_SIZE_M:
            raise SystemExit(f"[ERROR] unknown class {a.target_class}")
        alt = S.max_altitude_m(a.target_class, a.mode, cam, a.assess_px) * a.margin
        g = S.gsd_m(alt, cam.modes[a.mode][0], cam, a.mode)
        spd = a.speed or min(10.0, S.max_speed_mps(g, a.exposure, 1.0))
        print(f"\n  Target {a.target_class}: smallest instance {S.class_small_size_m(a.target_class) * 100:.1f} cm "
              f"-> fly at <= {alt:.1f} m (x{a.margin} margin), <= {spd:.1f} m/s")
        legs.append((alt, spd))
    if not legs and not a.class_table:
        legs = [(60, 10), (25, 8), (8, 3)]
        print("\n  (no --legs given: showing one leg per renderer altitude profile)")

    out = {"camera": cam.name, "mode": a.mode, "tile_ms": tile_ms, "legs": []}
    for alt, spd in legs:
        p = S.plan_mission(cam, a.mode, alt, spd, a.exposure, a.views, a.max_object_m, a.side_overlap,
                           a.tile, a.overlap, tile_ms, 1.0, a.largest_object_m)
        out["legs"].append(print_plan(p, a.area_ha))

    if legs:
        print("\n  Reading this: piles / appliances / screens are found from the high leg; small Tier-1 items")
        print("  (batteries, boards, chips) need a low evidence pass over the hotspots it flags. A sweep of a")
        print("  whole site at chip altitude is rarely affordable -- check the ha/h line before promising one.")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  Plan written to {a.json}")


if __name__ == "__main__":
    main()
