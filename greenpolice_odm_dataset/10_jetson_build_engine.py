#!/usr/bin/env python3
"""
10_jetson_build_engine.py — build FP16 and INT8 TensorRT engines ON THE JETSON, then keep INT8
only if it earns its place.

Why on the Jetson: a TensorRT engine is tied to the GPU and TensorRT version that built it.
Why a gate: INT8 post-training quantisation is not free for a transformer detector like RT-DETR
(softmax/attention and box regression are range-sensitive).  It can lose more accuracy than it is
worth, mostly on small objects.  So both engines are scored with the SAME full-frame tiled
evaluation (08, through the same TensorRT backend 07 deploys) and timed with 07's benchmark, and
INT8 is chosen only if the accuracy drop stays inside your limits.

Steps
  1. environment check (TensorRT, CUDA, Ultralytics export API: `quantize=` vs older `half=/int8=`)
  2. export   best.pt -> <out>/ewads_fp16_b<B>.engine  and  <out>/ewads_int8_b<B>.engine
              (dynamic batch up to B: no padding waste when a frame's tile count is not a multiple of B;
               INT8 calibrated on tools/build_calib_set.py output)
  3. accuracy 08_tiled_eval.py --engine on --gate_data (full-resolution VAL frames, never test)
              FP16 picks the operating threshold (--calibrate); INT8 is scored at the same threshold
  4. speed    07_edge_benchmark.py --benchmark (per-tile ms, batched)
  5. decide   INT8 if  envelope mAP drop <= --max_map_drop  AND  recall drop <= --max_recall_drop
              -> <out>/engine_decision.json + the 09_mission_planner command with the measured ms

Usage (on the Jetson, inside the ewads env)
  python tools/build_calib_set.py --tiled_dir data/tiled --splits real synthetic --out data/calib
  python 10_jetson_build_engine.py --weights best.pt --calib data/calib/calib.yaml \\
         --gate_data data/val_full --out runs/jetson
  python 10_jetson_build_engine.py --out runs/jetson --skip_build      # re-score existing engines
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ewads_hier_eval as HE  # noqa: E402  (selection guard)


def env_report() -> dict:
    rep = {}
    try:
        import tensorrt as trt
        rep["tensorrt"] = trt.__version__
    except ImportError:
        rep["tensorrt"] = None
    try:
        import torch
        rep["torch"] = torch.__version__
        rep["cuda"] = torch.cuda.is_available()
        rep["gpu"] = torch.cuda.get_device_name(0) if rep["cuda"] else None
    except ImportError:
        rep["torch"] = None
        rep["cuda"] = False
    try:
        import ultralytics
        from ultralytics.cfg import DEFAULT_CFG_DICT
        rep["ultralytics"] = ultralytics.__version__
        rep["export_api"] = "quantize" if "quantize" in DEFAULT_CFG_DICT else "half/int8"
    except ImportError:
        rep["ultralytics"] = None
    p = Path("/etc/nv_tegra_release")
    rep["jetson"] = p.read_text().split(",")[0].strip() if p.exists() else None
    return rep


def export_engine(weights: str, precision: str, imgsz: int, batch: int, workspace: float,
                  calib: str, api: str, out: Path) -> Path:
    sys.path.insert(0, str(HERE))
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("eval08", HERE / "08_tiled_eval.py")
    m08 = module_from_spec(spec)
    spec.loader.exec_module(m08)
    model = m08.load_ultralytics(weights)
    kw = dict(format="engine", imgsz=imgsz, batch=batch, dynamic=True, device=0, simplify=True)
    if workspace:
        kw["workspace"] = workspace
    if api == "quantize":
        kw["quantize"] = 8 if precision == "int8" else 16
    else:
        kw["int8" if precision == "int8" else "half"] = True
    if precision == "int8":
        if not calib or not Path(calib).exists():
            raise SystemExit("[ERROR] INT8 needs --calib (tools/build_calib_set.py output)")
        kw["data"] = calib
        kw["fraction"] = 1.0
    print(f"\n  exporting {precision.upper()}: {kw}")
    t0 = time.time()
    f = Path(model.export(**kw))
    dst = out / f"ewads_{precision}_b{batch}.engine"
    shutil.move(str(f), dst)
    print(f"  -> {dst}  ({(time.time() - t0) / 60:.1f} min)")
    return dst


def run(cmd: list) -> None:
    print("  $ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode:
        raise SystemExit(f"[ERROR] command failed ({r.returncode})")


def score(engine: Path, gate: str, out: Path, conf, calibrate: bool, max_images) -> dict:
    ev = out / f"eval_{engine.stem}"
    cmd = [sys.executable, HERE / "08_tiled_eval.py", "--engine", engine, "--data_dir", gate, "--out_dir", ev]
    if calibrate:
        cmd += ["--calibrate"]
    if conf is not None:
        cmd += ["--conf", f"{conf:.4f}"]
    if max_images:
        cmd += ["--max_images", str(max_images)]
    run(cmd)
    rep = json.loads((ev / "eval_report.json").read_text())
    env = rep.get("envelope", {})
    return {"map5095": rep["map5095"], "map50": rep["map50"], "conf": rep["conf"],
            "recall": rep["operating_point"]["recall"], "fp_share": rep["operating_point"]["fp_share"],
            "env_map5095": env.get("map5095"), "env_recall": env.get("recall"),
            "per_class_ap": rep.get("per_class_ap", {}), "report": str(ev / "eval_report.json")}


def bench(engine: Path, out: Path) -> dict:
    bd = out / f"bench_{engine.stem}"
    run([sys.executable, HERE / "07_edge_benchmark.py", "--mode", "trt", "--engine_path", engine,
         "--benchmark", "--conf", "0.25", "--val_dir", "/nonexistent", "--output_dir", bd])
    lat = json.loads((bd / "benchmark_report.json").read_text())["latency"]
    return {"per_tile_ms": lat.get("per_tile_ms_effective", lat["median_ms"]), "p95_ms_batch1": lat["p95_ms"],
            "max_batch": lat.get("max_batch"), "report": str(bd / "benchmark_report.json")}


TIER1 = ("battery", "circuit_board", "crt_monitor")      # same tiers as 07


def decide(res: dict, max_map_drop: float, max_recall_drop: float, max_tier1_drop: float = 0.03) -> dict:
    """Pick INT8 only if it is within the accuracy limits (envelope numbers when present) AND no
    Tier-1 (toxic) class loses more than max_tier1_drop AP -- an average can hide one class collapsing."""
    if "fp16" not in res or "int8" not in res:
        only = next(iter(res))
        return {"choice": only, "reason": "only one engine was built"}
    f, i = res["fp16"], res["int8"]
    key_m = "env_map5095" if f.get("env_map5095") is not None else "map5095"
    key_r = "env_recall" if f.get("env_recall") is not None else "recall"
    dm, dr = f[key_m] - i[key_m], f[key_r] - i[key_r]
    sp = f["per_tile_ms"] / i["per_tile_ms"] if i.get("per_tile_ms") else None
    worst = sorted(((c, f["per_class_ap"].get(c, 0) - i["per_class_ap"].get(c, 0)) for c in f["per_class_ap"]),
                   key=lambda t: -t[1])[:3]
    t1 = {c: f["per_class_ap"].get(c, 0) - i["per_class_ap"].get(c, 0) for c in TIER1 if c in f["per_class_ap"]}
    t1_bad = {c: v for c, v in t1.items() if v > max_tier1_drop}
    ok = dm <= max_map_drop and dr <= max_recall_drop and not t1_bad
    if ok:
        why = f"INT8 within limits (mAP -{dm:.4f} <= {max_map_drop}, recall -{dr:.4f} <= {max_recall_drop})"
    else:
        why = (f"INT8 rejected (mAP -{dm:.4f}, recall -{dr:.4f}; limits {max_map_drop}/{max_recall_drop}"
               + (f"; Tier-1 AP drops {', '.join(f'{c} -{v:.3f}' for c, v in t1_bad.items())} > {max_tier1_drop}"
                  if t1_bad else "") + ")")
    return {"choice": "int8" if ok else "fp16", "map_drop": dm, "recall_drop": dr, "metric": key_m,
            "int8_speedup": sp, "worst_class_drops": worst, "tier1_drops": t1, "reason": why}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="trained best.pt (RT-DETR or YOLO)")
    ap.add_argument("--calib", help="calib.yaml from tools/build_calib_set.py")
    ap.add_argument("--gate_data", help="full-resolution VAL frames: images/ + labels/ (never test)")
    ap.add_argument("--allow_selection_on_heldout", action="store_true",
                    help="override the guard that refuses a held-out/test folder as --gate_data")
    ap.add_argument("--out", default="runs/jetson")
    ap.add_argument("--imgsz", type=int, default=768, help="tile size the model was trained on")
    ap.add_argument("--batch", type=int, default=8, help="max tiles per engine call (dynamic batch)")
    ap.add_argument("--workspace", type=float, default=4.0, help="GiB for the TensorRT builder")
    ap.add_argument("--precisions", nargs="+", default=["fp16", "int8"], choices=["fp16", "int8"])
    ap.add_argument("--conf", type=float, default=None, help="operating threshold (else calibrated on FP16)")
    ap.add_argument("--max_map_drop", type=float, default=0.01)
    ap.add_argument("--max_recall_drop", type=float, default=0.01)
    ap.add_argument("--max_tier1_drop", type=float, default=0.03, help="max AP drop for any Tier-1 class")
    ap.add_argument("--gate_max_images", type=int, default=None)
    ap.add_argument("--skip_build", action="store_true", help="use engines already in --out")
    ap.add_argument("--skip_gate", action="store_true", help="build only")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.gate_data and not a.skip_gate:      # check BEFORE spending time on engine builds
        why = HE.selection_data_problem(a.gate_data)
        if why and not a.allow_selection_on_heldout:
            raise SystemExit(f"[ERROR] --gate_data refused: {why}")
        if why:
            print(f"  WARNING OVERRIDE (--allow_selection_on_heldout): {why}")
    env = env_report()
    print("\n  environment: " + json.dumps(env))
    if not env.get("tensorrt") or not env.get("cuda"):
        raise SystemExit("[ERROR] needs TensorRT + CUDA: run this on the Jetson (JetPack), not on the workstation")
    if not env.get("jetson"):
        print("  ⚠  /etc/nv_tegra_release not found: this does not look like a Jetson. Engines built here "
              "will NOT run on the Orin NX.")

    engines = {}
    for p in a.precisions:
        if a.skip_build:
            f = out / f"ewads_{p}_b{a.batch}.engine"
            if f.exists():
                engines[p] = f
            else:
                print(f"  ⚠  {f} missing, skipping {p}")
        else:
            if not a.weights:
                raise SystemExit("[ERROR] --weights required to build")
            engines[p] = export_engine(a.weights, p, a.imgsz, a.batch, a.workspace, a.calib,
                                       env.get("export_api", "quantize"), out)
    if a.skip_gate or not engines:
        print("\n  built: " + ", ".join(str(e) for e in engines.values()))
        return
    if not a.gate_data:
        raise SystemExit("[ERROR] --gate_data (full-res val frames) required for the accuracy gate")

    res, conf = {}, a.conf
    for p in ("fp16", "int8"):
        if p not in engines:
            continue
        r = score(engines[p], a.gate_data, out, conf, calibrate=(conf is None and p == "fp16"),
                  max_images=a.gate_max_images)
        conf = r["conf"] if conf is None else conf
        r.update(bench(engines[p], out))
        res[p] = r
        print(f"\n  {p.upper()}: mAP {r['map5095']:.4f} (envelope {r['env_map5095']}) | recall {r['recall']:.4f} "
              f"@conf {r['conf']:.2f} | {r['per_tile_ms']:.2f} ms/tile")

    d = decide(res, a.max_map_drop, a.max_recall_drop, a.max_tier1_drop)
    chosen = res[d["choice"]]
    summary = {"environment": env, "engines": {k: str(v) for k, v in engines.items()}, "results": res,
               "decision": d, "chosen_engine": str(engines[d["choice"]]), "operating_conf": conf}
    (out / "engine_decision.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  DECISION: {d['choice'].upper()} -- {d['reason']}")
    if d.get("int8_speedup"):
        print(f"  INT8 speed-up {d['int8_speedup']:.2f}x; biggest per-class AP drops: "
              + ", ".join(f"{c} -{v:.3f}" for c, v in d.get("worst_class_drops", [])))
    print(f"  engine: {engines[d['choice']]}")
    print(f"  next:   python 09_mission_planner.py --tile_ms {chosen['per_tile_ms']:.1f} --legs <alt:speed> ...")
    print(f"          python 07_edge_benchmark.py --mode trt --engine_path {engines[d['choice']]} "
          f"--conf {conf:.2f} --video <flight.MP4> --srt_file <flight.SRT> --auto_levels")
    print(f"  written {out / 'engine_decision.json'}")


if __name__ == "__main__":
    main()
