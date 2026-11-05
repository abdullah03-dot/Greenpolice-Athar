#!/usr/bin/env python3
"""
Paired comparison of two models scored on the SAME images (ATHAR, 2026-10-08).

Used for the camera-mismatch ablation (true camera vs 50 mm-equivalent render, both evaluated on the
synthetic held-out set at the TRUE geometry) and usable for AdaIN vs no-AdaIN.  Inputs are 08's
detection caches (detections.npz) for model A and model B on one --data_dir, and each model's OWN
operating threshold (fixed on validation).  Every resample draws one set of images and scores both
models on it -> a 95% CI for the difference B - A (ewads_hier_eval.paired_bootstrap_diff).

  python tools/compare_evals.py --data_dir data/splits/main/heldout \\
      --a_cache eval/abl_true_heldout/detections.npz --a_conf 0.31 --a_name true_camera \\
      --b_cache eval/abl_mis_heldout/detections.npz  --b_conf 0.28 --b_name mismatch_50mm \\
      --meta data/splits/main/heldout/scene_index.jsonl --out results_bundle/ablation_camera.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ewads_hier_eval as HE   # noqa: E402
import ewads_survey as es      # noqa: E402


def _load_08():
    spec = importlib.util.spec_from_file_location("eval08", ROOT / "08_tiled_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_cache(p: Path) -> dict:
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files if k != "__timing__"}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--a_cache", required=True); ap.add_argument("--a_conf", type=float, required=True)
    ap.add_argument("--b_cache", required=True); ap.add_argument("--b_conf", type=float, required=True)
    ap.add_argument("--a_name", default="A"); ap.add_argument("--b_name", default="B")
    ap.add_argument("--meta", default=None, help="altitudes (for site linking in metres), as 08 --meta")
    ap.add_argument("--camera", default="phantom4pro")
    ap.add_argument("--assess_px", type=float, default=16)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    m08 = _load_08()
    _, gts, sizes = m08.load_gt(Path(a.data_dir), return_sizes=True)
    da, db = load_cache(Path(a.a_cache)), load_cache(Path(a.b_cache))
    miss = [s for s in gts if s not in da or s not in db]
    if miss:
        raise SystemExit(f"[ERROR] {len(miss)} images lack detections in one cache (e.g. {miss[:3]}): "
                         f"both models must be scored on the SAME images")
    alt = m08.load_meta(a.meta)
    cam = es.CAMERAS.get(a.camera)
    gsd = {s: es.gsd_m(alt[s], sizes[s][0], cam) for s in gts if s in alt and alt[s] > 0} if (alt and cam) else {}
    specs = HE.DEFAULT_CI_SPECS + [
        {"name": "env_map50", "metric": "map50", "min_px": a.assess_px},
        {"name": "env_tier1_recall", "metric": "recall", "classes": "tier1", "min_px": a.assess_px}]
    res = HE.paired_bootstrap_diff(gts, da, db, specs, a.a_conf, a.b_conf, n_boot=a.bootstrap, seed=a.seed, gsd=gsd)
    out = {"created_utc": datetime.now(timezone.utc).isoformat(), "data_dir": str(Path(a.data_dir).resolve()),
           "n_images": len(gts), "a": {"name": a.a_name, "cache": a.a_cache, "cache_sha256": sha256_file(Path(a.a_cache)),
                                       "conf": a.a_conf},
           "b": {"name": a.b_name, "cache": a.b_cache, "cache_sha256": sha256_file(Path(a.b_cache)), "conf": a.b_conf},
           "difference": "b - a", "metrics": res, "assess_px": a.assess_px,
           "tool_sha256": sha256_file(Path(__file__))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  {a.b_name} - {a.a_name} on {len(gts)} images (paired bootstrap, {a.bootstrap} resamples)")
    print(f"  {'metric':<18}{a.a_name:>14}{a.b_name:>16}{'diff':>10}   95% CI of diff")
    for k, r in res.items():
        ci = "n/a" if r["diff_lo"] is None else f"[{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}]"
        print(f"  {k:<18}{r['a']:>14.4f}{r['b']:>16.4f}{r['diff']:>+10.4f}   {ci}")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
