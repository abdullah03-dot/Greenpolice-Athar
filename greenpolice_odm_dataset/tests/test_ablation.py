"""
Tests for the camera-mismatch ablation, Task 6.2 (added 2026-10-08).

  1. 02's ablation profile: same sensor + resolutions as the real camera, focal = 50 mm-equivalent
     (crop factor 43.27 / diag(13.2, 8.8)); the real profile is unchanged.  (Parsed with ast: 02 only
     runs inside Blender.  The override itself was checked with a live Blender render: GSD x 0.480.)
  2. 05 --max_epochs caps every stage's epochs (and patience) and leaves the rest of the schedule alone.
  3. ewads_hier_eval.paired_bootstrap_diff: identical models -> diff 0 with CI [0, 0]; a better model B
     -> diff > 0 with the whole CI above 0; deterministic per seed.
  4. tools/compare_evals.py end to end on two 08 caches; refuses caches that miss images.

Run: python tests/test_ablation.py
"""
import ast
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import ewads_hier_eval as HE      # noqa: E402
import compare_evals as CE        # noqa: E402


def renderer_constants():
    tree = ast.parse((ROOT / "02_blenderproc_synthetic_v2.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CAMERA_PROFILES":
            return ast.literal_eval(node.value)
    raise AssertionError("CAMERA_PROFILES not found")


def test_profile():
    cp = renderer_constants()
    real, mis = cp["dji_phantom4_pro"], cp["p4p_mismatch_50mm_equiv"]
    assert real["focal_mm"] == 8.8 and real["sensor_width_mm"] == 13.2
    crop = math.hypot(36, 24) / math.hypot(13.2, 8.8)
    assert abs(mis["focal_mm"] - 50 / crop) < 0.01 and abs(real["focal_mm"] * crop - 24) < 0.05
    assert mis["sensor_width_mm"] == real["sensor_width_mm"] and mis["resolutions"] == real["resolutions"]
    print("  ok  ablation profile: 50 mm-equivalent focal on the real sensor/resolutions; real profile unchanged")


def test_epoch_cap():
    spec = importlib.util.spec_from_file_location("train05", ROOT / "05_train_rtdetr.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for st in (1, 2, 3):
        cfg = m.SYNTH_STAGE_CONFIGS[st]
        capped = m.apply_epoch_cap(cfg, 4)
        assert capped["epochs"] == min(cfg["epochs"], 4) and capped["patience"] <= 4
        assert {k: v for k, v in capped.items() if k not in ("epochs", "patience")} == \
               {k: v for k, v in cfg.items() if k not in ("epochs", "patience")}
        assert m.apply_epoch_cap(cfg, None) == cfg
    print("  ok  05 --max_epochs caps epochs/patience only")


def test_paired_and_tool():
    import test_bootstrap_zeroshot as TB
    g, d = TB.make_set(80, 0.6, seed=2)
    same = HE.paired_bootstrap_diff(g, d, d, HE.DEFAULT_CI_SPECS, 0.5, 0.5, n_boot=200, seed=1)
    for k, r in same.items():
        assert r["diff"] == 0 and r["diff_lo"] == 0 and r["diff_hi"] == 0, (k, r)
    _, d_good = TB.make_set(80, 0.95, seed=2)                 # same images/boxes, more objects found
    better = HE.paired_bootstrap_diff(g, d, d_good, [{"name": "recall", "metric": "recall"}], 0.5, 0.5,
                                      n_boot=300, seed=1)["recall"]
    assert better["diff"] > 0 and better["diff_lo"] > 0, better
    again = HE.paired_bootstrap_diff(g, d, d_good, [{"name": "recall", "metric": "recall"}], 0.5, 0.5,
                                     n_boot=300, seed=1)["recall"]
    assert again == better
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        TB.write_eval_dir(td, n=15)
        with np.load(td / "dets.npz") as z:
            a = {k: z[k] for k in z.files}
        b = {k: v[:1] for k, v in a.items()}                  # B keeps only its first detection per image
        np.savez_compressed(td / "a.npz", **a)
        np.savez_compressed(td / "b.npz", **b)
        CE.main(["--data_dir", str(td / "d"), "--a_cache", str(td / "a.npz"), "--a_conf", "0.5", "--a_name", "full",
                 "--b_cache", str(td / "b.npz"), "--b_conf", "0.5", "--b_name", "one_det",
                 "--meta", str(td / "meta.jsonl"), "--bootstrap", "100", "--out", str(td / "cmp.json")])
        out = json.loads((td / "cmp.json").read_text())
        assert out["metrics"]["recall"]["diff"] < 0 and out["a"]["name"] == "full" and out["n_images"] == 15
        assert len(out["a"]["cache_sha256"]) == 64 and "env_map50" in out["metrics"]
        np.savez_compressed(td / "short.npz", **{k: v for i, (k, v) in enumerate(a.items()) if i > 0})
        try:
            CE.main(["--data_dir", str(td / "d"), "--a_cache", str(td / "a.npz"), "--a_conf", "0.5",
                     "--b_cache", str(td / "short.npz"), "--b_conf", "0.5", "--out", str(td / "x.json")])
            raise AssertionError("missing images must be refused")
        except SystemExit as e:
            assert "SAME images" in str(e)
    print("  ok  paired bootstrap: identical -> 0 [0,0], better -> CI above 0, deterministic; compare_evals end to end")


if __name__ == "__main__":
    test_profile()
    test_epoch_cap()
    test_paired_and_tool()
    print("\nablation tests passed")
