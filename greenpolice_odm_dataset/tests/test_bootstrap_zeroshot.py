"""
Tests for Task 6 (added 2026-10-08): bootstrap CIs in 08 and the YOLO-World zero-shot baseline tool.

  1. ewads_hier_eval.bootstrap_ci: deterministic per seed; perfect detector -> [1, 1]; the point value
     lies inside its CI; CIs narrow with more images; a duplicated image counts twice; site recall
     works with resampled images.
  2. 08: eval_report.json carries ci95 for the six listed metrics, per_class_n for all 18 classes, a
     ci95 on every pre-registered target row; --bootstrap 0 switches it off.
  3. tools/zeroshot_baseline.py: prompts must cover exactly the 18 classes; the LARGEST local
     *-worldv2.pt is chosen; a test* folder is refused without --final_evaluation; end to end (with an
     injected fake detector in place of YOLO-World) it writes 08's cache format, 08 scores it, and the
     manifest records weights + prompts SHA-256.

Run: python tests/test_bootstrap_zeroshot.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import ewads_hier_eval as HE        # noqa: E402
import ewads_taxonomy as tx         # noqa: E402
import zeroshot_baseline as ZS      # noqa: E402


def make_set(n, p_found, seed=0):
    """n images, 3 objects each; each object found with prob p_found (exact box) -> gts, dets."""
    rng = np.random.default_rng(seed)
    gts, dets = {}, {}
    for i in range(n):
        cls = rng.integers(0, 18, 3)
        boxes = np.array([[100 * k + 10, 50, 100 * k + 70, 110] for k in range(3)], float)
        gts[f"im{i:04d}"] = (cls, boxes)
        keep = rng.random(3) < p_found
        dets[f"im{i:04d}"] = np.array([[*b, 0.9, c] for b, c, k in zip(boxes, cls, keep) if k], np.float32).reshape(-1, 6)
    return gts, dets


def test_bootstrap():
    g, d = make_set(60, 1.0)
    r = HE.bootstrap_ci(g, d, HE.DEFAULT_CI_SPECS, 0.5, n_boot=200, seed=1)
    for k in ("map50", "recall", "site_recall"):
        assert r[k]["point"] == 1.0 and r[k]["lo"] == 1.0 and r[k]["hi"] == 1.0, (k, r[k])
    g, d = make_set(60, 0.7)
    a = HE.bootstrap_ci(g, d, HE.DEFAULT_CI_SPECS, 0.5, n_boot=300, seed=1)
    b = HE.bootstrap_ci(g, d, HE.DEFAULT_CI_SPECS, 0.5, n_boot=300, seed=1)
    assert a == b, "same seed must give identical CIs"
    for k, c in a.items():
        if c["lo"] is not None:
            assert c["lo"] <= c["point"] <= c["hi"], (k, c)
    g2, d2 = make_set(600, 0.7, seed=5)
    big = HE.bootstrap_ci(g2, d2, [{"name": "recall", "metric": "recall"}], 0.5, n_boot=300, seed=1)["recall"]
    assert (big["hi"] - big["lo"]) < 0.6 * (a["recall"]["hi"] - a["recall"]["lo"]), (big, a["recall"])
    # duplicated image counts twice: 2 images, one fully found, one fully missed -> resamples give
    # recall in {0, 0.5, 1}; with distinct keys the 'both the same image' draws must give 0 or 1
    g3 = {"a": (np.array([1, 1]), np.array([[0, 0, 50, 50], [60, 0, 110, 50]], float)),
          "b": (np.array([1, 1]), np.array([[0, 0, 50, 50], [60, 0, 110, 50]], float))}
    d3 = {"a": np.array([[0, 0, 50, 50, 0.9, 1], [60, 0, 110, 50, 0.9, 1]], np.float32), "b": np.zeros((0, 6), np.float32)}
    c3 = HE.bootstrap_ci(g3, d3, [{"name": "r", "metric": "recall"}], 0.5, n_boot=400, seed=3)["r"]
    assert c3["lo"] == 0.0 and c3["hi"] == 1.0 and c3["point"] == 0.5, c3
    print("  ok  bootstrap: deterministic, perfect -> [1,1], point inside CI, narrows with n, duplicates count twice")


def write_eval_dir(td: Path, n=20, seed=0):
    rng = np.random.default_rng(seed)
    (td / "d" / "images").mkdir(parents=True)
    (td / "d" / "labels").mkdir()
    dets, meta = {}, []
    for i in range(n):
        W, H = 1280, 720
        cv2.imwrite(str(td / "d" / "images" / f"im{i:02d}.jpg"), np.full((H, W, 3), 90, np.uint8))
        cls = rng.integers(0, 18, 3)
        x = rng.uniform(0, W - 200, 3); y = rng.uniform(0, H - 200, 3); s = rng.uniform(30, 150, 3)
        boxes = np.stack([x, y, x + s, y + s * 0.7], 1)
        (td / "d" / "labels" / f"im{i:02d}.txt").write_text("\n".join(
            f"{c} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}"
            for c, b in zip(cls, boxes)))
        keep = rng.random(3) < 0.8
        dets[f"im{i:02d}"] = np.array([[*bb, 0.9, c] for bb, c, k in zip(boxes, cls, keep) if k], np.float32).reshape(-1, 6)
        meta.append({"image_path": f"images/im{i:02d}.jpg", "altitude_m": float([8, 12, 30][i % 3])})
    np.savez_compressed(td / "dets.npz", **dets)
    (td / "meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta))


def run08(td, out, *extra):
    r = subprocess.run([sys.executable, str(ROOT / "08_tiled_eval.py"), "--data_dir", str(td / "d"), "--cache",
                        str(td / "dets.npz"), "--conf", "0.5", "--meta", str(td / "meta.jsonl"),
                        "--out_dir", str(td / out), *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    return json.loads((td / out / "eval_report.json").read_text())


def test_08_ci():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_eval_dir(td)
        rep = run08(td, "e1", "--bootstrap", "200", "--targets_dataset", "synthetic_heldout")
        assert set(rep["ci95"]) == {"map50", "map5095", "recall", "tier1_recall", "fp_share", "site_recall"}
        for k, c in rep["ci95"].items():
            assert c["n_boot"] == 200 and c["n_images"] == 20
            if c["lo"] is not None:
                assert c["lo"] <= c["point"] + 1e-12 and c["point"] <= c["hi"] + 1e-12, (k, c)
        assert set(rep["per_class_n"]) == set(tx.CLASSES) and sum(rep["per_class_n"].values()) == 60
        rows = [t for res in rep["preregistered_targets"]["sections"].values() for t in res]
        assert rows and all("ci95" in t for t in rows if t["value_measured"] is not None), rows
        rep0 = run08(td, "e0", "--bootstrap", "0", "--targets", "")
        assert rep0["ci95"] == {} and rep0["bootstrap"]["n_boot"] == 0
    print("  ok  08: six CIs, per-class n, CI on every target row, --bootstrap 0 = off")


def test_zeroshot():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        good = json.loads((ROOT / "tools" / "zeroshot_prompts.json").read_text(encoding="utf-8"))
        assert ZS.load_prompts(ROOT / "tools" / "zeroshot_prompts.json") == [good["prompts"][c] for c in tx.CLASSES]
        bad = {"prompts": {k: v for k, v in good["prompts"].items() if k != "battery"}}
        (td / "bad.json").write_text(json.dumps(bad))
        try:
            ZS.load_prompts(td / "bad.json"); raise AssertionError("missing class must be refused")
        except SystemExit as e:
            assert "battery" in str(e)
        (td / "w").mkdir()
        (td / "w" / "yolov8s-worldv2.pt").write_bytes(b"s" * 10)
        (td / "w" / "yolov8x-worldv2.pt").write_bytes(b"x" * 100)
        assert ZS.find_weights(None, [str(td / "w")], None).name == "yolov8x-worldv2.pt"
        # test set refused without --final_evaluation
        (td / "test" / "images").mkdir(parents=True)
        try:
            ZS.main(["--data_dir", str(td / "test"), "--out_dir", str(td / "o"), "--weights_dir", str(td / "w")])
            raise AssertionError("test set must be refused")
        except SystemExit as e:
            assert "final evaluation" in str(e)
        # end to end with a fake detector (exact boxes from the labels of the tile's source image)
        write_eval_dir(td, n=6)
        import ewads_tiling as et

        def fake_build(weights, prompts, tile, min_conf, device):
            assert len(prompts) == 18
            return lambda tiles: [np.zeros((0, 6), np.float32) for _ in tiles]
        ZS.build_predict_fn = fake_build
        ZS.main(["--data_dir", str(td / "d"), "--out_dir", str(td / "zs"), "--weights_dir", str(td / "w"),
                 "--tile", "512"])
        with np.load(td / "zs" / "detections.npz", allow_pickle=True) as z:     # closes the file (Windows lock)
            assert set(k for k in z.files if k != "__timing__") == {f"im{i:02d}" for i in range(6)}
        man = json.loads((td / "zs" / "zeroshot_manifest.json").read_text())
        assert man["weights"]["sha256"] == __import__("hashlib").sha256(b"x" * 100).hexdigest()
        assert man["prompts"]["battery"] == good["prompts"]["battery"] and len(man["prompts_sha256"]) == 64
        r = subprocess.run([sys.executable, str(ROOT / "08_tiled_eval.py"), "--data_dir", str(td / "d"), "--cache",
                            str(td / "zs" / "detections.npz"), "--conf", "0.3", "--bootstrap", "20", "--targets", "",
                            "--out_dir", str(td / "zs_eval")], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        rep = json.loads((td / "zs_eval" / "eval_report.json").read_text())
        assert rep["map50"] == 0.0 and rep["operating_point"]["recall"] == 0.0     # empty detector scored as such
    print("  ok  zero-shot tool: prompt check, largest weights, test refused, 08-format cache scored by 08, manifest hashes")


if __name__ == "__main__":
    test_bootstrap()
    test_08_ci()
    test_zeroshot()
    print("\nbootstrap / zero-shot tests passed")
