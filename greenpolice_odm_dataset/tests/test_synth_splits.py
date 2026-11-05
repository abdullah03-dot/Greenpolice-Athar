"""
Tests for the synthetic-only regime, Task 2 (added 2026-10-08).

  1. tools/make_synth_splits.py: stratified scene split, no scene in two splits, sequences kept whole,
     LOEO folds (train/val from the other two environments, heldout = every scene of the third),
     manifest file lists hash-verifiable, refuses a non-empty --out_dir and clashing val_seq ids.
  2. its output is read as-is by 03b (--dry_run) -- the folder layout contract.
  3. 03 records the TRAIN sources in dataset.yaml; 05 --synthetic_only refuses a dataset with 'real'
     in TRAIN (before any training starts).
  4. ewads_survey.sequence_offsets_m: centred, along the yawed image axis, 11-scheduler spacing,
     terrain check.

Run: python tests/test_synth_splits.py
"""
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import ewads_survey as es          # noqa: E402
import make_synth_splits as mss    # noqa: E402

ENVS = ("desert", "european_urban", "european_rural")
PROFILES = ("high_surveillance", "mid_confirmation", "low_evidence")


def fake_render(root: Path, n_per=10, seq_env="desert", seq_frames=3, start=0, size=(320, 180)):
    """renderer-shaped output: <root>/<env>/{images,labels,scene_index.jsonl}; one env also has sequences."""
    rng = np.random.default_rng(start)
    sid = start
    for env in ENVS:
        d = root / env
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(exist_ok=True)
        with open(d / "scene_index.jsonl", "a", encoding="utf-8") as idx:
            for prof in PROFILES:
                for _ in range(n_per):
                    nf = seq_frames if env == seq_env else 1
                    for k in range(nf):
                        stem = f"synthetic_{env}_{sid:06d}" + (f"_f{k:02d}" if nf > 1 else "")
                        img = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
                        cv2.imwrite(str(d / "images" / f"{stem}.jpg"), img)
                        (d / "labels" / f"{stem}.txt").write_text(f"{int(rng.integers(0, 18))} 0.5 0.5 0.2 0.3\n")
                        idx.write(json.dumps({
                            "scene_id": sid, "environment": env, "altitude_profile": prof,
                            "altitude_m": 20.0, "capture_mode": "video_4k", "negative": False,
                            "sequence_id": sid if nf > 1 else None, "frame_index": k, "n_frames": nf,
                            "image_path": f"C:\\render\\{env}\\images\\{stem}.jpg",
                            "label_path": f"C:\\render\\{env}\\labels\\{stem}.txt"}) + "\n")
                    sid += 1
    return sid


def keys_of(split_dir: Path):
    out = set()
    for line in (split_dir / "scene_index.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.add((r["environment"], r["sequence_id"] if r["sequence_id"] is not None else r["scene_id"]))
    return out


def test_splits():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        nxt = fake_render(td / "syn")
        fake_render(td / "valseq", n_per=1, start=nxt + 1000)          # distinct ids, sequences
        m = mss.main(["--synthetic_dir", str(td / "syn"), "--out_dir", str(td / "sp"),
                      "--val_seq_dir", str(td / "valseq"), "--seed", "7"])
        sp = td / "sp"
        tr, va, ho = keys_of(sp / "main/synthetic"), keys_of(sp / "main/val"), keys_of(sp / "main/heldout")
        assert not (tr & va) and not (tr & ho) and not (va & ho), "scene in two main splits"
        assert len(tr) + len(va) + len(ho) == 90
        # stratified: every (env, profile) stratum of 10 gets ceil(1)=1 heldout and 1 val
        for env in ENVS:
            idx = [json.loads(l) for l in (sp / "main/heldout/scene_index.jsonl").read_text().splitlines()]
            for prof in PROFILES:
                n = len({r["scene_id"] for r in idx if r["environment"] == env and r["altitude_profile"] == prof})
                assert n == 1, (env, prof, n)
        # sequences whole: every desert sequence has all 3 frames in exactly one split
        for split in ("main/synthetic", "main/val", "main/heldout"):
            rs = [json.loads(l) for l in (sp / split / "scene_index.jsonl").read_text().splitlines()]
            per = {}
            for r in rs:
                if r["sequence_id"] is not None:
                    per.setdefault(r["sequence_id"], set()).add(r["frame_index"])
            assert all(v == {0, 1, 2} for v in per.values()), (split, per)
        # LOEO
        for env in ENVS:
            f = sp / f"loeo_{env}"
            assert {k[0] for k in keys_of(f / "heldout")} == {env} and len(keys_of(f / "heldout")) == 30
            assert env not in {k[0] for k in keys_of(f / "synthetic") | keys_of(f / "val")}
            assert keys_of(f / "synthetic") <= tr and keys_of(f / "val") <= va
        # val_seq copied whole, sequences present
        assert len(list((sp / "main/val_seq/images").iterdir())) == 3 * 3 + 2 * 3   # desert seqs + 2 single envs
        # manifest: list hashes and per-file hashes verify
        for name, s in m["splits"].items():
            lst = sp / s["file_list"]
            assert hashlib.sha256(lst.read_bytes()).hexdigest() == s["file_list_sha256"]
            for line in lst.read_text(encoding="utf-8").splitlines()[:5]:
                h, rel = line.split("  ", 1)
                assert hashlib.sha256((sp / rel).read_bytes()).hexdigest() == h, rel
        # refuses to overwrite
        try:
            mss.main(["--synthetic_dir", str(td / "syn"), "--out_dir", str(td / "sp")])
            raise AssertionError("non-empty out_dir must be refused")
        except SystemExit as e:
            assert "not empty" in str(e)
        # refuses val_seq ids that clash with the main render
        try:
            mss.main(["--synthetic_dir", str(td / "syn"), "--out_dir", str(td / "sp2"),
                      "--val_seq_dir", str(td / "syn")])
            raise AssertionError("clashing val_seq must be refused")
        except SystemExit as e:
            assert "distinct --start_id" in str(e)
        # 03b reads the layout as-is
        r = subprocess.run([sys.executable, str(ROOT / "03b_tile_dataset.py"), "--input_dir", str(sp / "main"),
                            "--output_dir", str(td / "tiled"), "--splits", "synthetic", "val", "--dry_run"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "Tiles:" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    print("  ok  make_synth_splits: stratified, no leakage, sequences whole, LOEO, hashes verify, 03b reads it")


def test_heldout_seq():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        nxt = fake_render(td / "syn", n_per=3)
        fake_render(td / "vseq", n_per=1, start=nxt + 1000)
        fake_render(td / "hseq", n_per=1, start=nxt + 5000)
        m = mss.main(["--synthetic_dir", str(td / "syn"), "--out_dir", str(td / "sp"), "--no_hash_files", "--no_loeo",
                      "--val_seq_dir", str(td / "vseq"), "--heldout_seq_dir", str(td / "hseq")])
        assert m["splits"]["main/heldout_seq"]["n_frames"] == m["splits"]["main/val_seq"]["n_frames"] == 15
        assert not keys_of(td / "sp/main/heldout_seq") & keys_of(td / "sp/main/val_seq")
        try:                                         # same ids as the validation sequences -> refused
            mss.main(["--synthetic_dir", str(td / "syn"), "--out_dir", str(td / "sp2"), "--no_hash_files",
                      "--val_seq_dir", str(td / "vseq"), "--heldout_seq_dir", str(td / "vseq")])
            raise AssertionError("clashing heldout_seq must be refused")
        except SystemExit as e:
            assert "distinct --start_id" in str(e)
    print("  ok  heldout_seq: copied whole, disjoint from val_seq and the main render")


def test_sources_and_guard():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for split in ("synthetic", "val"):
            (td / "in" / split / "images").mkdir(parents=True)
            (td / "in" / split / "labels").mkdir()
            for i in range(3):
                cv2.imwrite(str(td / "in" / split / "images" / f"{split}_{i}.jpg"),
                            np.full((768, 768, 3), 100 + i, np.uint8))
                (td / "in" / split / "labels" / f"{split}_{i}.txt").write_text("3 0.5 0.5 0.1 0.1\n")
        r = subprocess.run([sys.executable, str(ROOT / "03_augmentation_pipeline.py"), "--input_dir", str(td / "in"),
                            "--output_dir", str(td / "aug"), "--n_aug_per_image", "1", "--rfs_t", "0",
                            "--image_size", "768"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        import yaml
        y = yaml.safe_load((td / "aug" / "dataset.yaml").read_text(encoding="utf-8"))
        assert y["ewads"]["sources"] == ["synthetic"], y["ewads"]
        # a dataset whose TRAIN includes real images must be refused in synthetic-only mode
        bad = td / "bad.yaml"
        bad.write_text((td / "aug" / "dataset.yaml").read_text(encoding="utf-8")
                       .replace('sources: ["synthetic"]', 'sources: ["synthetic", "real"]'), encoding="utf-8")
        r = subprocess.run([sys.executable, str(ROOT / "05_train_rtdetr.py"), "--dataset_yaml", str(bad),
                            "--output_dir", str(td / "runs")], capture_output=True, text=True)
        assert r.returncode == 1 and "synthetic-only training, but TRAIN was built from" in r.stdout, \
            r.stdout[-2000:] + r.stderr[-2000:]
        assert not (td / "runs" / "Stage1_FrozenBackbone").exists()
    print("  ok  03 records TRAIN sources; 05 --synthetic_only refuses real data before training")


def test_sequence_geometry():
    r = es.sequence_offsets_m(3, 18.0, 32.0, 0.0, views=3, max_object_m=1.0)
    assert math.isclose(r["spacing_m"], 17.0 / 3) and math.isclose(r["overlap_frac"], 1 - (17 / 3) / 18)
    (x0, y0), (x1, y1), (x2, y2) = r["offsets"]
    assert abs(x1) < 1e-9 and abs(y1) < 1e-9 and math.isclose(y2, -y0) and abs(x0) < 1e-9     # yaw 0: along +y
    r90 = es.sequence_offsets_m(3, 18.0, 32.0, 90.0)
    assert math.isclose(r90["offsets"][2][0], -r90["spacing_m"]) and abs(r90["offsets"][2][1]) < 1e-9
    tiny = es.sequence_offsets_m(5, 1.2, 2.0, 0.0, max_object_m=1.19)       # (0.01/3) < 5% floor
    assert math.isclose(tiny["spacing_m"], 0.06)
    assert not es.sequence_offsets_m(9, 18.0, 32.0, 0.0, terrain_half_side_m=30.0)["fits"]
    assert es.sequence_offsets_m(3, 18.0, 32.0, 0.0, terrain_half_side_m=46.7)["fits"]
    print("  ok  sequence geometry: centred, yawed axis, scheduler spacing (5% floor), terrain check")


def test_selection_guard():
    """Model selection (08 --calibrate, 10 --gate_data, 11 deriving stage thresholds) refuses
    held-out / test folders unless explicitly overridden."""
    import ewads_hier_eval as HE
    assert HE.selection_data_problem("data/splits/main/val") is None
    assert HE.selection_data_problem("data/splits/main/val_seq") is None
    for bad in ("data/splits/main/heldout", "data/splits/loeo_desert/heldout", "data/test", "data/test_seq"):
        assert HE.selection_data_problem(bad), bad
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        d = td / "splits" / "main" / "heldout"
        (d / "images").mkdir(parents=True)
        (d / "labels").mkdir()
        cv2.imwrite(str(d / "images" / "a.jpg"), np.full((200, 300, 3), 90, np.uint8))
        (d / "labels" / "a.txt").write_text("3 0.5 0.5 0.2 0.2\n")
        np.savez_compressed(td / "c.npz", a=np.array([[120, 80, 180, 120, 0.9, 3]], np.float32))
        base = [sys.executable, str(ROOT / "08_tiled_eval.py"), "--data_dir", str(d), "--cache", str(td / "c.npz"),
                "--calibrate", "--targets", ""]
        r = subprocess.run(base + ["--out_dir", str(td / "e1")], capture_output=True, text=True)
        assert r.returncode != 0 and "--calibrate refused" in (r.stdout + r.stderr), r.stdout[-1500:] + r.stderr[-1500:]
        r = subprocess.run(base + ["--out_dir", str(td / "e2"), "--allow_selection_on_heldout"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "OVERRIDE" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
        r = subprocess.run([sys.executable, str(ROOT / "10_jetson_build_engine.py"), "--weights", "x.pt",
                            "--gate_data", str(d), "--out", str(td / "j")], capture_output=True, text=True)
        assert r.returncode != 0 and "--gate_data refused" in (r.stdout + r.stderr), r.stdout[-1500:] + r.stderr[-1500:]
        r = subprocess.run([sys.executable, str(ROOT / "11_evidence_pipeline.py"), "--weights", "x.pt",
                            "--data_dir", str(d), "--out_dir", str(td / "ev11")], capture_output=True, text=True)
        assert r.returncode != 0 and "deriving stage thresholds refused" in (r.stdout + r.stderr), \
            r.stdout[-1500:] + r.stderr[-1500:]
        assert not (td / "ev11").exists()                    # refused before creating any output
    print("  ok  selection guard: 08 --calibrate / 10 --gate_data / 11 thresholds refuse held-out/test (override recorded)")


if __name__ == "__main__":
    test_sequence_geometry()
    test_splits()
    test_heldout_seq()
    test_sources_and_guard()
    test_selection_guard()
    print("\nsynthetic-split tests passed")
