"""
Tests for tools/dataset_stats.py, Task 7 (added 2026-10-08).  Every number is checked against what
was put on disk:
  - renders (renderer-shaped fake, incl. 3-frame sequences): scenes vs frames, per environment /
    altitude profile / capture mode, instances per class and per environment, size buckets, negatives;
  - tiles per split from tile_index.jsonl; split counts from make_synth_splits' manifest;
  - assets: Objaverse reviewed vs UNREVIEWED (review index.json), procedural, rejected, distractors;
  - licences: CC0 textures / HDRIs from their manifests; Objaverse licences ONLY from the local
    metadata cache, uids not in it reported as unknown.

Run: python tests/test_dataset_stats.py
"""
import gzip
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import dataset_stats as DS         # noqa: E402
import make_synth_splits as MSS    # noqa: E402
import test_synth_splits as TSS    # noqa: E402

U = [f"{i:032x}" for i in range(1, 7)]      # fake Objaverse uids


def build(td: Path):
    TSS.fake_render(td / "syn")                                     # 90 scenes, desert as 3-frame sequences
    for idx in (td / "syn").glob("*/scene_index.jsonl"):            # add resolution + 1 negative scene
        rows = [json.loads(l) for l in idx.read_text(encoding="utf-8").splitlines()]
        for r in rows:
            r["resolution"] = [320, 180]
            r["negative"] = r["scene_id"] == rows[0]["scene_id"] and r["environment"] == "european_urban"
        idx.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (td / "syn" / "desert" / "generation_stats_000000.json").write_text(json.dumps(
        {"visible_unlabelled_per_class": {"cpu_chip": 7, "battery": 2}}))
    MSS.main(["--synthetic_dir", str(td / "syn"), "--out_dir", str(td / "splits"), "--no_hash_files"])
    tdir = td / "tiled"
    tdir.mkdir()
    (tdir / "tile_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (
        [{"split": "synthetic", "n_labels": 2}] * 5 + [{"split": "synthetic", "n_labels": 0}] * 3 +
        [{"split": "val", "n_labels": 1}] * 2)))
    m = td / "assets" / "models"
    for c in ("battery", "crt_monitor"):
        (m / c).mkdir(parents=True)
    for u in U[:3]:
        (m / "battery" / f"{u}.glb").write_bytes(b"x")
    (m / "battery" / "proc_battery_0000.glb").write_bytes(b"x")
    (m / "crt_monitor" / f"{U[3]}.glb").write_bytes(b"x")
    (m / "_rejected" / "battery").mkdir(parents=True)
    (m / "_rejected" / "battery" / f"{'f' * 32}.glb").write_bytes(b"x")
    (m / "_review_v2").mkdir()
    (m / "_review_v2" / "index.json").write_text(json.dumps({"battery": {"1": U[0], "2": U[1]}, "crt_monitor": {"1": U[3]}}))
    (td / "assets" / "distractors" / "tyre").mkdir(parents=True)
    (td / "assets" / "distractors" / "tyre" / f"{U[4]}.glb").write_bytes(b"x")
    g = td / "assets" / "textures" / "ground" / "Ground001"
    g.mkdir(parents=True)
    (g / "asset.json").write_text(json.dumps({"id": "Ground001", "license": "CC0 1.0 (ambientCG)"}))
    (td / "assets" / "haven").mkdir()
    (td / "assets" / "haven" / "manifest.json").write_text(json.dumps(
        {"source": "Poly Haven (CC0)", "environments": {"desert": ["a", "b"], "european_rural": ["c"]}}))
    meta = td / "objaverse_meta"
    meta.mkdir()
    with gzip.open(meta / "000-000.json.gz", "wt", encoding="utf-8") as f:
        json.dump({U[0]: {"license": "by", "name": "car battery", "user": {"username": "maker1"}},
                   U[1]: {"license": "by"}, U[3]: {"license": "cc0"},
                   U[4]: {"license": "by-sa"}, "a" * 32: {"license": "by-nc"}}, f)


def test_stats():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        build(td)
        out = td / "rb" / "dataset_stats.json"
        DS.main(["--synthetic_dir", str(td / "syn"), "--tiled_dirs", str(td / "tiled"),
                 "--splits_manifest", str(td / "splits" / "split_manifest.json"), "--assets_dir", str(td / "assets"),
                 "--objaverse_metadata_dir", str(td / "objaverse_meta"), "--out", str(out)])
        s = json.loads(out.read_text(encoding="utf-8"))
        r = s["renders"]["main"]
        assert r["n_scenes"] == 90 and r["n_frames"] == 30 * 3 + 60, (r["n_scenes"], r["n_frames"])
        assert r["scenes_by"]["environment"] == {"desert": 30, "european_urban": 30, "european_rural": 30}
        assert r["frames_by"]["environment"]["desert"] == 90
        assert r["scenes_by"]["altitude_profile"] == {p: 30 for p in TSS.PROFILES}
        assert r["scenes_by"]["capture_mode"] == {"video_4k": 90}
        assert r["instances_total"] == 150 and sum(r["instances_per_class"].values()) == 150
        assert sum(r["instances_per_class_by_environment"]["desert"].values()) == 90
        assert r["objects_by_size_bucket_px"] == {"64-128": 150}           # max(0.2*320, 0.3*180) = 64 px
        assert r["negative_scenes"] == 1 and abs(r["negative_scene_share"] - 1 / 90) < 1e-12
        assert r["visible_but_unlabelled_per_class"]["cpu_chip"] == 7
        assert s["tiles"][str(td / "tiled")] == {"synthetic": {"tiles": 8, "empty_tiles": 3},
                                                 "val": {"tiles": 2, "empty_tiles": 0}}
        assert s["splits"]["main/heldout"]["n_scenes"] == 9 and len(s["splits_manifest_sha256"]) == 64
        a = s["assets"]["per_class"]
        assert a["battery"] == {"total": 4, "objaverse_reviewed": 2, "objaverse_UNREVIEWED": 1, "procedural": 1,
                                "other": 0, "rejected": 1}
        assert a["crt_monitor"]["objaverse_reviewed"] == 1 and s["assets"]["distractors_total"] == 1
        lic = s["licences"]
        assert lic["ground_textures"]["by_licence"] == {"CC0 1.0 (ambientCG)": 1}
        assert lic["hdris"]["n"] == 3 and lic["hdris"]["source"] == "Poly Haven (CC0)"
        ob = lic["objaverse_models_and_distractors"]
        assert ob["n_models"] == 5                                    # 3 battery + 1 crt + 1 distractor
        assert ob["by_licence"] == {"by": 2, "cc0": 1, "by-sa": 1, "unknown (not in local metadata cache)": 1}, ob
        assert ob["missing_uids"] == [U[2]] and ob["non_commercial_uids"] == []
        import csv
        rows = list(csv.DictReader(open(ob["attributions_csv"], encoding="utf-8")))
        assert {r["uid"] for r in rows} == {U[0], U[1], U[3], U[4]} and all(r["license"] for r in rows)
        # snapshot fallback: a uid missing from the local cache is filled from the repo's CSV snapshot
        snap = td / "snap.csv"
        snap.write_text("uid,license,name,creator,creator_url,url\n"
                        f"{U[2]},by-nc,old phone,someone,,https://example/x\n", encoding="utf-8")
        DS.main(["--synthetic_dir", str(td / "syn"), "--tiled_dirs", "", "--assets_dir", str(td / "assets"),
                 "--objaverse_metadata_dir", str(td / "objaverse_meta"), "--licence_snapshot", str(snap),
                 "--splits_manifest", "", "--out", str(td / "rb2" / "s.json")])
        ob2 = json.loads((td / "rb2" / "s.json").read_text(encoding="utf-8"))["licences"]["objaverse_models_and_distractors"]
        assert ob2["missing_uids"] == [] and ob2["n_from_snapshot"] == 1 and ob2["n_from_local_cache"] == 4
        assert ob2["non_commercial_uids"] == [U[2]] and ob2["by_licence"]["by-nc"] == 1
    print("  ok  dataset stats: renders, tiles, splits, assets (reviewed/unreviewed/procedural/rejected), licences exact")


if __name__ == "__main__":
    test_stats()
    print("\ndataset-stats tests passed")
