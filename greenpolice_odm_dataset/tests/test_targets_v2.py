"""
Tests for schema-v2 targets (targets_v2.json, synthetic-only regime; added 2026-10-08).

  1. targets_v2.json (and v1) pass validation; its tier-1 decision matches ewads_taxonomy.TIER_1.
  2. validate_spec rejects: missing scope/why, a metric 08 cannot compute in an 08 section, duplicate ids.
  3. select_sections: v2 needs a dataset; each dataset selects only its own 08 sections; H2/H3
     sections (other evaluators) are never selected by 08; v1 'auto' still selects every section.
  4. freeze_targets.py freezes a valid v2 file and refuses an invalid one (no lock written).
  5. 08 end to end with v2: --targets_dataset synthetic_heldout scores exactly the two synthetic
     sections; without --targets_dataset the run still succeeds and the targets are skipped.

Run: python tests/test_targets_v2.py
"""
import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ewads_hier_eval as HE  # noqa: E402
import ewads_taxonomy as tx    # noqa: E402

V2 = json.loads((ROOT / "targets_v2.json").read_text(encoding="utf-8"))
V1 = json.loads((ROOT / "targets_v1.json").read_text(encoding="utf-8"))


def test_valid_files():
    assert HE.validate_spec(V2) == [], HE.validate_spec(V2)
    assert HE.validate_spec(V1) == [], HE.validate_spec(V1)
    assert HE.schema_version(V2) == 2 and HE.schema_version(V1) == 1
    assert sorted(V2["decisions"]["tier1_classes"]) == sorted(tx.TIER_1)
    assert "lighting" not in tx.TIER_1
    print("  ok  targets_v2.json and targets_v1.json valid; tier-1 decision == ewads_taxonomy.TIER_1")


def test_invalid_specs():
    bad = copy.deepcopy(V2)
    del bad["sections"]["sim_to_real"]["targets"][0]["why"]
    assert any("missing" in e and "why" in e for e in HE.validate_spec(bad))
    bad = copy.deepcopy(V2)
    bad["sections"]["sim_to_real"]["targets"][0]["metric"] = "abs_rel_error"
    assert any("not computable by 08" in e for e in HE.validate_spec(bad))
    bad = copy.deepcopy(V2)
    bad["sections"]["sim_to_real"]["targets"][1]["id"] = "R1"
    assert any("duplicate" in e for e in HE.validate_spec(bad))
    bad = copy.deepcopy(V2)
    del bad["sections"]["sim_to_real"]["dataset"]
    assert any("dataset" in e for e in HE.validate_spec(bad))
    print("  ok  invalid specs rejected (missing why, non-08 metric in an 08 section, duplicate id, no dataset)")


def test_select_sections():
    try:
        HE.select_sections(V2, "auto", None)
        raise AssertionError("v2 without dataset must raise")
    except ValueError as e:
        assert "--targets_dataset" in str(e)
    names, _ = HE.select_sections(V2, "auto", "synthetic_heldout")
    assert names == ["synthetic_heldout.evidence", "synthetic_heldout.survey"], names
    assert HE.select_sections(V2, "auto", "sim_to_real")[0] == ["sim_to_real"]
    assert HE.select_sections(V2, "auto", "sim_to_sim_loeo")[0] == ["sim_to_sim_loeo"]
    for ds in ("h2_reference_target", "h3_tamper_trials"):     # not 08's sections
        try:
            HE.select_sections(V2, "auto", ds)
            raise AssertionError(ds)
        except ValueError:
            pass
    try:
        HE.select_sections(V2, "sim_to_real", "synthetic_heldout")
        raise AssertionError("section from another dataset must raise")
    except ValueError:
        pass
    assert HE.select_sections(V2, "synthetic_heldout.survey", "synthetic_heldout")[0] == ["synthetic_heldout.survey"]
    assert HE.select_sections(V1, "auto", None)[0] == list(V1["sections"])
    print("  ok  section selection: v2 per dataset (H2/H3 never scored by 08), v1 unchanged")


def test_freeze_v2():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        t = td / "targets_v2.json"
        shutil.copy(ROOT / "targets_v2.json", t)
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "freeze_targets.py"), "--targets", str(t), "--by", "test"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert HE.load_targets(t)["status"] == "FROZEN"
        bad = copy.deepcopy(V2)
        del bad["sections"]["h2_gsd"]["targets"][0]["scope"]
        tb = td / "targets_bad.json"
        tb.write_text(json.dumps(bad), encoding="utf-8")
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "freeze_targets.py"), "--targets", str(tb), "--by", "test"],
                           capture_output=True, text=True)
        assert r.returncode == 2 and "INVALID" in r.stdout, r.stdout + r.stderr
        assert not tb.with_suffix(".lock.json").exists()
    print("  ok  freeze_targets: valid v2 frozen, invalid v2 refused (no lock)")


def _make_eval_set(td: Path):
    rng = np.random.default_rng(1)
    (td / "d" / "images").mkdir(parents=True)
    (td / "d" / "labels").mkdir()
    dets, meta = {}, []
    for i in range(24):
        W, H = 1280, 720
        cv2.imwrite(str(td / "d" / "images" / f"im{i:02d}.jpg"), np.full((H, W, 3), 90, np.uint8))
        n = int(rng.integers(1, 5))
        cls = rng.integers(0, 18, n)
        s = rng.uniform(20, 120, n)
        x, y = rng.uniform(0, W - 130, n), rng.uniform(0, H - 130, n)
        boxes = np.stack([x, y, x + s, y + s * 0.7], 1)
        (td / "d" / "labels" / f"im{i:02d}.txt").write_text("\n".join(
            f"{c} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}"
            for c, b in zip(cls, boxes)))
        dets[f"im{i:02d}"] = np.array([[*(b + rng.normal(0, 1, 4)), 0.9, c] for c, b in zip(cls, boxes)], np.float32)
        meta.append({"image_path": f"images/im{i:02d}.jpg", "altitude_m": float([6, 12, 30][i % 3])})
    np.savez_compressed(td / "dets.npz", **dets)
    (td / "meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta))


def _run08(td: Path, out: str, *extra):
    r = subprocess.run([sys.executable, str(ROOT / "08_tiled_eval.py"), "--data_dir", str(td / "d"),
                        "--cache", str(td / "dets.npz"), "--conf", "0.3", "--meta", str(td / "meta.jsonl"),
                        "--out_dir", str(td / out), "--targets", str(ROOT / "targets_v2.json"), *extra],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r, json.loads((td / out / "eval_report.json").read_text())


def test_08_with_v2():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _make_eval_set(td)
        _, rep = _run08(td, "ev_syn", "--targets_dataset", "synthetic_heldout")
        pr = rep["preregistered_targets"]
        assert pr["schema_version"] == 2
        assert set(pr["sections"]) == {"synthetic_heldout.evidence", "synthetic_heldout.survey"}, pr["sections"].keys()
        ids = {t["id"] for res in pr["sections"].values() for t in res}
        assert ids == {"E1", "E2", "E3", "E4", "S1", "S2"}, ids
        ev = {t["id"]: t for t in pr["sections"]["synthetic_heldout.evidence"]}
        assert "16 frames with altitude in [0, 15)" in ev["E2"]["scope"], ev["E2"]["scope"]  # the 6 m + 12 m frames
        assert ev["E2"]["scope_declared"].startswith("held-out synthetic"), ev["E2"]      # v2 declared scope kept
        r, rep = _run08(td, "ev_none")
        assert rep["preregistered_targets"] is None and "skipped" in r.stdout and "--targets_dataset" in r.stdout
        _, rep = _run08(td, "ev_real", "--targets_dataset", "sim_to_real")
        assert set(rep["preregistered_targets"]["sections"]) == {"sim_to_real"}
    print("  ok  08 + v2: dataset selects its sections, altitude scope applied; no dataset -> targets skipped, run OK")


if __name__ == "__main__":
    test_valid_files()
    test_invalid_specs()
    test_select_sections()
    test_freeze_v2()
    test_08_with_v2()
    print("\ntargets-v2 tests passed")
