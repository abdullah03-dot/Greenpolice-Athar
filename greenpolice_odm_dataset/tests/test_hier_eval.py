"""
Tests for the evaluation extensions (ewads_taxonomy, ewads_hier_eval, 08's new sections,
tools/freeze_targets.py).  No model needed.

  1. One class list everywhere: taxonomy == 07 == ewads_survey; tier-1 identical to 07.
  2. Hierarchy: a CRT called 'flat_screen' is wrong at fine level, right at group and binary;
     two boxes of different fine classes on ONE display count once at group level; a chip lying on a
     board stays two objects.
  3. Sites: clustering by metres (via GSD), site found by any touching detection, nearby false
     detections = one false alert.
  4. Targets: draft -> freeze -> FROZEN -> edit -> MODIFIED AFTER FREEZE; sections scored only on
     frames inside their altitude scope.
  5. 08 end to end from a detections cache (+ --meta): every new report block present and sane.
Run:  python tests/test_hier_eval.py
"""
import ast
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
import ewads_hier_eval as HE   # noqa: E402
import ewads_survey as es      # noqa: E402
import ewads_taxonomy as tx    # noqa: E402

C = {n: i for i, n in enumerate(tx.CLASSES)}


def test_class_lists():
    tree = ast.parse((ROOT / "07_edge_benchmark.py").read_text(encoding="utf-8"))
    got = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("EWASTE_CLASSES", "TIER_1_CLASSES"):
            got[node.targets[0].id] = node.value
    assert ast.literal_eval(got["EWASTE_CLASSES"]) == tx.CLASSES == es.EWASTE_CLASSES
    # 07 must take tier-1 from the taxonomy (single source), not keep its own copy
    assert ast.unparse(got["TIER_1_CLASSES"]) == "set(tx.TIER_1)", ast.unparse(got["TIER_1_CLASSES"])
    print("  ok  one class list (taxonomy == 07 == ewads_survey), 07 tier-1 is taxonomy.TIER_1")


def _g(*objs):
    return (np.array([C[c] for c, _ in objs], int), np.array([b for _, b in objs], float).reshape(-1, 4))


def test_hierarchy():
    disp = [100, 100, 160, 150]
    gts = {"a": _g(("flat_screen", disp))}
    dets = {"a": np.array([[*disp, 0.9, C["crt_monitor"]]], float)}
    fine = HE.metric(gts, dets, metric="map50", level="fine")[0]
    grp = HE.metric(gts, dets, metric="map50", level="group")[0]
    bnr = HE.metric(gts, dets, metric="map50", level="binary")[0]
    assert fine == 0.0 and abs(grp - 1) < 1e-9 and abs(bnr - 1) < 1e-9, (fine, grp, bnr)
    # two fine labels on one display -> one display at group level, no FP
    dets2 = {"a": np.array([[*disp, 0.9, C["crt_monitor"]], [101, 100, 160, 151, 0.8, C["flat_screen"]]], float)}
    v, d = HE.metric(gts, dets2, metric="fp_share", level="group", conf=0.5)
    assert v == 0.0 and d["tp"] == 1, d
    v, d = HE.metric(gts, dets2, metric="fp_share", level="fine", conf=0.5)
    assert d["fp"] == 1, d                                    # at fine level the crt box IS a false positive
    # nested: chip on a board stays two board_chip objects
    gts3 = {"b": _g(("circuit_board", [0, 0, 200, 120]), ("cpu_chip", [80, 40, 110, 70]))}
    dets3 = {"b": np.array([[0, 0, 200, 120, 0.9, C["circuit_board"]], [80, 40, 110, 70, 0.8, C["cpu_chip"]]], float)}
    v, d = HE.metric(gts3, dets3, metric="recall", level="group", conf=0.5)
    assert v == 1.0 and d["tp"] == 2, d
    # tier-1 subset
    gts4 = {"c": _g(("battery", [0, 0, 30, 30]), ("keyboard", [100, 100, 160, 120]))}
    dets4 = {"c": np.array([[100, 100, 160, 120, 0.9, C["keyboard"]]], float)}
    assert HE.metric(gts4, dets4, metric="recall", classes="tier1", conf=0.5)[0] == 0.0
    assert HE.metric(gts4, dets4, metric="recall", conf=0.5)[0] == 0.5
    print("  ok  hierarchy: confusion forgiven at group/binary, cross-class duplicates merged, nested kept, tier-1 subset")


def test_sites():
    # gsd 1 cm/px, link 3 m = 300 px.  Site A: 3 objects within 1 m; site B: one object 10 m away.
    gts = {"f": _g(("battery", [100, 100, 130, 130]), ("keyboard", [200, 120, 260, 140]),
                   ("crt_monitor", [150, 200, 200, 250]), ("printer", [1300, 900, 1360, 950]))}
    dets = {"f": np.array([[205, 118, 258, 142, 0.7, C["plastic_casing"]],          # touches site A (wrong class: still found)
                           [700, 700, 720, 720, 0.6, 3], [730, 700, 750, 720, 0.6, 3],   # two FPs 10 px apart
                           [1700, 100, 1720, 120, 0.6, 3]], float)}                 # FP far away
    s = HE.site_metrics(gts, dets, 0.5, gsd={"f": 0.01}, link_m=3.0)
    assert s["n_sites"] == 2 and s["sites_found"] == 1 and s["false_alerts"] == 2, s
    s2 = HE.site_metrics(gts, dets, 0.5, gsd={"f": 0.10}, link_m=3.0)     # 10 cm/px: 30 px link
    assert s2["n_sites"] == 4, s2                                          # objects > 30 px apart split
    print("  ok  sites: metric linking via GSD, any-class hit, clustered false alerts")


def test_targets_and_freeze():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        t = td / "targets_v1.json"
        shutil.copy(ROOT / "targets_v1.json", t)
        assert HE.load_targets(t)["status"] == "NOT FROZEN"
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "freeze_targets.py"), "--targets", str(t), "--by", "test"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert HE.load_targets(t)["status"] == "FROZEN"
        spec = json.loads(t.read_text())
        spec["sections"]["evidence"]["targets"][0]["value"] = 0.50          # moving the goalposts
        t.write_text(json.dumps(spec, indent=2))
        assert HE.load_targets(t)["status"] == "MODIFIED AFTER FREEZE"
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "freeze_targets.py"), "--targets", str(t)],
                           capture_output=True, text=True)
        assert r.returncode == 1 and "MODIFIED" in r.stdout
        # altitude scope: evidence targets only see the low frame
        gts = {"low": _g(("battery", [0, 0, 40, 40])), "high": _g(("battery", [0, 0, 40, 40]))}
        dets = {"low": np.array([[0, 0, 40, 40, 0.9, C["battery"]]], float), "high": np.zeros((0, 6))}
        sec = json.loads((ROOT / "targets_v1.json").read_text())["sections"]["evidence"]
        res = HE.check_targets(sec, gts, dets, 0.5, alt={"low": 8.0, "high": 40.0})
        e3 = next(r_ for r_ in res if r_["id"] == "E3")
        assert e3["value_measured"] == 1.0 and e3["pass"] and "1 frames" in e3["scope"], e3
        res2 = HE.check_targets(sec, gts, dets, 0.5, alt=None)
        assert "NOT enforced" in res2[0]["scope"] and all(r_["pass"] is None for r_ in res2)   # no verdict
        res3 = HE.check_targets(sec, gts, dets, 0.5, alt=None, scope_asserted=True)
        assert "ASSERTED" in res3[0]["scope"] and any(r_["pass"] is not None for r_ in res3)
    print("  ok  targets: draft / frozen / modified-after-freeze detected; altitude scope enforced")


def test_08_end_to_end():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "d" / "images").mkdir(parents=True); (td / "d" / "labels").mkdir()
        dets, meta = {}, []
        for i in range(30):
            W, H = 1280, 720
            cv2.imwrite(str(td / "d" / "images" / f"im{i:02d}.jpg"), np.full((H, W, 3), 90, np.uint8))
            n = int(rng.integers(0, 6))
            cls = rng.integers(0, 18, n)
            s = rng.uniform(10, 120, n)
            x, y = rng.uniform(0, W - 130, n), rng.uniform(0, H - 130, n)
            boxes = np.stack([x, y, x + s, y + s * 0.7], 1)
            lines = [f"{c} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}"
                     for c, b in zip(cls, boxes)]
            (td / "d" / "labels" / f"im{i:02d}.txt").write_text("\n".join(lines))
            d = []
            for c, b in zip(cls, boxes):
                if rng.random() < 0.85:
                    cc = c if rng.random() < 0.8 else tx.GROUPS[tx.FINE_TO_GROUP[tx.CLASSES[c]]][0]
                    d.append([*(b + rng.normal(0, 1.5, 4)), rng.uniform(0.4, 1), C[cc] if isinstance(cc, str) else cc])
            d.append([10, 10, 30, 30, rng.uniform(0.05, 0.7), 3])
            dets[f"im{i:02d}"] = np.array(d, np.float32)
            meta.append({"image_path": f"images/im{i:02d}.jpg", "altitude_m": float([6, 12, 30][i % 3])})
        np.savez_compressed(td / "dets.npz", **dets)
        (td / "meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta))
        r = subprocess.run([sys.executable, str(ROOT / "08_tiled_eval.py"), "--data_dir", str(td / "d"),
                            "--cache", str(td / "dets.npz"), "--conf", "0.3", "--meta", str(td / "meta.jsonl"),
                            "--out_dir", str(td / "ev"), "--targets", str(ROOT / "targets_v1.json")],
                           capture_output=True, text=True)   # v1 explicitly: default is targets_v2.json since 2026-10-08
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        rep = json.loads((td / "ev" / "eval_report.json").read_text())
        h = rep["hierarchy"]
        assert h["group"]["map50"] >= h["fine"]["map50"] - 1e-9 and h["binary"]["map50"] >= h["group"]["map50"] - 0.02
        assert rep["size_x_altitude"] and rep["sites"]["n_sites"] > 0 and rep["tier1"]["recall"] >= 0
        pr = rep["preregistered_targets"]
        assert pr["status"] in ("NOT FROZEN", "FROZEN", "MODIFIED AFTER FREEZE") and set(pr["sections"]) == {"evidence", "survey"}
        assert all(k in rep for k in ("map5095", "map50", "conf", "operating_point", "per_class_ap"))   # 10 reads these
        for line in r.stdout.splitlines():
            if any(k in line for k in ("HIERARCHY", "fine ", "group ", "binary ", "TIER-1", "SITES", "TARGETS", " E1", " S1")):
                print("      " + line.strip())
    print("  ok  08 end to end: hierarchy, tier-1, altitude x size, sites, pre-registered targets; 10's keys intact")


if __name__ == "__main__":
    test_class_lists()
    test_hierarchy()
    test_sites()
    test_targets_and_freeze()
    test_08_end_to_end()
    print("\nevaluation-extension tests passed")
