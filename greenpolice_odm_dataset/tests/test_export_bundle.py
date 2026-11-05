"""
Tests for tools/export_results_bundle.py, Task 9 (added 2026-10-08).

A complete fake project is built in a temp folder by running the REAL tools on synthetic inputs (frozen
targets_v2, 08 reports incl. CIs, H2, H3, dataset stats, camera ablation, AdaIN decision reports, an
evidence crop), deliberately leaving one LOEO fold and the Jetson files missing.  Then:
  - SUMMARY.md: targets status FROZEN; every pre-registered target row's measured value and verdict equal
    the source report's; H2 / H3 target rows equal their JSON; LOEO each_fold rule says 'incomplete';
    missing inputs are NOT RUN and listed; AdaIN rule applied to the two source numbers;
  - figures: 3 synthetic GT overlays, the real-test prediction + assessability frame, the evidence crop,
    the H2 scatter;
  - environment.txt, git_commit.txt (+ code_manifest.sha256 whose hash it states);
  - MANIFEST.sha256 lists every file and every hash verifies.

Run: python tests/test_export_bundle.py
"""
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools", ROOT / "tests"):
    sys.path.insert(0, str(p))
import export_results_bundle as EX      # noqa: E402
import test_bootstrap_zeroshot as TB    # noqa: E402
import test_synth_splits as TSS         # noqa: E402
import make_synth_splits as MSS         # noqa: E402
import tamper_experiment as TE          # noqa: E402
import compare_evals as CE              # noqa: E402
import dataset_stats as DS              # noqa: E402
import test_gsd_validation as TG        # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run08(proj, data_dir, cache, out, *extra):
    r = subprocess.run([sys.executable, str(ROOT / "08_tiled_eval.py"), "--data_dir", str(data_dir), "--cache",
                        str(cache), "--conf", "0.5", "--bootstrap", "40", "--targets", str(proj / "targets_v2.json"),
                        "--out_dir", str(proj / out), *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return json.loads((proj / out / "eval_report.json").read_text())


def build_project(proj: Path):
    shutil.copy(ROOT / "targets_v2.json", proj / "targets_v2.json")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "freeze_targets.py"), "--targets",
                        str(proj / "targets_v2.json"), "--by", "test"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rb = proj / "results_bundle"
    rb.mkdir()
    (rb / "freeze_record.txt").write_text("frozen by test\n")
    # eval data (fake 08 inputs) -> synthetic held-out, 2 of 3 LOEO folds, real test, zero-shot, AdaIN val
    TB.write_eval_dir(proj / "evaldata", n=12)
    ed = proj / "evaldata"
    meta = ["--meta", str(ed / "meta.jsonl")]
    run08(proj, ed / "d", ed / "dets.npz", "eval/final/synthetic_heldout", *meta, "--targets_dataset", "synthetic_heldout")
    for f in ("loeo_desert", "loeo_european_urban"):
        run08(proj, ed / "d", ed / "dets.npz", f"eval/final/{f}", "--targets_dataset", "sim_to_sim_loeo")
    (proj / "eval/final/real_test_best").mkdir(parents=True)
    shutil.copy(ed / "dets.npz", proj / "eval/final/real_test_best/detections.npz")
    shutil.copytree(ed / "d", proj / "data/test")
    with open(proj / "data/test/heights.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stem", "height_m", "camera"])
        for p in (proj / "data/test/images").iterdir():
            w.writerow([p.stem, 12.0, "phantom4pro"])
    run08(proj, proj / "data/test", proj / "eval/final/real_test_best/detections.npz", "eval/final/real_test_best",
          *meta, "--targets_dataset", "sim_to_real")
    run08(proj, ed / "d", ed / "dets.npz", "eval/val_A_noadain", "--targets", "")
    with np.load(ed / "dets.npz") as z:
        fewer = {k: z[k][:1] for k in z.files}
    np.savez_compressed(ed / "fewer.npz", **fewer)
    run08(proj, ed / "d", ed / "fewer.npz", "eval/val_A_adain", "--targets", "")
    CE.main(["--data_dir", str(ed / "d"), "--a_cache", str(ed / "dets.npz"), "--a_conf", "0.5", "--a_name", "true_camera",
             "--b_cache", str(ed / "fewer.npz"), "--b_conf", "0.5", "--b_name", "mismatch", "--bootstrap", "30",
             "--out", str(rb / "ablation_camera.json")])
    # synthetic held-out frames for the GT figures
    TSS.fake_render(proj / "syn", n_per=4)
    MSS.main(["--synthetic_dir", str(proj / "syn"), "--out_dir", str(proj / "data/splits"), "--no_hash_files", "--no_loeo"])
    # H2, H3, dataset stats
    hd = proj / "h2in"
    hd.mkdir()
    TG.build(hd)
    import gsd_validation as GV
    GV.main(["--photos", str(hd / "photos"), "--heights", str(hd / "heights.csv"), "--endpoints", str(hd / "endpoints.csv"),
             "--out_dir", str(rb), "--targets", str(proj / "targets_v2.json")])
    TE.run(6, 10, 1, rb, None, None, proj / "targets_v2.json")
    DS.main(["--synthetic_dir", str(proj / "syn"), "--tiled_dirs", "", "--splits_manifest",
             str(proj / "data/splits/split_manifest.json"), "--assets_dir", str(proj / "noassets"),
             "--objaverse_metadata_dir", str(proj / "nometa"), "--licence_snapshot", "", "--out", str(rb / "dataset_stats.json")])
    crop = proj / "abl/real_evidence/run_x/review_crops"
    crop.mkdir(parents=True)
    import cv2
    cv2.imwrite(str(crop / "obj00001_battery.png"), np.full((40, 60, 3), 120, np.uint8))


def test_export():
    with tempfile.TemporaryDirectory() as td:
        proj = Path(td)
        build_project(proj)
        EX.main(["--root", str(proj), "--bundle", "results_bundle"])
        rb = proj / "results_bundle"
        s = (rb / "SUMMARY.md").read_text(encoding="utf-8")
        assert "status **FROZEN**" in s and "not frozen" not in s.lower().split("pre-registration")[1].split("## H1")[0]
        # every target row equals its source
        n_rows = 0
        for name, rel in (("synthetic_heldout_single_model", "eval/final/synthetic_heldout"),
                          ("loeo_desert", "eval/final/loeo_desert"), ("real_test_best_model", "eval/final/real_test_best")):
            rep = json.loads((proj / rel / "eval_report.json").read_text())
            for sec, res in rep["preregistered_targets"]["sections"].items():
                for t in res:
                    line = next(l for l in s.splitlines() if l.startswith(f"| {name} | {sec} | {t['id']} |"))
                    cells = [c.strip() for c in line.strip("|").split("|")]
                    assert cells[6] == EX.fmt(t["value_measured"]) and cells[8] == EX.verdict(t["pass"]), (line, t)
                    if t.get("ci95") and t["ci95"].get("lo") is not None:
                        assert cells[7] == EX.ci_txt(t["ci95"]), (line, t["ci95"])
                    n_rows += 1
        assert n_rows >= 6
        assert "| loeo_european_rural | NOT RUN |" in s
        assert all(l.rstrip().endswith("| incomplete |") for l in s.splitlines() if l.startswith("| L") and "/3 |" in l)
        for t in json.loads((rb / "h3_tamper.json").read_text())["targets"]["checks"]:
            assert f"| {t['id']} | {t['tamper']} | {t['check']} | {t['op']} {t['target']} | {EX.fmt(t['measured'])} | {t['n']} | {EX.verdict(t['pass'])} |" in s
        h2c = json.loads((rb / "h2_gsd.json").read_text())["targets"]["checks"][0]
        assert f"| H2 | abs_rel_error | p95 | <= 0.1 | {EX.fmt(h2c['measured'])} |" in s
        a0 = json.loads((proj / "eval/val_A_noadain/eval_report.json").read_text())["map50"]
        a1 = json.loads((proj / "eval/val_A_adain/eval_report.json").read_text())["map50"]
        assert f"no-AdaIN {EX.fmt(a0)}" in s and f"AdaIN {EX.fmt(a1)}" in s
        assert ("→ **AdaIN**" in s) == (a1 > a0)
        assert "INT8 vs FP16: **NOT RUN**" in s and "Jetson benchmark: **NOT RUN**" in s
        miss = s.split("## Missing inputs")[1]
        for k in ("engine_decision", "jetson_benchmark", "loeo_european_rural"):
            assert k in miss, k
        # figures
        figs = {p.name for p in (rb / "figures").iterdir()}
        assert {"synthetic_desert_ground_truth.jpg", "synthetic_european_urban_ground_truth.jpg",
                "synthetic_european_rural_ground_truth.jpg", "real_test_predictions_assessability.jpg",
                "evidence_review_crop.png", "h2_gsd_scatter.png"} <= figs, figs
        # environment / code identity
        env = (rb / "environment.txt").read_text()
        assert "python:" in env and "ultralytics:" in env
        gc = (rb / "git_commit.txt").read_text()
        assert "not a git repository" in gc and sha(rb / "code_manifest.sha256") in gc
        # manifest: complete and verifiable
        lines = (rb / "MANIFEST.sha256").read_text().splitlines()
        listed = {l.split("  ", 1)[1] for l in lines}
        actual = {p.relative_to(rb).as_posix() for p in rb.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"}
        assert listed == actual
        for l in lines:
            h, rel = l.split("  ", 1)
            assert sha(rb / rel) == h, rel
    print("  ok  export bundle: SUMMARY numbers/verdicts = sources, NOT RUN + missing listed, LOEO incomplete, "
          "figures, environment, code manifest, MANIFEST verifies")


if __name__ == "__main__":
    test_export()
    print("\nexport-bundle tests passed")
