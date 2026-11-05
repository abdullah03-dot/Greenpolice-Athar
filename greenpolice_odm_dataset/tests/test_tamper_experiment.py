"""
Tests for the H3 tamper experiment, Task 4 (added 2026-10-08).

  1. synthetic run (small N): the expected detection pattern per check level -- chain alone catches
     edit / reorder / insert / middle deletion and misses truncation / rebuild / source swap; the signed
     head adds truncation and rebuild; media re-hash adds source swap. Outputs JSON + CSV + per-trial log,
     with counts, the targets' freeze status, and PASS/FAIL per H3 target.
  2. deterministic: same seed -> identical results table.
  3. real-log mode: a COPY of the log is tampered; the original log and its media are byte-identical
     afterwards; without --media_root, source_swap is reported not-applicable (never touches media).

Run: python tests/test_tamper_experiment.py
"""
import csv
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tamper_experiment as TE   # noqa: E402

EXPECT = {  # tamper: (chain, chain+head, chain+head+rehash)
    "field_edit": (1, 1, 1), "reorder": (1, 1, 1), "insert": (1, 1, 1), "middle_deletion": (1, 1, 1),
    "tail_truncation": (0, 1, 1), "full_rebuild": (0, 1, 1), "source_swap": (0, 0, 1),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_synthetic():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        r = TE.run(12, 15, 7, td / "out", None, None, ROOT / "targets_v2.json")
        for kind, exp in EXPECT.items():
            for lv, e in zip(TE.LEVELS, exp):
                c = r["results"][kind][lv]
                assert c["n"] == 12 and c["rate"] == float(e), (kind, lv, c)
        assert r["targets"]["freeze_status"] in ("NOT FROZEN", "FROZEN", "MODIFIED AFTER FREEZE")
        assert all(t["pass"] is True for t in r["targets"]["checks"]), r["targets"]["checks"]
        rows = list(csv.DictReader(open(td / "out" / "h3_tamper.csv", encoding="utf-8")))
        assert len(rows) == len(TE.TAMPERS) * len(TE.LEVELS) and all(int(x["n"]) == 12 for x in rows)
        trials = (td / "out" / "h3_tamper_trials.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(trials) == 12 * len(TE.TAMPERS)
        r2 = TE.run(12, 15, 7, td / "out2", None, None, None)
        assert r2["results"] == r["results"], "same seed must give the same table"
    print("  ok  synthetic: expected pattern per level, outputs + targets (with freeze status), deterministic")


def test_real_log_copy():
    sys.path.insert(0, str(ROOT))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        media = td / "media"
        media.mkdir()
        lg = TE.b07().ForensicAuditLogger(td / "real.jsonl", "real-model", model_sha256="aa" * 32)
        rng = random.Random(1)
        for i in range(10):
            f = media / f"DJI_{i:04d}.JPG"
            f.write_bytes(bytes(rng.getrandbits(8) for _ in range(300)))
            lg.log_detection(str(f), sha(f), [], None, 20.0, 0.3)
        before = {p.name: sha(p) for p in [td / "real.jsonl", *media.iterdir()]}
        r = TE.run(8, 0, 3, td / "out", td / "real.jsonl", media, ROOT / "targets_v2.json")
        after = {p.name: sha(p) for p in [td / "real.jsonl", *media.iterdir()]}
        assert before == after, "the real log and its media must never be modified"
        assert r["source"]["kind"] == "copy of real log" and r["source"]["sha256"] == before["real.jsonl"]
        assert r["results"]["source_swap"]["chain+head+rehash"]["rate"] == 1.0
        assert r["results"]["tail_truncation"]["chain"]["rate"] == 0.0
        r = TE.run(4, 0, 3, td / "out_nomedia", td / "real.jsonl", None, None)
        assert r["not_applicable"].get("source_swap") == 4 and r["results"]["source_swap"]["chain"]["n"] == 0
        assert {p.name: sha(p) for p in [td / "real.jsonl", *media.iterdir()]} == before
    print("  ok  real-log mode: only copies tampered (originals byte-identical); no media_root -> source_swap n/a")


if __name__ == "__main__":
    test_synthetic()
    test_real_log_copy()
    print("\ntamper-experiment tests passed")
