#!/usr/bin/env python3
"""
H3 tamper experiment (ATHAR, added 2026-10-08).

Builds a hash-chained audit log with 07's own ForensicAuditLogger from a deterministic set of synthetic
records whose media files exist on disk (or uses a COPY of a real log with --log), signs its head with
an EPHEMERAL Ed25519 key (stands in for the operator's key; the simulated attacker never holds it), then
for N trials per tamper type applies one tampering to a fresh copy and checks it at three levels:

  chain               verify_chain() alone                                  (07 ForensicAuditLogger)
  chain+head          + signed chain-head verification against the trusted   (tools/sign_chain_head.py)
                        key fingerprint
  chain+head+rehash   + re-hashing every referenced media file against the record's image_sha256

Tamper types (threat model: attacker can edit the log and the media files, does NOT hold the key):
  field_edit       change one field of one record, record_hash left as is
  reorder          swap two records
  insert           insert a forged record mid-chain; its own hash is computed correctly and its
                   prev_hash points at its predecessor (the attacker does what is cheap)
  middle_deletion  delete one record that is not the last
  tail_truncation  drop the last k records
  full_rebuild     edit one record and recompute every hash and prev_hash from there to the end
  source_swap      replace the media file under one record; the log is untouched

Output: results_bundle/h3_tamper.json and .csv (detection rate per type x check level, with counts),
each H3 target of targets_v2.json marked PASS/FAIL, per-trial records in h3_tamper_trials.jsonl.
Every number is computed here; nothing is estimated.

Usage
  python tools/tamper_experiment.py                                   # synthetic chain, N=200
  python tools/tamper_experiment.py --log path/to/forensic_audit.jsonl --media_root <dir>   # real log copy
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import sign_chain_head as SCH  # noqa: E402

TAMPERS = ["field_edit", "reorder", "insert", "middle_deletion", "tail_truncation", "full_rebuild", "source_swap"]
LEVELS = ["chain", "chain+head", "chain+head+rehash"]
EDITABLE = ["altitude_m", "conf_threshold", "n_detections", "image_sha256", "timestamp_utc",
            "capture_time_utc", "model_sha256", "gps", "detections"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _record_hash(rec: dict) -> str:
    """Identical to 07's ForensicAuditLogger: SHA-256 of json.dumps(payload, sort_keys=True, default=str)."""
    body = {k: v for k, v in rec.items() if k != "record_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def b07():
    return SCH._bench07()


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


# ───────────────────────── baseline ─────────────────────────
def build_synthetic(work: Path, n_records: int, seed: int) -> Path:
    """Deterministic records whose media files exist: <work>/media/frame_XXXX.bin (random bytes)."""
    rng = random.Random(seed)
    media = work / "media"
    media.mkdir(parents=True)
    log = work / "audit.jsonl"
    with quiet():
        lg = b07().ForensicAuditLogger(log, "tamper-experiment|tile768", model_sha256="5c" * 32)
    for i in range(n_records):
        f = media / f"frame_{i:04d}.bin"
        f.write_bytes(bytes(rng.getrandbits(8) for _ in range(512)))
        dets = [{"class_id": c, "class_name": ["crt_monitor", "battery", "circuit_board"][c % 3],
                 "confidence": round(rng.uniform(0.3, 0.99), 4),
                 "bbox_xyxy": [rng.randint(0, 900), rng.randint(0, 500), rng.randint(900, 1900), rng.randint(500, 1000)],
                 "tier": 1} for c in range(rng.randint(0, 4))]
        with quiet():
            lg.log_detection(str(f), sha256_file(f), dets,
                             {"latitude": 25 + rng.random() / 100, "longitude": 55 + rng.random() / 100},
                             round(rng.uniform(8, 60), 2), 0.31, {"frame": i},
                             capture_time={"capture_time_utc": f"2026-10-01T06:{i // 60:02d}:{i % 60:02d}+00:00",
                                           "capture_time_local": None, "capture_time_source": "synthetic"})
    return log


def read(log: Path) -> list:
    return [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]


def write(log: Path, recs: list):
    log.write_text("".join(json.dumps(r, default=str) + "\n" for r in recs), encoding="utf-8")


def media_path(rec: dict, media_root: Path | None, overlay: Path | None = None) -> Path | None:
    """Where a record's media is read from: the trial's OVERLAY (tampered copies) first, then
    media_root, then the path stored in the record.  Originals are only ever READ."""
    p = str(rec.get("image_path", ""))
    if "#frame=" in p:
        p = p.split("#frame=")[0]
    name = Path(p.replace("\\", "/")).name
    for q in ((overlay / name) if overlay else None, (media_root / name) if media_root else None, Path(p)):
        if q is not None and q.exists():
            return q
    return None


_HASH_CACHE: dict = {}


def _sha_cached(p: Path, overlay: Path | None) -> str:
    if overlay is not None and overlay in p.parents:      # tampered copy: never cached
        return sha256_file(p)
    k = str(p.resolve())
    if k not in _HASH_CACHE:
        _HASH_CACHE[k] = sha256_file(p)
    return _HASH_CACHE[k]


def rehash_ok(recs: list, media_root: Path | None, overlay: Path | None = None) -> tuple:
    """-> (all_ok, n_checked). Video records (#frame=) compare the video file to context.video_sha256."""
    n = 0
    for r in recs:
        mp = media_path(r, media_root, overlay)
        if mp is None:
            continue
        expect = r.get("image_sha256")
        if "#frame=" in str(r.get("image_path", "")):
            expect = (r.get("context") or {}).get("video_sha256", expect)
        n += 1
        if _sha_cached(mp, overlay) != expect:
            return False, n
    return True, n


# ───────────────────────── tampering ─────────────────────────
def _mutate(v, rng):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + (1 if isinstance(v, int) else round(rng.uniform(0.5, 5.0), 3))
    if isinstance(v, str):
        return v[:-1] + ("0" if v[-1:] != "0" else "1") if v else "x"
    if isinstance(v, dict):
        v = copy.deepcopy(v)
        if v:
            k = sorted(v)[0]
            v[k] = _mutate(v[k], rng)
        else:
            v["x"] = 1
        return v
    if isinstance(v, list):
        v = copy.deepcopy(v)
        if v:
            v[0] = _mutate(v[0], rng)
        else:
            v.append({"class_name": "battery", "confidence": 0.9})
        return v
    return "tampered"


def tamper(kind: str, recs: list, rng: random.Random, overlay: Path, media_root: Path | None) -> dict:
    """Apply one tampering in place (recs and/or media files). -> description of what was done."""
    n = len(recs)
    if kind == "field_edit":
        i = rng.randrange(n)
        f = rng.choice([k for k in EDITABLE if k in recs[i]] or ["record_index"])
        recs[i][f] = _mutate(recs[i][f], rng)
        return {"record": i, "field": f}
    if kind == "reorder":
        i, j = sorted(rng.sample(range(n), 2))
        recs[i], recs[j] = recs[j], recs[i]
        return {"swapped": [i, j]}
    if kind == "insert":
        k = rng.randrange(1, n)                       # between record k-1 and k
        forged = copy.deepcopy(recs[k - 1])
        forged["detections"] = []
        forged["n_detections"] = 0
        forged["prev_hash"] = recs[k - 1]["record_hash"]
        forged["record_hash"] = _record_hash(forged)
        recs.insert(k, forged)
        return {"inserted_at": k}
    if kind == "middle_deletion":
        k = rng.randrange(0, n - 1)
        del recs[k]
        return {"deleted": k}
    if kind == "tail_truncation":
        k = rng.randint(1, n - 1)
        del recs[n - k:]
        return {"dropped_last": k}
    if kind == "full_rebuild":
        i = rng.randrange(n)
        f = rng.choice([k for k in EDITABLE if k in recs[i]] or ["record_index"])
        recs[i][f] = _mutate(recs[i][f], rng)
        prev = recs[i - 1]["record_hash"] if i else "GENESIS"
        for r in recs[i:]:
            r["prev_hash"] = prev
            r["record_hash"] = _record_hash(r)
            prev = r["record_hash"]
        return {"record": i, "field": f, "rehashed_from": i}
    if kind == "source_swap":
        if media_root is None:          # never touch media in place: only an explicit media_root is used
            return {"not_applicable": "no --media_root: source_swap needs the referenced media"}
        cand = [i for i, r in enumerate(recs) if media_path(r, media_root) is not None]
        if not cand:
            return {"not_applicable": "no referenced media file exists under media_root"}
        i = rng.choice(cand)
        src = media_path(recs[i], media_root)             # original: READ only
        data = bytearray(src.read_bytes())
        if data:
            pos = rng.randrange(len(data))
            data[pos] ^= 1 + rng.randrange(255)           # change one byte (smallest possible swap)
        else:
            data = bytearray(b"x")
        overlay.mkdir(parents=True, exist_ok=True)
        (overlay / src.name).write_bytes(bytes(data))     # the swapped file the checker will now find
        return {"record": i, "file": src.name}
    raise ValueError(kind)


# ───────────────────────── checks ─────────────────────────
def check(log: Path, fpr: str, media_root: Path | None, overlay: Path | None = None) -> dict:
    with quiet():
        chain = SCH._verify_chain(log)
        head = SCH.verify_head(log, trusted_fingerprint=fpr)
    recs = read(log) if chain.get("valid") else _read_lenient(log)
    rh_ok, n_rh = rehash_ok(recs, media_root, overlay)
    chain_ok, head_ok = bool(chain.get("valid")), bool(head.get("valid"))
    return {"detected": {"chain": not chain_ok,
                         "chain+head": not (chain_ok and head_ok),
                         "chain+head+rehash": not (chain_ok and head_ok and rh_ok)},
            "chain_reason": chain.get("reason"), "head_reasons": head.get("reasons"), "rehash_ok": rh_ok,
            "media_checked": n_rh}


def _read_lenient(log: Path) -> list:
    out = []
    for l in log.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return out


# ───────────────────────── experiment ─────────────────────────
def run(n_trials: int, n_records: int, seed: int, out_dir: Path, log: Path | None, media_root: Path | None,
        targets: Path | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="athar_h3_") as tmp:
        tmp = Path(tmp)
        base = tmp / "base"
        base.mkdir()
        if log:
            shutil.copy2(log, base / "audit.jsonl")
            base_log = base / "audit.jsonl"
            source = {"kind": "copy of real log", "path": str(log), "sha256": sha256_file(log)}
            with quiet():
                v = SCH._verify_chain(base_log)
            if not v.get("valid"):
                raise SystemExit(f"[ERROR] the supplied log is already broken: {v}")
        else:
            base_log = build_synthetic(base, n_records, seed)
            media_root = base / "media"
            source = {"kind": "synthetic records (07 ForensicAuditLogger)", "n_records": n_records, "seed": seed}
        recs0 = read(base_log)
        if len(recs0) < 3:
            raise SystemExit("[ERROR] need at least 3 records")
        key = SCH.keygen(tmp / "ephemeral_key" / "k.pem")       # temp dir: outside the repository
        with quiet():
            SCH.sign_head(base_log, Path(key["private_key"]))
        head0 = SCH.head_path_for(base_log).read_text(encoding="utf-8")
        with quiet():
            sanity = check(base_log, key["fingerprint"], media_root)
        if any(sanity["detected"].values()):
            raise SystemExit(f"[ERROR] untampered baseline flagged as tampered: {sanity}")
        # the untampered copy written the same way as a trial must also pass (no false alarms from I/O)
        ctl = tmp / "control"
        ctl.mkdir()
        write(ctl / "audit.jsonl", copy.deepcopy(recs0))
        SCH.head_path_for(ctl / "audit.jsonl").write_text(head0, encoding="utf-8")
        control = check(ctl / "audit.jsonl", key["fingerprint"], media_root, ctl / "overlay")
        if any(control["detected"].values()):
            raise SystemExit(f"[ERROR] untampered control copy flagged as tampered (serialisation?): {control}")

        counts = defaultdict(lambda: defaultdict(int))
        n_done = defaultdict(int)
        na = defaultdict(int)
        trials_f = open(out_dir / "h3_tamper_trials.jsonl", "w", encoding="utf-8")
        for kind in TAMPERS:
            for t in range(n_trials):
                rng = random.Random(f"{seed}|{kind}|{t}")
                td = tmp / "trial"
                if td.exists():
                    shutil.rmtree(td)
                td.mkdir()
                tlog = td / "audit.jsonl"
                SCH.head_path_for(tlog).write_text(head0, encoding="utf-8")   # the head signed on the baseline
                overlay = td / "overlay"                    # tampered media copies; originals untouched
                recs = copy.deepcopy(recs0)
                what = tamper(kind, recs, rng, overlay, media_root)
                if what.get("not_applicable"):
                    na[kind] += 1
                    trials_f.write(json.dumps({"tamper": kind, "trial": t, **what}) + "\n")
                    continue
                write(tlog, recs)                          # source_swap: recs unchanged (log untouched)
                res = check(tlog, key["fingerprint"], media_root, overlay)
                n_done[kind] += 1
                for lv in LEVELS:
                    counts[kind][lv] += int(res["detected"][lv])
                trials_f.write(json.dumps({"tamper": kind, "trial": t, "what": what, **res}, default=str) + "\n")
        trials_f.close()

    rows, table = [], {}
    for kind in TAMPERS:
        table[kind] = {}
        for lv in LEVELS:
            n = n_done[kind]
            d = counts[kind][lv]
            table[kind][lv] = {"detected": d, "n": n, "rate": (d / n) if n else None}
            rows.append({"tamper": kind, "check": lv, "detected": d, "n": n,
                         "rate": "" if not n else f"{d / n:.4f}", "not_applicable": na[kind]})
    result = {
        "experiment": "H3 tamper detection", "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_trials_per_type": n_trials, "seed": seed, "source": source,
        "threat_model": "attacker edits log and media files; does NOT hold the signing key",
        "signing_key": {"kind": "ephemeral Ed25519 generated for this run (temp dir, deleted)",
                        "fingerprint": key["fingerprint"]},
        "levels": LEVELS, "results": table, "not_applicable": dict(na),
        "code_sha256": {f: sha256_file(ROOT / f) for f in
                        ("tools/tamper_experiment.py", "tools/sign_chain_head.py", "07_edge_benchmark.py")},
    }
    if targets and targets.exists():
        sys.path.insert(0, str(ROOT))
        import ewads_hier_eval as HE
        tl = HE.load_targets(targets)               # FROZEN / NOT FROZEN / MODIFIED AFTER FREEZE
        spec = tl["spec"]
        sec = spec.get("sections", {}).get("h3_tamper")
        if sec:
            res_t = []
            for t in sec["targets"]:
                cell = table.get(t["tamper"], {}).get(t["check"], {})
                v = cell.get("rate")
                ok = None if v is None else (v >= t["value"] if t["op"] == ">=" else v > t["value"] if t["op"] == ">"
                                             else v <= t["value"] if t["op"] == "<=" else v < t["value"])
                res_t.append({"id": t["id"], "tamper": t["tamper"], "check": t["check"], "op": t["op"],
                              "target": t["value"], "measured": v, "n": cell.get("n"), "pass": ok})
            result["targets"] = {"file": targets.name, "sha256": sha256_file(targets),
                                 "freeze_status": tl["status"], "lock": tl["lock"], "checks": res_t}
    (out_dir / "h3_tamper.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    with open(out_dir / "h3_tamper.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tamper", "check", "detected", "n", "rate", "not_applicable"])
        w.writeheader()
        w.writerows(rows)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--n_records", type=int, default=50, help="synthetic chain length")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--log", default=None, help="use a COPY of this real audit log instead of synthetic records")
    ap.add_argument("--media_root", default=None, help="folder holding the real log's media (for source_swap / rehash)")
    ap.add_argument("--out_dir", default="results_bundle")
    ap.add_argument("--targets", default=str(ROOT / "targets_v2.json"))
    a = ap.parse_args(argv)
    r = run(a.n_trials, a.n_records, a.seed, Path(a.out_dir), Path(a.log) if a.log else None,
            Path(a.media_root) if a.media_root else None, Path(a.targets) if a.targets else None)
    print(f"\n  H3 tamper experiment: {a.n_trials} trials per type, source = {r['source']['kind']}")
    print(f"  {'tamper':<17}" + "".join(f"{lv:>20}" for lv in LEVELS))
    for kind in TAMPERS:
        cells = []
        for lv in LEVELS:
            c = r["results"][kind][lv]
            cells.append("n/a" if c["n"] == 0 else f"{c['detected']}/{c['n']} ({c['rate']:.3f})")
        print(f"  {kind:<17}" + "".join(f"{x:>20}" for x in cells))
    if r.get("targets"):
        print(f"  targets {r['targets']['file']}: {r['targets']['freeze_status']}"
              + ("" if r["targets"]["freeze_status"] == "FROZEN" else
                 "  -- NOT pre-registered: do not report these PASS/FAIL lines as pre-registered results"))
    for t in r.get("targets", {}).get("checks", []):
        verdict = "no verdict" if t["pass"] is None else ("PASS" if t["pass"] else "FAIL")
        print(f"  {t['id']}  {t['tamper']:<16} {t['check']:<18} {t['op']} {t['target']}  measured "
              f"{t['measured'] if t['measured'] is None else round(t['measured'], 4)}  {verdict}")
    print(f"  -> {Path(a.out_dir) / 'h3_tamper.json'}, h3_tamper.csv, h3_tamper_trials.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
