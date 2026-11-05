"""
athar.chain + athar.verify: chain-of-custody end to end on a REAL 11 run (fake pixel detector, CPU).

  * athar.chain is format-compatible with 07's ForensicAuditLogger in both directions, and
    tools/sign_chain_head.py signs / verifies logs written by athar.chain;
  * athar.chain additionally checks record_index (07 does not);
  * python -m athar.verify passes an untampered signed run and FAILS, on the right check, for:
    field edit, tail truncation, full rebuild (re-hashed chain), crop swap, source-media swap,
    manifest edit, untrusted / wrong signing key.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from athar import chain
from athar import verify as V
import sign_chain_head as SCH

ROOT = Path(__file__).resolve().parents[1]


def _b07():
    spec = importlib.util.spec_from_file_location("b07_athar", ROOT / "07_edge_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rewrite(log: Path, edit_fn, rehash: bool):
    """Apply edit_fn to the parsed records; optionally re-hash the whole chain (an attacker's rebuild)."""
    recs = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    recs = edit_fn(recs)
    if rehash:
        prev = chain.GENESIS
        for i, r in enumerate(recs):
            r.pop("record_hash", None)
            r["prev_hash"], r["record_index"] = prev, i
            r["record_hash"] = prev = chain.payload_hash(r)
    log.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def test_compat_with_07_both_directions(tmp_path):
    b07 = _b07()
    a = tmp_path / "a.jsonl"
    lg = b07.ForensicAuditLogger(a, "m")
    for i in range(3):
        lg.log_detection(f"img{i}.jpg", "0" * 64, [], None, None, 0.5)
    assert chain.verify(a)["valid"] and chain.verify(a)["records"] == 3
    b = tmp_path / "b.jsonl"
    for i in range(4):
        chain.append(b, {"record_type": "x", "i": i})
    assert b07.ForensicAuditLogger(b, "v", verify_on_open=False).verify_chain()["valid"]
    k = SCH.keygen(tmp_path / "keys" / "k.pem")
    SCH.sign_head(b, Path(k["private_key"]))
    assert SCH.verify_head(b, trusted_fingerprint=k["fingerprint"])["valid"]


def test_record_index_checked(tmp_path):
    log = tmp_path / "c.jsonl"
    for i in range(3):
        chain.append(log, {"i": i})
    _rewrite(log, lambda rs: rs, rehash=True)
    assert chain.verify(log)["valid"]
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    recs[1]["record_index"] = 7          # consistent hashes, wrong index
    prev = chain.GENESIS
    for r in recs:
        r.pop("record_hash")
        r["prev_hash"] = prev
        r["record_hash"] = prev = chain.payload_hash(r)
    log.write_text("".join(json.dumps(r) + "\n" for r in recs))
    v = chain.verify(log)
    assert not v["valid"] and v["broken_at"] == 1 and "record_index" in v["reason"]


def test_append_refuses_broken_chain(tmp_path):
    log = tmp_path / "d.jsonl"
    for i in range(3):
        chain.append(log, {"i": i})
    _rewrite(log, lambda rs: [{**rs[0], "i": 99}] + rs[1:], rehash=False)
    with pytest.raises(RuntimeError):
        chain.append(log, {"i": 3})
    with pytest.raises(ValueError):
        chain.append(tmp_path / "e.jsonl", {"prev_hash": "x"})


def test_clean_run_verifies(run11):
    rd, keys = run11
    r = V.verify_run(rd, keys["evidence"]["fingerprint"], require_signature=True, require_media=True)
    assert r["valid"] and r["signed_valid"], r["checks"]
    c = r["checks"]
    assert c["evidence_chain"] and c["evidence_head"] and c["manifest"] and c["crops"] and c["media"]
    assert r["details"]["crops"]["checked"] > 0 and r["details"]["media"]["checked"] == 6
    assert V.main([str(rd), "--trusted_fingerprint", keys["evidence"]["fingerprint"], "--require_signature"]) == 0
    assert V.main([str(rd), "--public_key", keys["evidence"]["public_key"], "--require_signature"]) == 0


def test_signature_policy(run11):
    rd, keys = run11
    r = V.verify_run(rd)                                   # no trusted key: not signed-valid
    assert r["valid"] and not r["signed_valid"] and r["checks"]["evidence_head"] is None
    assert not V.verify_run(rd, require_signature=True)["valid"]
    assert not V.verify_run(rd, keys["reviewer"]["fingerprint"])["valid"]          # wrong key
    assert V.main([str(rd), "--require_signature"]) == 1


def _log(rd):
    return rd / "evidence_audit.jsonl"


def test_field_edit_detected(run11):
    rd, keys = run11

    def edit(rs):
        rs[2]["detections"] = []
        return rs
    _rewrite(_log(rd), edit, rehash=False)
    r = V.verify_run(rd, keys["evidence"]["fingerprint"])
    assert not r["valid"] and r["checks"]["evidence_chain"] is False


def test_tail_truncation_detected_only_by_head(run11):
    rd, keys = run11
    _rewrite(_log(rd), lambda rs: rs[:-2], rehash=False)
    assert chain.verify(_log(rd))["valid"]               # the chain alone cannot see this
    r = V.verify_run(rd, keys["evidence"]["fingerprint"])
    assert not r["valid"] and r["checks"]["evidence_head"] is False


def test_full_rebuild_detected_only_by_head(run11):
    rd, keys = run11

    def edit(rs):
        for x in rs:
            if (x.get("context") or {}).get("record_type") == "evidence_object":
                x["detections"][0]["class_name"] = "microwave"
                break
        return rs
    _rewrite(_log(rd), edit, rehash=True)
    assert chain.verify(_log(rd))["valid"]
    r = V.verify_run(rd, keys["evidence"]["fingerprint"])
    assert not r["valid"] and r["checks"]["evidence_head"] is False


def test_crop_swap_detected(run11):
    rd, keys = run11
    crop = sorted((rd / "review_crops").glob("*_raw.png"))[0]   # *_boxed.png is NOT hash-pinned by 11
    crop.write_bytes(crop.read_bytes() + b"\0")
    r = V.verify_run(rd, keys["evidence"]["fingerprint"])
    assert not r["valid"] and r["checks"]["crops"] is False and r["checks"]["evidence_chain"]


def test_media_swap_detected(run11, tmp_path):
    rd, keys = run11
    media = tmp_path / "media"
    media.mkdir()
    for f in keys["media"].iterdir():
        (media / f.name).write_bytes(f.read_bytes())
    assert V.verify_run(rd, keys["evidence"]["fingerprint"], media_root=media)["checks"]["media"] is True
    victim = sorted(media.iterdir())[0]
    victim.write_bytes(victim.read_bytes()[:-1] + b"\x01")
    r = V.verify_run(rd, keys["evidence"]["fingerprint"], media_root=media)
    assert not r["valid"] and r["checks"]["media"] is False
    victim.unlink()
    assert V.verify_run(rd, keys["evidence"]["fingerprint"], media_root=media)["valid"]   # missing = not checked
    assert not V.verify_run(rd, keys["evidence"]["fingerprint"], media_root=media, require_media=True)["valid"]


def test_manifest_edit_detected(run11):
    rd, keys = run11
    m = rd / "run_manifest.json"
    m.write_text(m.read_text().replace('"created_utc"', '"created_utc_x"'))
    r = V.verify_run(rd, keys["evidence"]["fingerprint"])
    assert not r["valid"] and r["checks"]["manifest"] is False
