"""
Hash-chained JSONL log -- an independent second implementation of the convention used by
07_edge_benchmark.ForensicAuditLogger, so logs written here verify with 07 (and with
tools/sign_chain_head.py) and logs written by 07/11 verify here:

    record_hash = sha256( json.dumps(record_without_record_hash, sort_keys=True, default=str) )
    record["prev_hash"] = previous record's record_hash, or "GENESIS" for the first record

This module adds a check 07 does not make: record_index must equal the line number (07 writes it but
does not verify it).  It has no third-party dependencies, so the verifier runs anywhere.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

GENESIS = "GENESIS"


def payload_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lines(log: Path) -> list:
    with open(log, "r", encoding="utf-8") as f:
        return [ln for ln in f.read().splitlines() if ln.strip()]


def read_records(log: Path) -> Iterator[dict]:
    for ln in _lines(Path(log)):
        yield json.loads(ln)


def verify(log: Path, check_index: bool = True) -> dict:
    """-> {"valid": True, "records": n, "chain_tail": hash} or
          {"valid": False, "broken_at": i, "reason": ...}."""
    log = Path(log)
    if not log.exists():
        return {"valid": False, "broken_at": None, "reason": f"log not found: {log}"}
    try:
        raw = log.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return {"valid": False, "broken_at": None, "reason": "log is not valid UTF-8"}
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    prev = GENESIS
    for i, ln in enumerate(lines):
        try:
            rec = json.loads(ln)
            stored = rec.pop("record_hash")
        except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
            return {"valid": False, "broken_at": i, "reason": "unparseable record or missing record_hash"}
        if rec.get("prev_hash") != prev:
            return {"valid": False, "broken_at": i, "reason": "prev_hash mismatch",
                    "expected": prev, "found": rec.get("prev_hash")}
        if payload_hash(rec) != stored:
            return {"valid": False, "broken_at": i, "reason": "record_hash mismatch (record edited)"}
        if check_index and rec.get("record_index") != i:
            return {"valid": False, "broken_at": i, "reason": "record_index does not match position",
                    "expected": i, "found": rec.get("record_index")}
        prev = stored
    return {"valid": True, "records": len(lines), "chain_tail": prev}


def tail(log: Path) -> tuple:
    """(record_count, last_record_hash) of an existing log (GENESIS for an empty / missing one)."""
    log = Path(log)
    if not log.exists():
        return 0, GENESIS
    lines = _lines(log)
    return (len(lines), json.loads(lines[-1])["record_hash"]) if lines else (0, GENESIS)


def append(log: Path, payload: dict, verify_first: bool = True) -> dict:
    """Append one record.  The existing chain is verified first (appending to a broken chain would bury
    the break under valid-looking records) -- same rule as 07's logger on resume."""
    log = Path(log)
    if any(k in payload for k in ("record_hash", "prev_hash", "record_index")):
        raise ValueError("payload must not set record_hash / prev_hash / record_index")
    if verify_first and log.exists() and log.stat().st_size:
        v = verify(log)
        if not v["valid"]:
            raise RuntimeError(f"refusing to append to {log}: chain broken at record {v.get('broken_at')} "
                               f"({v.get('reason')}). Preserve it as evidence; start a new log.")
    n, prev = tail(log)
    rec = {**payload, "record_index": n, "prev_hash": prev}
    rec.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    rec["record_hash"] = payload_hash(rec)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
    return rec


def record_at(log: Path, index: int) -> Optional[dict]:
    lines = _lines(Path(log))
    return json.loads(lines[index]) if 0 <= index < len(lines) else None
