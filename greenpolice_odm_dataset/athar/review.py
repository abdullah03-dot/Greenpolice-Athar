"""
Human-in-the-loop review of an 11 evidence run.

The machine output of 11 is a CANDIDATE.  A person approves or rejects each candidate object, and
every decision is written to  <run_dir>/review_decisions.jsonl  -- a hash chain in the same format
as the evidence log (athar.chain), so tools/sign_chain_head.py signs and verifies it unchanged.

Why a SEPARATE chain anchored to the evidence chain (instead of appending to evidence_audit.jsonl):
  * the evidence log and its signed head stay exactly as the machine wrote them -- reviewing never
    invalidates a signature that was made over the machine output;
  * the first record of the review chain ("review_session_open") pins the evidence log it reviews
    (record count, last record hash, whole-file SHA-256, signed-head fingerprint if any), and every
    decision pins the evidence record it decides (its record_hash).  Editing either chain, or
    swapping the evidence log under the decisions, is detected by athar.verify.

Decisions are never edited: a changed mind is a NEW decision record that supersedes the earlier one
(the history stays in the chain).  The reviewer name is self-declared at session start; to bind it
cryptographically, the reviewer signs the review chain's head with their own key
(python tools/sign_chain_head.py sign --log <run_dir>/review_decisions.jsonl --private_key ...).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import chain

EVIDENCE_LOG = "evidence_audit.jsonl"
REVIEW_LOG = "review_decisions.jsonl"
DECISIONS = ("approve", "reject")


def evidence_objects(run_dir: Path) -> list:
    """Candidate objects from the CHAINED evidence log (not from the unchained evidence_report.json)."""
    out = []
    for rec in chain.read_records(Path(run_dir) / EVIDENCE_LOG):
        ctx = rec.get("context") or {}
        if ctx.get("record_type") != "evidence_object":
            continue
        det = (rec.get("detections") or [{}])[0]
        out.append({"track_id": ctx.get("track_id"), "class_name": det.get("class_name"),
                    "tier": ctx.get("tier", det.get("tier")), "score": ctx.get("score", det.get("confidence")),
                    "consistency": ctx.get("consistency"), "confirmed": ctx.get("confirmed"),
                    "n_views": ctx.get("n_views"), "best_frame": ctx.get("best_frame"),
                    "best_box_xyxy": ctx.get("best_box_xyxy"), "ground_position": ctx.get("ground_position"),
                    "crop_raw": ctx.get("crop_raw"), "crop_raw_sha256": ctx.get("crop_raw_sha256"),
                    "crop_boxed": ctx.get("crop_boxed"), "image_sha256": rec.get("image_sha256"),
                    "gps": rec.get("gps"), "capture_time_utc": rec.get("capture_time_utc"),
                    "capture_time_local": rec.get("capture_time_local"),
                    "evidence_record_index": rec.get("record_index"), "evidence_record_hash": rec.get("record_hash")})
    return out


def evidence_anchor(run_dir: Path) -> dict:
    log = Path(run_dir) / EVIDENCE_LOG
    n, last = chain.tail(log)
    head = Path(str(log) + ".head.json")
    fp = None
    if head.exists():
        try:
            fp = json.loads(head.read_text(encoding="utf-8")).get("public_key_fingerprint")
        except (json.JSONDecodeError, OSError):
            fp = "UNREADABLE"
    return {"evidence_log": EVIDENCE_LOG, "evidence_record_count": n, "evidence_last_record_hash": last,
            "evidence_log_sha256": chain.sha256_file(log), "evidence_head_fingerprint": fp}


def _safe_child(run_dir: Path, rel: Optional[str], sub: str = "review_crops") -> Optional[Path]:
    """Resolve a crop file name strictly inside <run_dir>/<sub> (no path traversal)."""
    if not rel:
        return None
    base = (Path(run_dir) / sub).resolve()
    p = (base / Path(rel).name).resolve()
    try:
        p.relative_to(base)
    except ValueError:
        return None
    return p if p.is_file() else None


def crop_status(run_dir: Path, obj: dict) -> dict:
    """Re-hash the raw crop against the SHA-256 the evidence chain recorded for it."""
    p = _safe_child(run_dir, obj.get("crop_raw"))
    if p is None:
        return {"crop_raw_present": False, "crop_raw_verified": False}
    return {"crop_raw_present": True, "crop_raw_verified": chain.sha256_file(p) == obj.get("crop_raw_sha256")}


class ReviewSession:
    def __init__(self, run_dir: Path, reviewer: str, verify_head_fn=None):
        """verify_head_fn(log_path) -> {"valid": bool, ...}: optional signed-head check of the evidence
        log at session start (athar.verify supplies one bound to a trusted key)."""
        self.run_dir = Path(run_dir)
        self.reviewer = (reviewer or "").strip()
        if not self.reviewer:
            raise ValueError("a reviewer name is required")
        self.evidence_log = self.run_dir / EVIDENCE_LOG
        self.log = self.run_dir / REVIEW_LOG
        v = chain.verify(self.evidence_log)
        if not v["valid"]:
            raise RuntimeError(f"evidence chain is broken ({v.get('reason')} at record {v.get('broken_at')}): "
                               f"refusing to review it. Preserve the run folder as evidence.")
        if verify_head_fn is not None:
            h = verify_head_fn(self.evidence_log)
            if not h.get("valid"):
                raise RuntimeError(f"evidence chain head does not verify: {h.get('reasons')}")
        if self.log.exists() and self.log.stat().st_size:
            rv = chain.verify(self.log)
            if not rv["valid"]:
                raise RuntimeError(f"review chain is broken ({rv.get('reason')} at record {rv.get('broken_at')})")
        self.objects = {o["track_id"]: o for o in evidence_objects(self.run_dir)}
        self.anchor = evidence_anchor(self.run_dir)
        self.session_record = chain.append(self.log, {"record_type": "review_session_open",
                                                      "reviewer": self.reviewer, **self.anchor})

    def decide(self, track_id, decision: str, reviewer_class: Optional[str] = None, note: str = "") -> dict:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            raise ValueError(f"bad track_id {track_id!r}")
        obj = self.objects.get(track_id)
        if obj is None:
            raise KeyError(f"track {track_id} is not an evidence object of this run")
        if evidence_anchor(self.run_dir) != self.anchor:
            raise RuntimeError("the evidence log changed during the review session; stop and investigate")
        cs = crop_status(self.run_dir, obj)
        boxed = _safe_child(self.run_dir, obj.get("crop_boxed"))
        return chain.append(self.log, {
            "record_type": "review_decision", "reviewer": self.reviewer, "track_id": track_id,
            "decision": decision, "machine_class": obj["class_name"],
            "reviewer_class": (reviewer_class or None) if decision == "approve" else None,
            "note": str(note or "")[:2000],
            "evidence_record_index": obj["evidence_record_index"],
            "evidence_record_hash": obj["evidence_record_hash"],
            "session_record_hash": self.session_record["record_hash"],
            **cs, "crop_boxed_sha256": chain.sha256_file(boxed) if boxed else None})

    def log_event(self, record_type: str, **fields) -> dict:
        """Other actors (the agent) record what they produced in the same chain."""
        return chain.append(self.log, {"record_type": record_type, "reviewer": self.reviewer,
                                       "session_record_hash": self.session_record["record_hash"], **fields})


def current_decisions(run_dir: Path) -> dict:
    """track_id -> latest decision record (later decisions supersede earlier ones)."""
    log = Path(run_dir) / REVIEW_LOG
    out = {}
    if not log.exists():
        return out
    for rec in chain.read_records(log):
        if rec.get("record_type") == "review_decision":
            out[rec["track_id"]] = rec
    return out
