"""
athar.review + athar.dashboard: every human decision is a record in a hash chain anchored to the
evidence chain, and the local web app only accepts loopback, CSRF-protected requests.
"""
import io
import json
import urllib.parse
from pathlib import Path

import pytest

from athar import chain, review
from athar import verify as V
from athar.dashboard import Dashboard, serve
import sign_chain_head as SCH


def _tids(rd):
    return [o["track_id"] for o in review.evidence_objects(rd)]


def test_decisions_chained_and_verified(run11):
    rd, keys = run11
    tids = _tids(rd)
    assert len(tids) >= 2
    s = review.ReviewSession(rd, "Analyst A")
    s.decide(tids[0], "approve", reviewer_class="battery", note="clear cells visible")
    s.decide(tids[1], "reject", note="shadow")
    s.decide(tids[1], "approve")                         # changed mind: a new record, history kept
    recs = list(chain.read_records(rd / review.REVIEW_LOG))
    assert [r["record_type"] for r in recs] == ["review_session_open"] + ["review_decision"] * 3
    cur = review.current_decisions(rd)
    assert cur[tids[1]]["decision"] == "approve" and cur[tids[0]]["reviewer_class"] == "battery"
    assert all(r["crop_raw_verified"] for r in recs[1:])
    r = V.verify_run(rd, keys["evidence"]["fingerprint"], require_signature=False)
    assert r["valid"] and r["checks"]["review_chain"] is True and r["details"]["review_chain"]["decisions"] == 3
    # reviewer signs the review chain with their own key
    SCH.sign_head(rd / review.REVIEW_LOG, Path(keys["reviewer"]["private_key"]))
    r = V.verify_run(rd, keys["evidence"]["fingerprint"], keys["reviewer"]["fingerprint"], require_signature=True)
    assert r["valid"] and r["checks"]["review_head"] is True
    # ... and the evidence head is untouched by the review
    assert SCH.verify_head(rd / review.EVIDENCE_LOG, trusted_fingerprint=keys["evidence"]["fingerprint"])["valid"]


def test_bad_input_refused(run11):
    rd, _ = run11
    s = review.ReviewSession(rd, "A")
    with pytest.raises(ValueError):
        s.decide(_tids(rd)[0], "maybe")
    with pytest.raises(KeyError):
        s.decide(999999, "approve")
    with pytest.raises(ValueError):
        review.ReviewSession(rd, "  ")


def test_refuses_broken_evidence(run11):
    rd, keys = run11
    log = rd / review.EVIDENCE_LOG
    lines = log.read_text().splitlines()
    rec = json.loads(lines[1]); rec["n_detections"] = 999
    lines[1] = json.dumps(rec)
    log.write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError):
        review.ReviewSession(rd, "A")


def test_refuses_untrusted_head(run11):
    rd, keys = run11
    vfn = lambda log: SCH.verify_head(log, trusted_fingerprint=keys["reviewer"]["fingerprint"])  # noqa: E731
    with pytest.raises(RuntimeError):
        review.ReviewSession(rd, "A", verify_head_fn=vfn)


def test_edited_decision_detected(run11):
    rd, keys = run11
    s = review.ReviewSession(rd, "A")
    s.decide(_tids(rd)[0], "reject")
    log = rd / review.REVIEW_LOG
    log.write_text(log.read_text().replace('"decision": "reject"', '"decision": "approve"'))
    r = V.verify_run(rd, keys["evidence"]["fingerprint"])
    assert not r["valid"] and r["checks"]["review_chain"] is False and r["checks"]["evidence_chain"]


def test_evidence_swapped_under_decisions(run11):
    """Attacker truncates the evidence log and re-signs it with their own key: the review chain's
    anchor no longer matches (and the head fails against the trusted key)."""
    rd, keys = run11
    review.ReviewSession(rd, "A").decide(_tids(rd)[-1], "approve")
    log = rd / review.EVIDENCE_LOG
    lines = log.read_text().splitlines()
    log.write_text("\n".join(lines[:-1]) + "\n")
    r = V.verify_run(rd)
    assert not r["valid"] and r["checks"]["review_chain"] is False


def test_evidence_change_mid_session_stops(run11):
    rd, _ = run11
    s = review.ReviewSession(rd, "A")
    chain.append(rd / review.EVIDENCE_LOG, {"record_type": "late"})
    with pytest.raises(RuntimeError):
        s.decide(_tids(rd)[0], "approve")


def test_tampered_crop_recorded_in_decision(run11):
    rd, _ = run11
    s = review.ReviewSession(rd, "A")
    o = s.objects[_tids(rd)[0]]
    p = rd / "review_crops" / o["crop_raw"]
    p.write_bytes(p.read_bytes() + b"x")
    d = s.decide(o["track_id"], "approve")
    assert d["crop_raw_present"] and d["crop_raw_verified"] is False


# ── dashboard (WSGI called in-process; no socket) ──

def _call(app, method, path, host="127.0.0.1:8765", body=b""):
    env = {"REQUEST_METHOD": method, "PATH_INFO": path, "HTTP_HOST": host,
           "CONTENT_LENGTH": str(len(body)), "wsgi.input": io.BytesIO(body)}
    out = {}

    def sr(status, headers):
        out["status"], out["headers"] = status, dict(headers)
    out["body"] = b"".join(app(env, sr))
    return out


def test_dashboard_flow(run11):
    rd, _ = run11
    app = Dashboard(rd, "Analyst B")
    tids = _tids(rd)
    r = _call(app, "GET", "/")
    assert r["status"].startswith("200") and f"crop {tids[0]}".encode() in r["body"]
    assert b"raw crop SHA-256 verified" in r["body"]
    img = _call(app, "GET", f"/crop/{tids[0]}")
    assert img["status"].startswith("200") and img["body"][:8] == b"\x89PNG\r\n\x1a\n"
    form = urllib.parse.urlencode({"csrf": app.csrf, "track_id": tids[0], "decision": "approve",
                                   "reviewer_class": "", "note": "ok"}).encode()
    r = _call(app, "POST", "/decide", body=form)
    assert r["status"].startswith("303")
    assert review.current_decisions(rd)[tids[0]]["reviewer"] == "Analyst B"
    st = json.loads(_call(app, "GET", "/status.json")["body"])
    assert st["n_decided"] == 1 and st["evidence_chain_valid"] and st["review_chain_valid"]


def test_dashboard_security(run11):
    rd, _ = run11
    app = Dashboard(rd, "B")
    tid = _tids(rd)[0]
    assert _call(app, "GET", "/", host="evil.example:8765")["status"].startswith("403")      # DNS rebinding
    bad = urllib.parse.urlencode({"csrf": "nope", "track_id": tid, "decision": "approve"}).encode()
    assert _call(app, "POST", "/decide", body=bad)["status"].startswith("403")
    assert tid not in review.current_decisions(rd)
    assert _call(app, "GET", "/crop/..%2F..%2Frun_manifest.json")["status"].startswith("404")
    assert _call(app, "GET", "/crop/999999")["status"].startswith("404")
    junk = urllib.parse.urlencode({"csrf": app.csrf, "track_id": "x", "decision": "approve"}).encode()
    assert _call(app, "POST", "/decide", body=junk)["status"].startswith("400")
    page = _call(app, "GET", "/")["body"].decode()
    assert "<script" not in page and "src=\"http" not in page and "href=\"http" not in page   # nothing external
    with pytest.raises(SystemExit):
        serve(rd, "B", host="0.0.0.0")
