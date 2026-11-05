"""
athar.agent: deterministic triage, report drafting from human-APPROVED facts only, hallucinated
numbers flagged, drafts logged in the review chain, and no network use (Ollama only on loopback).
"""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from athar import agent, chain, review
from athar import verify as V


def test_triage_rule():
    objs = [{"track_id": 1, "tier": 2, "confirmed": True, "score": 0.9},
            {"track_id": 2, "tier": 1, "confirmed": False, "score": 0.99},
            {"track_id": 3, "tier": 1, "confirmed": True, "score": 0.3},
            {"track_id": 4, "tier": 2, "confirmed": True, "score": 0.9}]
    t = agent.triage(objs)
    assert [o["track_id"] for o in t] == [3, 2, 1, 4]
    assert [o["triage_rank"] for o in t] == [1, 2, 3, 4] and "tier-1" in t[0]["triage_reason"]
    assert agent.triage(list(reversed(objs))) == t


@pytest.fixture
def no_network(monkeypatch):
    """Any connection to a non-loopback address fails the test."""
    real = socket.socket.connect

    def guard(self, addr):
        host = addr[0] if isinstance(addr, tuple) else addr
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"network access attempted: {addr}")
        return real(self, addr)
    monkeypatch.setattr(socket.socket, "connect", guard)


def test_draft_with_stub(run11, no_network):
    rd, keys = run11
    tids = [o["track_id"] for o in review.evidence_objects(rd)]
    s = review.ReviewSession(rd, "Analyst C")
    s.decide(tids[0], "approve")
    s.decide(tids[1], "reject")
    d1 = agent.draft_report(rd, agent.StubBackend(), s)
    f = d1["facts"]
    assert f["n_approved"] == 1 and f["n_rejected"] == 1 and f["n_unreviewed"] == len(tids) - 2
    assert [a["track_id"] for a in f["approved"]] == [tids[0]] and f["rejected_track_ids"] == [tids[1]]
    assert d1["status"].startswith("DRAFT") and d1["unsupported_numbers"] == []
    d2 = agent.draft_report(rd, agent.StubBackend(), s)
    assert d2["sha256"] == d1["sha256"]                    # deterministic
    recs = [r for r in chain.read_records(rd / review.REVIEW_LOG) if r["record_type"] == "agent_draft"]
    assert len(recs) == 2 and recs[0]["draft_sha256"] == d1["sha256"]
    assert chain.sha256_file(rd / recs[0]["draft_file"]) == d1["sha256"]
    assert V.verify_run(rd, keys["evidence"]["fingerprint"])["valid"]


class _Liar:
    name, model = "fake", "liar"

    def generate(self, prompt, seed=0):
        return "The reviewer approved 1 object; 57 drums and 3.5 tonnes of waste were found."


def test_unsupported_numbers_flagged(run11):
    rd, _ = run11
    s = review.ReviewSession(rd, "A")
    s.decide(review.evidence_objects(rd)[0]["track_id"], "approve")
    d = agent.draft_report(rd, _Liar(), s)
    assert d["unsupported_numbers"] == ["3.5", "57"]


def test_ollama_loopback_only():
    for url in ("http://10.0.0.5:11434", "http://example.com:11434", "https://127.0.0.1:11434"):
        with pytest.raises(ValueError):
            agent.OllamaBackend("m", url)
    agent.OllamaBackend("m", "http://localhost:11434")


def test_ollama_http_path(monkeypatch, no_network):
    """A fake Ollama on 127.0.0.1 -- exercises the real HTTP code without a model or network, with a
    bogus proxy configured (the backend must bypass it)."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            out = {"response": "ok text"} if self.path == "/api/generate" else {"digest": "sha256:abc"}
            b = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    monkeypatch.setenv("HTTP_PROXY", "http://203.0.113.1:9")
    monkeypatch.setenv("http_proxy", "http://203.0.113.1:9")
    try:
        b = agent.OllamaBackend("llama3.1:8b", f"http://127.0.0.1:{srv.server_port}", timeout_s=5)
        assert b.generate("hello", seed=7) == "ok text"
        assert b.model_digest() == "sha256:abc"
    finally:
        srv.shutdown()
    gen = [x for x in seen if x[0] == "/api/generate"][0][1]
    assert gen["options"] == {"temperature": 0, "seed": 7} and gen["stream"] is False


def test_cli(run11, capsys):
    rd, _ = run11
    assert agent.main(["triage", str(rd)]) == 0
    assert "#  1" in capsys.readouterr().out
    assert agent.main(["draft", str(rd), "--reviewer", "A"]) == 0
    assert (rd / "agent_drafts").is_dir()
