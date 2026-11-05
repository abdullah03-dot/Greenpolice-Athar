"""
Reviewer dashboard: a local-only web app where a person approves or rejects each candidate object of
an 11 run.  Every decision is a record in <run_dir>/review_decisions.jsonl (athar.review).

  python -m athar.dashboard <run_dir> --reviewer "<name>" [--port 8765]
         [--trusted_fingerprint sha256:...]     # refuse to open unless the evidence head verifies
  then open http://127.0.0.1:8765/

Standard library only (wsgiref): nothing to install, no CDN, no external fonts or scripts -- the page
works with the network cable pulled.  Safety:
  * binds to a loopback address only (anything else is refused);
  * rejects requests whose Host header is not loopback (DNS-rebinding);
  * every POST needs the per-process CSRF token embedded in the page (another web page open in the
    same browser cannot submit decisions);
  * crops are served only from <run_dir>/review_crops, after re-hashing the raw crop against the
    SHA-256 recorded in the evidence chain; a mismatch is shown in red and recorded in the decision.
"""
from __future__ import annotations

import argparse
import html
import json
import secrets
import sys
import urllib.parse
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, make_server

from . import agent, chain, review

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}

CSS = """body{font:14px system-ui,sans-serif;margin:16px;background:#fafafa;color:#111}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}
th{background:#eee;text-align:left}.ok{color:#060}.bad{color:#b00;font-weight:bold}
.approve{background:#e6f4e6}.reject{background:#f8e3e3}img{max-width:220px;max-height:220px}
form{margin:0}input[type=text]{width:12em}.banner{padding:8px;margin-bottom:12px;border:1px solid #999}"""


class Dashboard:
    def __init__(self, run_dir: Path, reviewer: str, verify_head_fn=None):
        self.run_dir = Path(run_dir)
        self.session = review.ReviewSession(self.run_dir, reviewer, verify_head_fn=verify_head_fn)
        self.csrf = secrets.token_urlsafe(32)

    # ── WSGI ──
    def __call__(self, environ, start_response):
        try:
            host = urllib.parse.urlsplit("//" + (environ.get("HTTP_HOST") or "")).hostname or ""
        except ValueError:
            host = ""
        if host not in LOOPBACK_HOSTS:
            return self._send(start_response, "403 Forbidden", "text/plain", b"loopback Host header required")
        path, method = environ.get("PATH_INFO", "/"), environ.get("REQUEST_METHOD", "GET")
        try:
            if method == "GET" and path == "/":
                return self._send(start_response, "200 OK", "text/html; charset=utf-8", self.index().encode())
            if method == "GET" and path.startswith("/crop/"):
                return self._crop(start_response, path[len("/crop/"):])
            if method == "GET" and path == "/status.json":
                return self._send(start_response, "200 OK", "application/json", json.dumps(self.status()).encode())
            if method == "POST" and path == "/decide":
                n = int(environ.get("CONTENT_LENGTH") or 0)
                form = urllib.parse.parse_qs(environ["wsgi.input"].read(min(n, 65536)).decode("utf-8", "replace"))
                f = {k: v[0] for k, v in form.items()}
                if not secrets.compare_digest(f.get("csrf", ""), self.csrf):
                    return self._send(start_response, "403 Forbidden", "text/plain", b"bad CSRF token")
                self.session.decide(f.get("track_id"), f.get("decision", ""), f.get("reviewer_class") or None,
                                    f.get("note", ""))
                start_response("303 See Other", [("Location", "/"), ("Content-Length", "0")])
                return [b""]
        except (ValueError, KeyError) as e:
            return self._send(start_response, "400 Bad Request", "text/plain", str(e).encode())
        except RuntimeError as e:      # evidence changed under us: stop, loudly
            return self._send(start_response, "409 Conflict", "text/plain", str(e).encode())
        return self._send(start_response, "404 Not Found", "text/plain", b"not found")

    @staticmethod
    def _send(start_response, status, ctype, body: bytes):
        start_response(status, [("Content-Type", ctype), ("Content-Length", str(len(body))),
                                ("Cache-Control", "no-store"), ("X-Frame-Options", "DENY"),
                                ("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'")])
        return [body]

    def _crop(self, start_response, tid: str):
        try:
            obj = self.session.objects[int(tid)]
        except (ValueError, KeyError):
            return self._send(start_response, "404 Not Found", "text/plain", b"no such track")
        p = review._safe_child(self.run_dir, obj.get("crop_raw"))
        if p is None or p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            return self._send(start_response, "404 Not Found", "text/plain", b"no crop")
        ctype = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        return self._send(start_response, "200 OK", ctype, p.read_bytes())

    # ── views ──
    def status(self) -> dict:
        dec = review.current_decisions(self.run_dir)
        ev = chain.verify(self.session.evidence_log)
        rv = chain.verify(self.session.log)
        return {"reviewer": self.session.reviewer, "n_candidates": len(self.session.objects),
                "n_decided": len(dec), "evidence_chain_valid": ev["valid"], "review_chain_valid": rv["valid"],
                "review_records": rv.get("records")}

    def index(self) -> str:
        e = html.escape
        st = self.status()
        dec = review.current_decisions(self.run_dir)
        rows = []
        for o in agent.triage(list(self.session.objects.values())):
            tid = o["track_id"]
            cs = review.crop_status(self.run_dir, o)
            d = dec.get(tid)
            crop = (f'<img src="/crop/{tid}" alt="crop {tid}"><br><span class="ok">raw crop SHA-256 verified</span>'
                    if cs["crop_raw_verified"] else
                    '<span class="bad">CROP MISSING OR HASH MISMATCH</span>' if cs["crop_raw_present"] or o.get("crop_raw")
                    else "no crop")
            pos = o.get("ground_position") or {}
            state = (f'<b>{e(d["decision"].upper())}</b> by {e(d["reviewer"])}'
                     f'{" as " + e(d["reviewer_class"]) if d.get("reviewer_class") else ""}<br><small>{e(d.get("note") or "")}</small>'
                     if d else "<i>undecided</i>")
            form = (f'<form method="post" action="/decide"><input type="hidden" name="csrf" value="{e(self.csrf)}">'
                    f'<input type="hidden" name="track_id" value="{tid}">'
                    f'<input type="text" name="reviewer_class" placeholder="correct class (optional)"><br>'
                    f'<input type="text" name="note" placeholder="note"><br>'
                    f'<button name="decision" value="approve">Approve</button> '
                    f'<button name="decision" value="reject">Reject</button></form>')
            rows.append(f'<tr class="{e(d["decision"]) if d else ""}"><td>{o["triage_rank"]}</td><td>{tid}</td>'
                        f'<td>{e(str(o["class_name"]))}<br><small>{e(o["triage_reason"])}</small></td>'
                        f'<td>views {e(str(o.get("n_views")))}<br>consistency {e(str(o.get("consistency")))}<br>'
                        f'frame {e(str(o.get("best_frame")))}<br>lat {e(str(pos.get("lat", "")))} lon {e(str(pos.get("lon", "")))}<br>'
                        f'captured {e(str(o.get("capture_time_utc") or o.get("capture_time_local") or "unknown"))}</td>'
                        f'<td>{crop}</td><td>{state}</td><td>{form}</td></tr>')
        ok = st["evidence_chain_valid"] and st["review_chain_valid"]
        return (f"<!doctype html><html><head><meta charset='utf-8'><title>ATHAR review</title><style>{CSS}</style></head><body>"
                f"<h1>ATHAR evidence review</h1><div class='banner'>Run: {e(self.run_dir.name)} &middot; reviewer: "
                f"<b>{e(st['reviewer'])}</b> &middot; {st['n_decided']}/{st['n_candidates']} decided &middot; "
                f"<span class='{'ok' if ok else 'bad'}'>evidence chain {'valid' if st['evidence_chain_valid'] else 'BROKEN'}, "
                f"review chain {'valid' if st['review_chain_valid'] else 'BROKEN'} ({st['review_records']} records)</span>"
                f"<br><small>Machine output is a CANDIDATE. Each decision is appended to review_decisions.jsonl "
                f"(hash-chained, anchored to the evidence log). Changing a decision adds a new record; nothing is "
                f"overwritten.</small></div>"
                f"<table><tr><th>#</th><th>track</th><th>machine class</th><th>context</th><th>crop</th>"
                f"<th>current decision</th><th>decide</th></tr>{''.join(rows)}</table></body></html>")


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("  [dashboard] " + fmt % args + "\n")


def serve(run_dir: Path, reviewer: str, host: str = "127.0.0.1", port: int = 8765, verify_head_fn=None):
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit(f"[ERROR] the dashboard binds to loopback only (got {host!r})")
    app = Dashboard(run_dir, reviewer, verify_head_fn)
    with make_server(host, port, app, handler_class=_QuietHandler) as srv:
        print(f"  ATHAR review dashboard: http://{host}:{port}/  (Ctrl+C to stop)")
        srv.serve_forever()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--trusted_fingerprint", default=None,
                    help="refuse to start unless the evidence log's signed head verifies against this key")
    a = ap.parse_args(argv)
    vfn = None
    if a.trusted_fingerprint:
        from .verify import _sch
        vfn = lambda log: _sch().verify_head(log, trusted_fingerprint=a.trusted_fingerprint)  # noqa: E731
    serve(Path(a.run_dir), a.reviewer, a.host, a.port, vfn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
