"""
Verify a whole 11 evidence run folder end to end.  Exit 0 = every REQUIRED check passed.

  python -m athar.verify evidence/2026-10-03/run_<UTC>_<sha> --trusted_fingerprint sha256:...
         [--review_fingerprint sha256:...] [--media_root D:/flights/2026-10-03/low]
         [--require_signature] [--require_media] [--json out.json]

Checks (each reported as true / false / null = not checked):
  evidence_chain        evidence_audit.jsonl: every record_hash, prev_hash link and record_index
  evidence_head         signed head (tools/sign_chain_head.py) valid against a TRUSTED key, and it
                        pins the current record count, last hash and whole-file SHA-256 -- catches
                        tail truncation and full re-hashing, which the chain alone cannot
  manifest              run_manifest.json SHA-256 == the hash every record cites == the run folder
                        suffix; manifest's signing-key fingerprint == the head's
  crops                 every review crop's raw file re-hashes to the SHA-256 in the evidence chain
  media                 every source image re-hashes to image_sha256 (needs the media; null if absent)
  review_chain          review_decisions.jsonl chain valid; every session anchor matches the evidence
                        log; every decision cites an evidence record that exists with that hash
  review_head           signed head of the review chain (if --review_fingerprint / --require_signature)

A signature is only meaningful against a key you trust independently -- pass the fingerprint you
recorded OUTSIDE this system.  Without one, heads are reported but the run is not 'signed-valid'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from . import chain
from .review import EVIDENCE_LOG, REVIEW_LOG, _safe_child

ROOT = Path(__file__).resolve().parents[1]


def _sch():
    sys.path.insert(0, str(ROOT / "tools"))
    import sign_chain_head as SCH  # noqa: E402
    return SCH


def _head_check(log: Path, fingerprint: Optional[str], required: bool) -> dict:
    head = Path(str(log) + ".head.json")
    if not head.exists():
        return {"ok": False if required else None, "reasons": [f"no signed head {head.name}"]}
    if not fingerprint:
        return {"ok": False if required else None,
                "reasons": ["signed head present but no trusted fingerprint given: not verified"]}
    r = _sch().verify_head(log, trusted_fingerprint=fingerprint)
    return {"ok": bool(r["valid"]), "reasons": r["reasons"], "checks": r["checks"]}


def verify_run(run_dir: Path, trusted_fingerprint: Optional[str] = None, review_fingerprint: Optional[str] = None,
               media_root: Optional[Path] = None, require_signature: bool = False,
               require_media: bool = False) -> dict:
    run_dir = Path(run_dir)
    ev = run_dir / EVIDENCE_LOG
    res = {"run_dir": str(run_dir), "checks": {}, "details": {}}
    C, D = res["checks"], res["details"]

    v = chain.verify(ev)
    C["evidence_chain"] = v["valid"]
    D["evidence_chain"] = v
    records = list(chain.read_records(ev)) if v["valid"] else []

    h = _head_check(ev, trusted_fingerprint, require_signature)
    C["evidence_head"] = h["ok"]
    D["evidence_head"] = h

    # manifest
    mp = run_dir / "run_manifest.json"
    if mp.exists():
        msha = chain.sha256_file(mp)
        cited = {(r.get("context") or {}).get("run_manifest_sha256") for r in records}
        problems = []
        if records and cited != {msha}:
            problems.append(f"records cite manifest hashes {sorted(map(str, cited))}, file is {msha}")
        suffix = run_dir.name.rsplit("_", 1)[-1]
        if run_dir.name.startswith("run_") and not msha.startswith(suffix):
            problems.append(f"run folder suffix {suffix} != manifest sha {msha[:10]}")
        try:
            mfp = json.loads(mp.read_text(encoding="utf-8")).get("signing_public_key_fingerprint")
        except json.JSONDecodeError:
            mfp = None
            problems.append("manifest is not valid JSON")
        hp = Path(str(ev) + ".head.json")
        if mfp and hp.exists():
            hfp = json.loads(hp.read_text(encoding="utf-8")).get("public_key_fingerprint")
            if hfp != mfp:
                problems.append(f"head signed by {hfp}, manifest names {mfp}")
        C["manifest"] = not problems if v["valid"] else False
        D["manifest"] = {"sha256": msha, "problems": problems}
    else:
        C["manifest"] = False
        D["manifest"] = {"problems": ["run_manifest.json missing"]}

    # crops + media
    bad_crops, n_crops, bad_media, missing_media, n_media = [], 0, [], [], 0
    seen = set()
    for r in records:
        ctx = r.get("context") or {}
        if ctx.get("crop_raw"):
            n_crops += 1
            p = _safe_child(run_dir, ctx["crop_raw"])
            if p is None or chain.sha256_file(p) != ctx.get("crop_raw_sha256"):
                bad_crops.append(ctx["crop_raw"])
        ip = r.get("image_path")
        if ip and (ip, r.get("image_sha256")) not in seen:
            seen.add((ip, r.get("image_sha256")))
            ref = ip.split("#", 1)[0]          # video frames are "<file>#<frame>": not re-hashable as files
            # --media_root (a copy of the flight media) is checked INSTEAD of the recorded path when given
            cand = [Path(media_root) / Path(ref).name] if media_root else [Path(ref)]
            f = next((c for c in cand if c.is_file()), None)
            if f is None or "#" in ip:
                missing_media.append(ip)
                continue
            n_media += 1
            if chain.sha256_file(f) != r.get("image_sha256"):
                bad_media.append(ip)
    C["crops"] = (not bad_crops) if n_crops else None
    D["crops"] = {"checked": n_crops, "mismatched_or_missing": bad_crops}
    if bad_media:
        C["media"] = False
    elif missing_media:
        C["media"] = False if require_media else (True if n_media else None)
    else:
        C["media"] = True if n_media else None
    D["media"] = {"checked": n_media, "mismatched": bad_media, "not_available": missing_media[:50],
                  "n_not_available": len(missing_media)}

    # review chain
    rl = run_dir / REVIEW_LOG
    if rl.exists():
        rv = chain.verify(rl)
        problems = [] if rv["valid"] else [f"review chain broken: {rv.get('reason')} at {rv.get('broken_at')}"]
        by_hash = {r["record_hash"]: r for r in records}
        n_dec = 0
        if rv["valid"]:
            for rec in chain.read_records(rl):
                t = rec.get("record_type")
                if t == "review_session_open":
                    n = rec.get("evidence_record_count")
                    anchor = chain.record_at(ev, n - 1) if n else None
                    if not n or anchor is None or anchor.get("record_hash") != rec.get("evidence_last_record_hash"):
                        problems.append(f"session {rec['record_index']}: anchor does not match the evidence log")
                    elif n == len(records) and rec.get("evidence_log_sha256") != chain.sha256_file(ev):
                        problems.append(f"session {rec['record_index']}: evidence log bytes differ from anchor")
                elif t == "review_decision":
                    n_dec += 1
                    e = by_hash.get(rec.get("evidence_record_hash"))
                    if e is None or (e.get("context") or {}).get("track_id") != rec.get("track_id"):
                        problems.append(f"decision {rec['record_index']}: cites no matching evidence record")
        C["review_chain"] = not problems
        D["review_chain"] = {"decisions": n_dec, "problems": problems}
        rh = _head_check(rl, review_fingerprint, require_signature)
        C["review_head"] = rh["ok"]
        D["review_head"] = rh
    else:
        C["review_chain"] = None
        C["review_head"] = None
        D["review_chain"] = {"problems": ["no review_decisions.jsonl (run not reviewed)"]}

    res["valid"] = all(x is not False for x in C.values())
    res["signed_valid"] = res["valid"] and C["evidence_head"] is True
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--trusted_fingerprint", default=None, help="evidence-head signer, recorded outside the system")
    ap.add_argument("--public_key", default=None, help="alternative to --trusted_fingerprint (PEM)")
    ap.add_argument("--review_fingerprint", default=None, help="reviewer's signing key fingerprint")
    ap.add_argument("--media_root", default=None)
    ap.add_argument("--require_signature", action="store_true", help="a missing / unverified head is a failure")
    ap.add_argument("--require_media", action="store_true", help="unavailable source media is a failure")
    ap.add_argument("--json", default=None, help="also write the full report here")
    a = ap.parse_args(argv)
    fp = a.trusted_fingerprint or (_sch().public_key_fingerprint(a.public_key) if a.public_key else None)
    r = verify_run(Path(a.run_dir), fp, a.review_fingerprint, Path(a.media_root) if a.media_root else None,
                   a.require_signature, a.require_media)
    for k, val in r["checks"].items():
        print(f"  {k:<16} {'PASS' if val is True else 'FAIL' if val is False else 'not checked'}")
        if val is False:
            d = r["details"].get(k, {})
            for msg in (d.get("problems") or d.get("reasons") or [d.get("reason")] or [])[:10]:
                if msg:
                    print(f"      - {msg}")
            for key in ("mismatched_or_missing", "mismatched"):
                for x in d.get(key, [])[:10]:
                    print(f"      - {key}: {x}")
    print(f"  {'VALID' if r['valid'] else 'INVALID'}"
          f"{' (signed head verified against trusted key)' if r['signed_valid'] else ' (head NOT verified)' if r['valid'] else ''}")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    return 0 if r["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
