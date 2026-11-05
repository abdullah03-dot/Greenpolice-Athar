#!/usr/bin/env python3
"""
Signed chain head for the hash-chained audit logs of 07 / 11 (ATHAR forensic hardening, 2026-10-08).

WHY.  A hash chain proves that no record was edited, reordered, inserted or deleted *in the middle*.
It cannot prove that records were not removed from the END (truncation), nor that the whole chain was
not rebuilt after an edit (re-hashing every later record).  A signed head pins the chain: record
count, first and last record hash and the SHA-256 of the whole log file, signed with an Ed25519 key
the attacker does not hold.

  keygen   python tools/sign_chain_head.py keygen --private_key D:/keys/athar_ed25519.pem
           (refuses any path inside this repository; --passphrase_env NAME encrypts the key with the
            passphrase in environment variable NAME)
  sign     python tools/sign_chain_head.py sign --log runs/.../forensic_audit.jsonl --private_key D:/keys/athar_ed25519.pem
           verifies the chain first (refuses a broken one); writes <log>.head.json; a previous head is
           kept as <log>.head.<UTC>.json
  verify   python tools/sign_chain_head.py verify --log <log> --trusted_fingerprint sha256:...
           (or --public_key <pem>).  Exit 0 = valid.  A signature is only meaningful against a key you
           trust independently: without --trusted_fingerprint / --public_key the result is 'not trusted'.

PRODUCTION NOTE.  Keep the private key on a hardware token (smart card / HSM / YubiKey PIV), not in a
file, and record its fingerprint outside this system (case file, email to the supervisor).  The next
step is RFC 3161 trusted timestamping of each head, so the signing time is attested by a third party.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
HEAD_FORMAT = "ATHAR-chain-head-v1"


def _crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
        return serialization, Ed25519PrivateKey, Ed25519PublicKey
    except ImportError:
        raise SystemExit("[ERROR] needs the 'cryptography' package (pip install 'cryptography>=42')")


_B07 = None


def _bench07():
    """07 loaded once per process (the tamper experiment verifies thousands of logs)."""
    global _B07
    if _B07 is None:
        spec = importlib.util.spec_from_file_location("bench07_sign", ROOT / "07_edge_benchmark.py")
        _B07 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_B07)
    return _B07


def _verify_chain(log_path: Path) -> dict:
    """07's ForensicAuditLogger.verify_chain -- the single implementation of chain checking."""
    return _bench07().ForensicAuditLogger(log_path, "verify-only", verify_on_open=False).verify_chain()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_public_bytes(pub) -> bytes:
    serialization, _, _ = _crypto()
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def fingerprint_of(pub_raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(pub_raw).hexdigest()


def load_public_key(path: Path):
    serialization, _, _ = _crypto()
    return serialization.load_pem_public_key(Path(path).read_bytes())


def public_key_fingerprint(path) -> str:
    """Fingerprint of a PEM public key file (recorded in 11's run manifest)."""
    return fingerprint_of(raw_public_bytes(load_public_key(Path(path))))


def _inside_repo(p: Path) -> bool:
    try:
        Path(p).resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def keygen(private_key: Path, public_key: Optional[Path] = None, passphrase: Optional[str] = None) -> dict:
    serialization, Ed25519PrivateKey, _ = _crypto()
    private_key = Path(private_key)
    if _inside_repo(private_key):
        raise SystemExit(f"[ERROR] refusing to write a private key inside the repository ({ROOT}); "
                         f"use a path outside it (ideally a hardware token in production)")
    if private_key.exists():
        raise SystemExit(f"[ERROR] {private_key} exists -- never overwrite a signing key")
    public_key = Path(public_key) if public_key else private_key.with_name(private_key.name + ".pub")
    key = Ed25519PrivateKey.generate()
    enc = (serialization.BestAvailableEncryption(passphrase.encode()) if passphrase
           else serialization.NoEncryption())
    private_key.parent.mkdir(parents=True, exist_ok=True)
    private_key.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, enc))
    try:
        os.chmod(private_key, 0o600)
    except OSError:
        pass
    pub = key.public_key()
    public_key.write_bytes(pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    return {"private_key": str(private_key), "public_key": str(public_key),
            "fingerprint": fingerprint_of(raw_public_bytes(pub)), "encrypted": bool(passphrase)}


def _canonical(head: dict) -> bytes:
    return json.dumps({k: v for k, v in head.items() if k != "signature_b64"},
                      sort_keys=True, separators=(",", ":")).encode()


def head_path_for(log: Path) -> Path:
    return Path(str(log) + ".head.json")


def _chain_facts(log: Path) -> dict:
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    first = json.loads(lines[0])["record_hash"] if lines else None
    last = json.loads(lines[-1])["record_hash"] if lines else None
    return {"record_count": len(lines), "first_record_hash": first, "last_record_hash": last,
            "log_sha256": sha256_file(log)}


def sign_head(log: Path, private_key: Path, passphrase: Optional[str] = None,
              head_path: Optional[Path] = None) -> dict:
    serialization, _, _ = _crypto()
    log = Path(log)
    v = _verify_chain(log)
    if not v.get("valid"):
        raise SystemExit(f"[ERROR] refusing to sign a broken chain: {v}")
    key = serialization.load_pem_private_key(Path(private_key).read_bytes(),
                                             password=passphrase.encode() if passphrase else None)
    pub_raw = raw_public_bytes(key.public_key())
    head = {"format": HEAD_FORMAT, "log_file": log.name, **_chain_facts(log),
            "signed_utc": datetime.now(timezone.utc).isoformat(), "algorithm": "Ed25519",
            "public_key_b64": base64.b64encode(pub_raw).decode(), "public_key_fingerprint": fingerprint_of(pub_raw)}
    head["signature_b64"] = base64.b64encode(key.sign(_canonical(head))).decode()
    hp = Path(head_path) if head_path else head_path_for(log)
    if hp.exists():                        # keep earlier heads: re-signing after appends is legitimate
        hp.rename(hp.with_name(hp.name.replace(".head.json", f".head.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json")))
    hp.write_text(json.dumps(head, indent=2), encoding="utf-8")
    return head


def verify_head(log: Path, head_path: Optional[Path] = None, trusted_fingerprint: Optional[str] = None,
                public_key: Optional[Path] = None) -> dict:
    """-> {"valid": bool, "checks": {name: bool|None}, "reasons": [...]}; valid only if every check passes
    AND the signing key is trusted (matches trusted_fingerprint / public_key)."""
    _, _, Ed25519PublicKey = _crypto()
    log = Path(log)
    hp = Path(head_path) if head_path else head_path_for(log)
    checks, reasons = {}, []
    if not hp.exists():
        return {"valid": False, "checks": {"head_present": False}, "reasons": [f"no signed head {hp}"]}
    head = json.loads(hp.read_text(encoding="utf-8"))
    checks["head_present"] = True
    try:
        pub_raw = base64.b64decode(head["public_key_b64"])
        Ed25519PublicKey.from_public_bytes(pub_raw).verify(base64.b64decode(head["signature_b64"]), _canonical(head))
        checks["signature"] = True
    except Exception as e:                 # InvalidSignature, bad base64, missing field
        checks["signature"] = False
        reasons.append(f"signature invalid ({type(e).__name__})")
        pub_raw = None
    if public_key:
        trusted_fingerprint = public_key_fingerprint(public_key)
    if trusted_fingerprint:
        checks["key_trusted"] = pub_raw is not None and fingerprint_of(pub_raw) == trusted_fingerprint
        if not checks["key_trusted"]:
            reasons.append("signing key does not match the trusted fingerprint")
    else:
        checks["key_trusted"] = None
        reasons.append("no trusted fingerprint / public key given: signature cannot be trusted")
    if not log.exists():
        checks["log_present"] = False
        reasons.append(f"log {log} missing")
        return {"valid": False, "checks": checks, "reasons": reasons, "head": head}
    v = _verify_chain(log)
    checks["chain"] = bool(v.get("valid"))
    if not checks["chain"]:
        reasons.append(f"chain broken: {v.get('reason', v.get('error'))} at record {v.get('broken_at')}")
    facts = _chain_facts(log) if checks["chain"] else {**_chain_facts_safe(log)}
    for k in ("record_count", "last_record_hash", "log_sha256"):
        checks[k] = facts.get(k) == head.get(k)
        if not checks[k]:
            reasons.append(f"{k}: log has {facts.get(k)!r}, signed head has {head.get(k)!r}")
    valid = all(v_ is True for v_ in checks.values())
    return {"valid": valid, "checks": checks, "reasons": reasons, "head": head}


def _chain_facts_safe(log: Path) -> dict:
    try:
        return _chain_facts(log)
    except Exception:
        return {"log_sha256": sha256_file(log)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen")
    k.add_argument("--private_key", required=True, help="path OUTSIDE the repository")
    k.add_argument("--public_key", default=None, help="default: <private_key>.pub")
    k.add_argument("--passphrase_env", default=None, help="encrypt with the passphrase in this environment variable")
    s = sub.add_parser("sign")
    s.add_argument("--log", required=True)
    s.add_argument("--private_key", required=True)
    s.add_argument("--passphrase_env", default=None)
    v = sub.add_parser("verify")
    v.add_argument("--log", required=True)
    v.add_argument("--head", default=None)
    v.add_argument("--trusted_fingerprint", default=None)
    v.add_argument("--public_key", default=None)
    a = ap.parse_args(argv)

    def pw(name):
        if not name:
            return None
        val = os.environ.get(name)
        if not val:
            raise SystemExit(f"[ERROR] environment variable {name} is empty")
        return val

    if a.cmd == "keygen":
        r = keygen(Path(a.private_key), Path(a.public_key) if a.public_key else None, pw(a.passphrase_env))
        print(json.dumps(r, indent=2))
        print("  Record the fingerprint OUTSIDE this system (case file / email). Production: hardware token.")
        return 0
    if a.cmd == "sign":
        h = sign_head(Path(a.log), Path(a.private_key), pw(a.passphrase_env))
        print(f"  signed {h['record_count']} records, last {h['last_record_hash'][:16]}..., "
              f"key {h['public_key_fingerprint']}\n  -> {head_path_for(Path(a.log))}")
        return 0
    r = verify_head(Path(a.log), Path(a.head) if a.head else None, a.trusted_fingerprint,
                    Path(a.public_key) if a.public_key else None)
    print(json.dumps({k_: v_ for k_, v_ in r.items() if k_ != "head"}, indent=2))
    print(f"  {'VALID' if r['valid'] else 'INVALID'}")
    return 0 if r["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
