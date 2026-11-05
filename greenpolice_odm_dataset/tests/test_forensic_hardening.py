"""
Tests for forensic hardening, Task 3 (added 2026-10-08).

  1. capture time: EXIF DateTimeOriginal (+SubSec, +OffsetTimeOriginal) -> UTC; no offset -> local only
     ('zone unknown'), operator-asserted offset labelled as such; XMP CreateDate; DJI SRT per-frame
     datetime (newer firmware) vs Phantom 4 Pro SRT (none); nothing -> 'unavailable'.
  2. audit records (07 ForensicAuditLogger): capture_time_*, model_sha256 and timestamp_utc (processing
     time) present in every record.
  3. verify on resume: an edited existing log is REFUSED on open; a valid one resumes and stays valid;
     verify-only open still reports; a truncated last line is reported, not a crash.
  4. tools/sign_chain_head.py: keygen refuses repo paths / existing files; sign + verify with a trusted
     fingerprint; untrusted / wrong key rejected; tail truncation, full rebuild and appends after
     signing detected by the head while verify_chain alone passes; broken chain not signed; earlier
     head kept on re-sign; passphrase-encrypted key.
  5. 11: every record carries capture_time_* and the ensemble members' model_sha256; the run manifest
     records the signing public-key fingerprint.

Run: python tests/test_forensic_hardening.py
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import ewads_survey as es          # noqa: E402
import sign_chain_head as SCH      # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B07 = _load("bench07", "07_edge_benchmark.py")


def _jpeg_with_exif(path, dto=None, subsec=None, offset=None, xmp=None):
    im = Image.new("RGB", (64, 48), (90, 90, 90))
    exif = Image.Exif()
    if dto:
        ifd = exif.get_ifd(0x8769)
        ifd[0x9003] = dto
        if subsec:
            ifd[0x9291] = subsec
        if offset:
            ifd[0x9011] = offset
    im.save(path, exif=exif.tobytes() if dto else b"")
    if xmp:
        with open(path, "ab") as f:          # XMP text near the file start is what the parser scans
            f.write(xmp.encode())


def test_capture_time():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _jpeg_with_exif(td / "a.jpg", "2026:10:01 10:20:30", "250", "+04:00")
        r = es.capture_time_from_image(td / "a.jpg")
        assert r["capture_time_utc"] == "2026-10-01T06:20:30.250000+00:00", r
        assert r["capture_time_source"] == "exif DateTimeOriginal+OffsetTimeOriginal", r
        _jpeg_with_exif(td / "b.jpg", "2026:10:01 10:20:30")
        r = es.capture_time_from_image(td / "b.jpg")
        assert r["capture_time_utc"] is None and r["capture_time_local"] == "2026-10-01T10:20:30"
        assert "zone unknown" in r["capture_time_source"], r
        r = es.capture_time_from_image(td / "b.jpg", assume_offset="+04:00")
        assert r["capture_time_utc"] == "2026-10-01T06:20:30+00:00" and "ASSERTED by operator" in r["capture_time_source"]
        _jpeg_with_exif(td / "c.jpg", xmp='<x:xmpmeta xmp:CreateDate="2026-10-01T06:20:30Z"/>')
        r = es.capture_time_from_image(td / "c.jpg")
        assert r["capture_time_utc"] == "2026-10-01T06:20:30+00:00" and r["capture_time_source"] == "xmp CreateDate", r
        _jpeg_with_exif(td / "d.jpg")
        assert es.capture_time_from_image(td / "d.jpg") == es.CAPTURE_TIME_UNAVAILABLE
        (td / "new.srt").write_text("1\n00:00:00,000 --> 00:00:00,033\n<font>FrameCnt: 1, DiffTime: 33ms\n"
                                    "2026-10-01 10:20:30,123,456\n[latitude: 25.2] [longitude: 55.3] "
                                    "[rel_alt: 12.4 abs_alt: 30.1]</font>\n\n", encoding="utf-8")
        rec = es.parse_dji_srt(td / "new.srt")[0]
        r = es.capture_time_from_srt(rec)
        assert r["capture_time_local"] == "2026-10-01T10:20:30.123000" and r["capture_time_utc"] is None, r
        assert es.capture_time_from_srt(rec, "+04:00")["capture_time_utc"] == "2026-10-01T06:20:30.123000+00:00"
        (td / "p4p.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nF/2.8, SS 320, ISO 100, EV 0, "
                                    "GPS (55.3, 25.2, 19), D 25.88m, H 12.40m, H.S 3.20m/s, V.S 0.00m/s\n\n",
                                    encoding="utf-8")
        assert es.capture_time_from_srt(es.parse_dji_srt(td / "p4p.srt")[0]) == es.CAPTURE_TIME_UNAVAILABLE
    print("  ok  capture time: EXIF(+offset) -> UTC, local-only kept as local, asserted offset labelled, XMP, SRT")


def _write_log(path, n=6, model_sha="ab" * 32, capture=None):
    lg = B07.ForensicAuditLogger(path, "test-model", model_sha256=model_sha)
    for i in range(n):
        lg.log_detection(f"img{i}.jpg", hashlib.sha256(str(i).encode()).hexdigest(),
                         [{"class_id": 3, "class_name": "battery", "confidence": 0.9, "bbox_xyxy": [1, 2, 3, 4], "tier": 1}],
                         None, 12.0, 0.3, {"note": i}, capture_time=capture)
    return lg


def test_records_and_resume():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        log = td / "audit.jsonl"
        _write_log(log, capture={"capture_time_utc": "2026-10-01T06:20:30+00:00",
                                 "capture_time_local": "2026-10-01T10:20:30",
                                 "capture_time_source": "exif DateTimeOriginal+OffsetTimeOriginal"})
        recs = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        for r in recs:
            assert r["model_sha256"] == "ab" * 32 and r["capture_time_utc"] == "2026-10-01T06:20:30+00:00"
            assert r["capture_time_source"] and r["timestamp_utc"] and "capture_time_local" in r
        # default (no capture time given) -> explicit 'unavailable'
        log2 = td / "audit2.jsonl"
        _write_log(log2, n=1)
        r = json.loads(log2.read_text(encoding="utf-8"))
        assert r["capture_time_utc"] is None and r["capture_time_source"] == "unavailable"
        # valid log resumes and stays valid
        lg = B07.ForensicAuditLogger(log, "test-model", model_sha256="ab" * 32)
        lg.log_detection("img9.jpg", "00" * 32, [], None, None, 0.3)
        assert lg.verify_chain()["valid"] and lg.verify_chain()["records"] == 7
        # edited log is refused on open
        lines = log.read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[2]); rec["altitude_m"] = 99.0; lines[2] = json.dumps(rec)
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            B07.ForensicAuditLogger(log, "test-model")
            raise AssertionError("broken chain must be refused on open")
        except RuntimeError as e:
            assert "REFUSING to append" in str(e) and "record 2" in str(e)
        n_before = len(log.read_text(encoding="utf-8").splitlines())
        v = B07.ForensicAuditLogger(log, "verify-only", verify_on_open=False).verify_chain()
        assert not v["valid"] and v["broken_at"] == 2
        assert len(log.read_text(encoding="utf-8").splitlines()) == n_before      # nothing appended
        # truncated last line (power cut mid-write) -> reported, not a crash
        log3 = td / "audit3.jsonl"
        _write_log(log3, n=3)
        log3.write_text(log3.read_text(encoding="utf-8")[:-40], encoding="utf-8")
        v = B07.ForensicAuditLogger(log3, "verify-only", verify_on_open=False).verify_chain()
        assert not v["valid"] and v["broken_at"] == 2 and "unparseable" in v["reason"], v
    print("  ok  records carry capture_time_*, model_sha256, timestamp_utc; broken log refused on open; truncation reported")


def test_signed_head():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for bad in (ROOT / "should_not_exist_key.pem",):
            try:
                SCH.keygen(bad)
                raise AssertionError("key inside repo must be refused")
            except SystemExit as e:
                assert "inside the repository" in str(e)
            assert not bad.exists()
        k = SCH.keygen(td / "keys" / "k.pem")
        fpr = k["fingerprint"]
        try:
            SCH.keygen(td / "keys" / "k.pem")
            raise AssertionError("existing key must not be overwritten")
        except SystemExit as e:
            assert "never overwrite" in str(e)
        other = SCH.keygen(td / "keys" / "other.pem")
        assert SCH.public_key_fingerprint(k["public_key"]) == fpr

        def fresh(name, n=8):
            p = td / name
            _write_log(p, n=n)
            SCH.sign_head(p, Path(k["private_key"]))
            return p

        log = fresh("ok.jsonl")
        r = SCH.verify_head(log, trusted_fingerprint=fpr)
        assert r["valid"], r
        assert not SCH.verify_head(log)["valid"]                                         # untrusted
        assert not SCH.verify_head(log, trusted_fingerprint=other["fingerprint"])["valid"]
        assert SCH.verify_head(log, public_key=Path(k["public_key"]))["valid"]
        # tampered signature
        h = json.loads(SCH.head_path_for(log).read_text()); h["record_count"] += 1
        SCH.head_path_for(log).write_text(json.dumps(h))
        assert SCH.verify_head(log, trusted_fingerprint=fpr)["checks"]["signature"] is False

        # tail truncation: chain alone still valid, head catches it
        log = fresh("trunc.jsonl")
        lines = log.read_text(encoding="utf-8").splitlines()
        log.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8")
        assert B07.ForensicAuditLogger(log, "v", verify_on_open=False).verify_chain()["valid"]
        r = SCH.verify_head(log, trusted_fingerprint=fpr)
        assert not r["valid"] and r["checks"]["record_count"] is False, r

        # full rebuild after an edit: chain alone valid, head catches it
        log = fresh("rebuild.jsonl")
        recs = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        recs[3]["altitude_m"] = 77.0
        prev = "GENESIS"
        for rr in recs:
            rr.pop("record_hash"); rr["prev_hash"] = prev
            rr["record_hash"] = hashlib.sha256(json.dumps(rr, sort_keys=True, default=str).encode()).hexdigest()
            prev = rr["record_hash"]
        log.write_text("\n".join(json.dumps(rr) for rr in recs) + "\n", encoding="utf-8")
        assert B07.ForensicAuditLogger(log, "v", verify_on_open=False).verify_chain()["valid"]
        r = SCH.verify_head(log, trusted_fingerprint=fpr)
        assert not r["valid"] and r["checks"]["last_record_hash"] is False

        # legitimate append after signing -> head stale until re-signed; old head kept
        log = fresh("append.jsonl")
        B07.ForensicAuditLogger(log, "test-model").log_detection("x.jpg", "11" * 32, [], None, None, 0.3)
        assert not SCH.verify_head(log, trusted_fingerprint=fpr)["valid"]
        SCH.sign_head(log, Path(k["private_key"]))
        assert SCH.verify_head(log, trusted_fingerprint=fpr)["valid"]
        assert len(list(td.glob("append.jsonl.head.*.json"))) == 1

        # broken chain is never signed
        log = td / "broken.jsonl"
        _write_log(log, n=4)
        lines = log.read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[1]); rec["n_detections"] = 5; lines[1] = json.dumps(rec)
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            SCH.sign_head(log, Path(k["private_key"]))
            raise AssertionError("broken chain must not be signed")
        except SystemExit as e:
            assert "broken chain" in str(e)
        assert not SCH.head_path_for(log).exists()

        # passphrase-encrypted key
        enc = SCH.keygen(td / "keys" / "enc.pem", passphrase="correct horse")
        log = td / "enc.jsonl"
        _write_log(log, n=2)
        SCH.sign_head(log, Path(enc["private_key"]), passphrase="correct horse")
        assert SCH.verify_head(log, trusted_fingerprint=enc["fingerprint"])["valid"]
        # CLI exit codes
        assert SCH.main(["verify", "--log", str(log), "--trusted_fingerprint", enc["fingerprint"]]) == 0
        assert SCH.main(["verify", "--log", str(log)]) == 1
    print("  ok  signed head: repo/overwrite refused, trust pinned, truncation/rebuild/append caught, broken chain not signed")


def test_11_records_and_manifest():
    import test_fusion as TF
    ev = TF._load_11()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        TF.make_flight(td / "seq", n_frames=6)
        for n in ("a.pt", "b.pt"):
            (td / n).write_bytes(n.encode())
        k = SCH.keygen(td / "keys" / "k.pem")
        out = td / "ev"
        sys.argv = ["11", "--weights", str(td / "a.pt"), str(td / "b.pt"), "--images", str(td / "seq" / "images"),
                    "--tta", "none", "--out_dir", str(out), "--tile", "512", "--report_conf", "0.2",
                    "--signing_pubkey", k["public_key"]]
        ev.main()
        rd = out / (out / "LATEST_RUN.txt").read_text()
        man = json.loads((rd / "run_manifest.json").read_text())
        assert man["signing_public_key_fingerprint"] == k["fingerprint"]
        want = [hashlib.sha256(n.encode()).hexdigest() for n in ("a.pt", "b.pt")]
        recs = [json.loads(l) for l in (rd / "evidence_audit.jsonl").read_text(encoding="utf-8").splitlines()]
        assert recs and all(r["model_sha256"] == want for r in recs)
        assert all("capture_time_utc" in r and r["capture_time_source"] for r in recs)
        SCH.sign_head(rd / "evidence_audit.jsonl", Path(k["private_key"]))
        assert SCH.verify_head(rd / "evidence_audit.jsonl", trusted_fingerprint=man["signing_public_key_fingerprint"])["valid"]
    print("  ok  11: records carry capture time + ensemble model_sha256; manifest holds the signing-key fingerprint")


if __name__ == "__main__":
    test_capture_time()
    test_records_and_resume()
    test_signed_head()
    test_11_records_and_manifest()
    print("\nforensic-hardening tests passed")
