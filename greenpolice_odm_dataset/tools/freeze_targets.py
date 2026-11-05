#!/usr/bin/env python3
"""
Freeze (pre-register) the EWADS evaluation targets BEFORE results exist.

Writes <targets>.lock.json next to the targets file with the SHA-256 of the targets file and of
ewads_taxonomy.py (the class groups the targets are measured on), the UTC time and who froze it.
08_tiled_eval.py then prints, on every report, one of
    FROZEN                          targets unchanged since the lock
    MODIFIED AFTER FREEZE           the targets file changed after the lock   (do not present as pre-registered)
    TAXONOMY MODIFIED AFTER FREEZE  the class groups changed after the lock
    NOT FROZEN                      no lock (draft targets)

Why bother: if targets can move after you see numbers, any "PASS" is worth nothing to a reviewer
or a court.  Put the lock file (or its hash) somewhere you cannot silently edit -- a git commit, an
email to your supervisor, a timestamping service -- the lock on disk alone proves nothing.

Usage
  python tools/freeze_targets.py --targets targets_v1.json --by "A. Shabbir Ali" --note "before Stage-1 training"
  python tools/freeze_targets.py --targets targets_v1.json --check
  python tools/freeze_targets.py --targets targets_v2.json --by "<name>"      # schema v2 (synthetic-only regime)

The file is validated first (ewads_hier_eval.validate_spec): an invalid file is never frozen.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ewads_taxonomy as tx  # noqa: E402
import ewads_hier_eval as HE  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="targets_v1.json")
    ap.add_argument("--by", default="", help="who froze it")
    ap.add_argument("--note", default="")
    ap.add_argument("--check", action="store_true", help="only report the status")
    a = ap.parse_args()

    p = Path(a.targets)
    spec = json.loads(p.read_text(encoding="utf-8"))   # must be valid JSON
    errs = HE.validate_spec(spec)                       # v1 and v2 (schema_version 2) schemas
    if errs:
        print(f"  {p.name}: INVALID -- not frozen")
        for e in errs:
            print(f"    - {e}")
        sys.exit(2)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    lock_p = p.with_suffix(".lock.json")
    if lock_p.exists():
        lock = json.loads(lock_p.read_text())
        same = lock["targets_sha256"] == sha and lock["taxonomy_sha256"] == tx.taxonomy_sha256()
        print(f"  {p.name}: {'FROZEN' if same else 'MODIFIED AFTER FREEZE'} (locked {lock['frozen_utc']}"
              f"{' by ' + lock['frozen_by'] if lock.get('frozen_by') else ''})")
        if not same:
            print("  -> do not overwrite the lock.  Copy the file to a new version (targets_v2.json), state "
                  "why it changed, and freeze that; report v1 and v2 results side by side.")
        sys.exit(0 if same else 1)
    if a.check:
        print(f"  {p.name}: NOT FROZEN (draft)")
        sys.exit(1)
    lock = {"targets_file": p.name, "targets_sha256": sha, "taxonomy_sha256": tx.taxonomy_sha256(),
            "frozen_utc": datetime.now(timezone.utc).isoformat(), "frozen_by": a.by, "note": a.note}
    lock_p.write_text(json.dumps(lock, indent=2))
    print(f"  froze {p.name}\n    sha256 {sha}\n    lock   {lock_p}")
    print("  Now record that hash somewhere outside this folder (git commit / email / timestamp service).")


if __name__ == "__main__":
    main()
