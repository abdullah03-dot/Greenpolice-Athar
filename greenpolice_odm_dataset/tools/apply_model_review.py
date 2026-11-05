"""
Apply a manual model-review round to assets/models/.

Each round is a JSON decisions file under tools/reviews/:
  {
    "index":      "assets/models/_review_v2/index.json",   # tile -> file stem
    "source_dir": "assets/models",                          # where the tiles' models live
    "reject": {"battery": [3, 7], ...},                     # -> assets/models/_rejected/<class>/
    "move":   [["flat_screen", 2, "crt_monitor"], ...]      # -> assets/models/<dest>/
  }
Tile numbers refer to the sheet_<class>.jpg next to the index. Nothing is
deleted. Every action is appended to assets/models/_review/review_log.jsonl.

Usage:
  python tools/apply_model_review.py tools/reviews/round2_current.json            # dry run
  python tools/apply_model_review.py tools/reviews/round2_current.json --apply
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

MODELS_DIR = Path("assets/models")
LOG = MODELS_DIR / "_review" / "review_log.jsonl"


def find_model(cls_dir: Path, stem: str) -> Path:
    matches = [p for p in cls_dir.glob(f"{stem}.*") if p.suffix.lower() in {".glb", ".obj", ".fbx"}]
    if len(matches) != 1:
        raise FileNotFoundError(f"{cls_dir}/{stem}.* -> {matches}")
    return matches[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decisions")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    d = json.loads(Path(args.decisions).read_text())
    index = json.loads(Path(d["index"]).read_text()) if d.get("index") else {}
    source = Path(d["source_dir"])

    actions = []
    # reject by model file stem (e.g. Objaverse uid) instead of sheet tile -- added 2026-10-08 for the
    # licence round; the class is found by searching the class folders (must be exactly one match)
    for stem in d.get("reject_stems", []):
        hits = [p for p in source.glob(f"*/{stem}.*") if p.suffix.lower() in {".glb", ".obj", ".fbx"}
                and not p.parent.name.startswith("_")]
        if len(hits) != 1:
            raise SystemExit(f"[ERROR] reject_stems: {stem} matched {[str(h) for h in hits]} (need exactly one)")
        actions.append(("reject", hits[0].parent.name, -1, hits[0], MODELS_DIR / "_rejected" / hits[0].parent.name))
    for cls, tiles in d.get("reject", {}).items():
        for t in tiles:
            src = find_model(source / cls, index[cls][str(t)])
            actions.append(("reject", cls, t, src, MODELS_DIR / "_rejected" / cls))
    for cls, t, dst_cls in d.get("move", []):
        src = find_model(source / cls, index[cls][str(t)])
        actions.append(("move", cls, t, src, MODELS_DIR / dst_cls))

    rejected = {(c, t) for k, c, t, _, _ in actions if k == "reject"}
    moved = {(c, t) for k, c, t, _, _ in actions if k == "move"}
    assert not rejected & moved, f"tile both rejected and moved: {rejected & moved}"

    for kind, cls, t, src, dst_dir in actions:
        print(f"{kind:6s} {cls}#{t:<3d} {src.name} -> {dst_dir}")
        if args.apply:
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if dst.exists():   # same Objaverse UID fetched into two classes
                src.unlink()
            else:
                shutil.move(str(src), str(dst))
            with open(LOG, "a") as f:
                f.write(json.dumps({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "round": Path(args.decisions).stem,
                    "action": kind, "class": cls, "tile": None if t == -1 else t,
                    "file": src.name, "dest": str(dst_dir),
                    **({"reason": d["reason"]} if d.get("reason") else {}),
                }) + "\n")

    print(f"\n{len(actions)} actions ({'applied' if args.apply else 'dry run'})")


if __name__ == "__main__":
    main()
