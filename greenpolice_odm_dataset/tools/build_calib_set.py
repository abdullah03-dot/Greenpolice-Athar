#!/usr/bin/env python3
"""
Build an INT8 calibration set for TensorRT from the TILED dataset (03b output).

INT8 post-training quantisation picks each tensor's range from what the calibration images
produce.  If those images are not what the drone sees, the ranges are wrong and accuracy drops
silently.  So the set must look like deployment inputs:
  * TILES at the model's input size (never full frames -- the engine never sees one)
  * the same mix of pyramid levels the tiler produces (mostly native level 0)
  * every class present, small ones included (their activations are the easiest to clip)
  * ~10% empty-ground tiles (most tiles in the field are background)
  * real tiles first when you have them (synthetic statistics differ: that is the domain gap)
  * NEVER test images (that would leak the test set into the deployed model)

Writes <out>/images, <out>/labels and <out>/calib.yaml, which Ultralytics' export reads:
  model.export(format="engine", quantize=8, data="<out>/calib.yaml", imgsz=768, ...)
(10_jetson_build_engine.py does this on the Jetson.)

Usage
  python tools/build_calib_set.py --tiled_dir data/tiled --splits real synthetic --n 800 --out data/calib
"""
import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ewads_survey as S  # noqa: E402  (class order)


def load_index(tiled: Path, splits):
    rows = []
    idx = tiled / "tile_index.jsonl"
    if idx.exists():
        for line in idx.read_text().splitlines():
            r = json.loads(line)
            if r.get("split") in splits:
                rows.append(r)
    else:                                   # no index: scan folders, parse level from the name
        for sp in splits:
            for p in sorted((tiled / sp / "images").glob("*.jpg")):
                lvl = int(p.stem.split("__L")[1].split("_")[0]) if "__L" in p.stem else 0
                rows.append({"tile": p.stem, "split": sp, "level": lvl})
    out = []
    for r in rows:
        img = tiled / r["split"] / "images" / f"{r['tile']}.jpg"
        lbl = tiled / r["split"] / "labels" / f"{r['tile']}.txt"
        if not img.exists():
            continue
        cls = []
        if lbl.exists():
            cls = [int(float(l.split()[0])) for l in lbl.read_text().splitlines() if l.strip()]
        out.append({**r, "img": img, "lbl": lbl, "classes": cls})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiled_dir", required=True, help="03b output root (has <split>/images, tile_index.jsonl)")
    ap.add_argument("--splits", nargs="+", default=["real", "synthetic"],
                    help="splits to draw from, in priority order (never your test split)")
    ap.add_argument("--n", type=int, default=800, help="calibration tiles (TensorRT suggests 500-1000)")
    ap.add_argument("--per_class", type=int, default=20, help="guarantee this many tiles per class")
    ap.add_argument("--neg_frac", type=float, default=0.10)
    ap.add_argument("--real_frac", type=float, default=0.5,
                    help="max share from the FIRST split (usually real), if it has enough tiles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if any(s.lower().startswith("test") for s in a.splits):
        sys.exit("[ERROR] refusing to calibrate on a test split (leakage)")
    rng = random.Random(a.seed)
    tiled = Path(a.tiled_dir)
    pool = load_index(tiled, set(a.splits))
    if not pool:
        sys.exit(f"[ERROR] no tiles found in {tiled} for splits {a.splits}")
    rng.shuffle(pool)
    first = [r for r in pool if r["split"] == a.splits[0]]
    pos = [r for r in pool if r["classes"]]
    neg = [r for r in pool if not r["classes"]]

    chosen, seen = [], set()

    def take(r):
        if r["tile"] + r["split"] not in seen:
            seen.add(r["tile"] + r["split"])
            chosen.append(r)

    # 1) every class: up to per_class tiles, preferring the first (real) split
    by_cls = defaultdict(list)
    for r in sorted(pos, key=lambda r: r["split"] != a.splits[0]):
        for c in set(r["classes"]):
            by_cls[c].append(r)
    for c in range(len(S.EWASTE_CLASSES)):
        for r in by_cls.get(c, [])[:a.per_class]:
            take(r)
    # 2) real share
    n_first = int(a.real_frac * a.n)
    for r in first:
        if sum(1 for x in chosen if x["split"] == a.splits[0]) >= n_first or len(chosen) >= a.n:
            break
        take(r)
    # 3) negatives
    for r in neg[:int(a.neg_frac * a.n)]:
        take(r)
    # 4) fill with the natural mix (keeps the pyramid-level proportions of the tiler)
    for r in pool:
        if len(chosen) >= a.n:
            break
        take(r)

    out = Path(a.out)
    for sub in ("images", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for r in chosen:
        name = f"{r['split']}__{r['tile']}"
        shutil.copy2(r["img"], out / "images" / f"{name}.jpg")
        if r["lbl"].exists():
            shutil.copy2(r["lbl"], out / "labels" / f"{name}.txt")
        else:
            (out / "labels" / f"{name}.txt").touch()
    names = {i: n for i, n in enumerate(S.EWASTE_CLASSES)}
    yaml_txt = (f"# INT8 calibration set built by tools/build_calib_set.py ({len(chosen)} tiles)\n"
                f"path: {out.resolve().as_posix()}\ntrain: images\nval: images\n"
                f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in names.items()))
    (out / "calib.yaml").write_text(yaml_txt)

    cls_cov = Counter(c for r in chosen for c in set(r["classes"]))
    lv = Counter(r.get("level", 0) for r in chosen)
    sp = Counter(r["split"] for r in chosen)
    print(f"  calibration tiles: {len(chosen)}  ({sum(1 for r in chosen if not r['classes'])} empty)")
    print(f"  by split : {dict(sp)}")
    print(f"  by level : {dict(sorted(lv.items()))}")
    missing = [S.EWASTE_CLASSES[c] for c in range(len(S.EWASTE_CLASSES)) if not cls_cov.get(c)]
    print(f"  classes  : {len(cls_cov)}/{len(S.EWASTE_CLASSES)} present" +
          (f" -- MISSING: {', '.join(missing)}" if missing else ""))
    if len(chosen) < 300:
        print("  ⚠  fewer than 300 tiles: INT8 ranges will be noisy")
    if a.splits[0] != "real" or not first:
        print("  ⚠  no real tiles used: calibration reflects synthetic statistics only")
    print(f"  wrote {out / 'calib.yaml'}")


if __name__ == "__main__":
    main()
