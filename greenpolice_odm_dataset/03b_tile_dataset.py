"""
EWADS -- Step 3b: cut full-resolution frames into native-resolution training tiles
==================================================================================
WHY: 03_augmentation_pipeline.py used to squeeze every frame to 640 px.  At that scale a
24 px phone is 3 px and a CPU chip vanishes, so the small classes the renderer now labels
were unlearnable.  This step keeps native pixels: it cuts each frame into overlapping
768 px tiles (plus coarser copies so big objects are still seen whole) -- see
ewads_tiling.py for the pyramid and label rules.

PIPELINE POSITION (new):
    BlenderProc / real frames -> 03b (tile) -> 04 AdaIN (on tiles) -> 03 augment (tiles)
                              -> 05 train (imgsz = tile) -> 08 full-frame evaluation

INPUT  (same layout 03 already expects):
    <input_dir>/<split>/images/*.jpg|png   native resolution
    <input_dir>/<split>/labels/*.txt       YOLO normalised  (empty file = true negative)
OUTPUT:
    <output_dir>/<split>/images/<stem>__L<level>_<x0>_<y0>.jpg
    <output_dir>/<split>/labels/<same>.txt
    <output_dir>/tile_index.jsonl          tile -> source frame, level, scale, origin (provenance)
    <output_dir>/tiling_manifest.json      settings + per-class coverage report (read by 03/05/06)

Never tile your TEST set: evaluate it full-frame with 08_tiled_eval.py.
Split by SOURCE FRAME (this script keeps each frame's tiles in its own split), otherwise
overlapping tiles of one scene leak between train and val.

Usage:
    python 03b_tile_dataset.py --input_dir data/raw --output_dir data/tiled --dry_run   # sizes only
    python 03b_tile_dataset.py --input_dir data/raw --output_dir data/tiled
"""
import argparse
import json
import shutil
import os
import sys
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import ewads_tiling as et

CLASSES = ["crt_monitor", "flat_screen", "circuit_board", "battery", "cable_bundle", "single_cable",
           "mobile_phone", "keyboard", "printer", "small_appliance", "large_appliance", "cpu_chip",
           "hard_drive", "power_supply", "ewaste_pile", "plastic_casing", "metal_component", "lighting"]
IMG_EXT = {".jpg", ".jpeg", ".png"}


def _image_size(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size            # (w, h)


def process_one(job: dict) -> dict:
    """Tile one source frame.  Runs in a worker process."""
    img_path, lbl_path = Path(job["img"]), Path(job["lbl"])
    split, out = job["split"], Path(job["out"])
    tile, overlap = job["tile"], job["overlap"]
    rng = np.random.default_rng(zlib.crc32(img_path.stem.encode()) ^ job["seed"])
    keep_neg = job["neg_keep_val"] if split == "val" else job["neg_keep"]

    measure = job["dry"] and job.get("measure", False)
    if job["dry"] and not measure:
        w, h = _image_size(img_path)
        img = None
    else:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return {"error": f"unreadable: {img_path}"}
        h, w = img.shape[:2]

    cls, boxes = et.yolo_to_xyxy(et.read_yolo(lbl_path), w, h)
    specs = et.plan_tiles(w, h, tile, overlap)
    n = len(cls)
    whole = np.zeros(n, bool)                       # object seen >=90% inside some tile as a label
    st = dict(frames=1, pos=Counter(), neg=Counter(), rejected=Counter(), drop=Counter(),
              labels_out=Counter(), labels_src=Counter(int(c) for c in cls), whole=Counter(),
              frames_negative=int(n == 0), tiny=Counter())
    rows, writes = [], []
    last_level, scaled = None, None

    for sp in specs:
        r = et.tile_labels(boxes, sp)
        k = r["keep"]
        for kk, v in r["reason"].items():
            st["drop"][kk] += v
        if k.any():
            kind = "pos"
        elif not r["contaminated"]:
            kind = "neg" if rng.random() < keep_neg else None
        else:
            kind = None
            st["rejected"][sp.level] += 1           # only unlabelled/ambiguous content: unusable
        if k.any():
            whole |= k & (r["vis"] >= 0.9)
        if kind is None:
            continue
        st[kind][sp.level] += 1
        name = f"{img_path.stem}__{sp.name}"
        lines = et.to_yolo_lines(cls[k], r["clipped"][k], tile)
        for c in cls[k]:
            st["labels_out"][int(c)] += 1
        rows.append({"tile": name, "split": split, "source": img_path.name, "level": sp.level,
                     "scale": round(sp.scale, 6), "x0": sp.x0, "y0": sp.y0, "src_w": w, "src_h": h,
                     "n_labels": int(k.sum()), "kind": kind})
        if not job["dry"] or measure:
            if sp.level != last_level:
                scaled, last_level = et.scale_frame(img, sp), sp.level
            writes.append((name, et.crop_tile(scaled, sp), lines))

    if measure:                                     # dry run: encode in memory, write nothing
        for _n, t, _l in writes:
            ok, buf = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, job["quality"]])
            st["bytes"] = st.get("bytes", 0) + (len(buf) if ok else 0)
            st["bytes_tiles"] = st.get("bytes_tiles", 0) + 1
        writes = []
    if not job["dry"]:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        for name, t, lines in writes:
            fp = out / split / "images" / f"{name}.jpg"
            cv2.imwrite(str(fp), t, [cv2.IMWRITE_JPEG_QUALITY, job["quality"]])
            st["bytes"] = st.get("bytes", 0) + fp.stat().st_size
            st["bytes_tiles"] = st.get("bytes_tiles", 0) + 1
            (out / split / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    long_side = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) if n else np.zeros(0)
    for i in range(n):
        if whole[i]:
            st["whole"][int(cls[i])] += 1
        elif long_side[i] < et.MIN_OBJ_PX:
            st["tiny"][int(cls[i])] += 1
    return {"stats": st, "rows": rows}


def merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, Counter):
            a.setdefault(k, Counter()).update(v)
        else:
            a[k] = a.get(k, 0) + v
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", default="./data/raw")
    ap.add_argument("--output_dir", default="./data/tiled")
    ap.add_argument("--splits", nargs="+", default=["synthetic", "real", "val"])
    ap.add_argument("--tile", type=int, default=et.DEFAULT_TILE, help="tile side in px (= training imgsz)")
    ap.add_argument("--overlap", type=int, default=et.DEFAULT_OVERLAP,
                    help="overlap in px; objects up to this size are always whole in some tile")
    ap.add_argument("--neg_keep", type=float, default=0.12,
                    help="fraction of CLEAN empty tiles kept (hard-negative background, clutter)")
    ap.add_argument("--neg_keep_val", type=float, default=0.05)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=0, help="0 = all cores - 1")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dry_run", action="store_true", help="plan only: counts and a MEASURED size estimate")
    ap.add_argument("--measure_frames", type=int, default=24,
                    help="dry run: frames whose tiles are JPEG-encoded in memory to measure bytes/tile")
    ap.add_argument("--n_styles", type=int, default=1, help="for the estimate: 04 AdaIN copies per synthetic tile")
    ap.add_argument("--n_aug", type=int, default=3, help="for the estimate: 03 --n_aug_per_image")
    args = ap.parse_args()

    inp, out = Path(args.input_dir), Path(args.output_dir)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    jobs = []
    for split in args.splits:
        # <split>/images, or one level of sub-folders: 02 --environment all writes
        # synthetic/<environment>/images (desert / european_urban / european_rural)
        roots = [inp / split] if (inp / split / "images").is_dir() else \
            sorted(r for r in (inp / split).glob("*") if (r / "images").is_dir())
        if not roots:
            print(f"  [SKIP] {inp / split}/images (or {inp / split}/*/images) not found")
            continue
        if roots != [inp / split]:
            print(f"  {split}: reading {len(roots)} sub-folders: {', '.join(r.name for r in roots)}")
        seen_stems = {}
        for root, p in ((r, p) for r in roots for p in sorted((r / "images").iterdir())):
            if p.suffix.lower() not in IMG_EXT:
                continue
            if p.stem in seen_stems:   # tiles are named by stem: a clash would overwrite tiles
                sys.exit(f"[ERROR] duplicate image name {p.stem!r} in {seen_stems[p.stem]} and {root}")
            seen_stems[p.stem] = root
            jobs.append(dict(img=str(p), lbl=str(root / "labels" / (p.stem + ".txt")),
                             split=split, out=str(out), tile=args.tile, overlap=args.overlap,
                             neg_keep=args.neg_keep, neg_keep_val=args.neg_keep_val,
                             quality=args.quality, seed=args.seed, dry=args.dry_run))
    if not jobs:
        sys.exit("No images found.")
    if args.dry_run:                                # encode a spread-out sample to measure bytes/tile
        step = max(1, len(jobs) // max(1, args.measure_frames))
        for j in jobs[::step][:args.measure_frames]:
            j["measure"] = True
    print(f"\n  {len(jobs)} frames | tile {args.tile} overlap {args.overlap} | workers {workers}"
          f"{' | DRY RUN' if args.dry_run else ''}")

    per_split: dict = defaultdict(dict)
    index_f = None
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        index_f = open(out / "tile_index.jsonl", "w")
    errors = []
    with ProcessPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(process_one, jobs, chunksize=4), 1):
            if "error" in res:
                errors.append(res["error"]); continue
            merge(per_split[jobs[i - 1]["split"]], res["stats"])
            if index_f:
                for r in res["rows"]:
                    index_f.write(json.dumps(r) + "\n")
            if i % 200 == 0 or i == len(jobs):
                print(f"    {i}/{len(jobs)} frames", end="\r")
    if index_f:
        index_f.close()
    print()

    manifest = {"tile": args.tile, "overlap": args.overlap, "min_obj_px": et.MIN_OBJ_PX,
                "min_vis_keep": et.MIN_VIS_KEEP, "neg_keep": args.neg_keep, "splits": {}}
    est_mb = 0.0
    nbytes = sum(s.get("bytes", 0) for s in per_split.values())
    ntile_m = sum(s.get("bytes_tiles", 0) for s in per_split.values())
    mb_per_tile = (nbytes / ntile_m / 2 ** 20) if ntile_m else 0.22
    tiles_by_split = {}
    for split, s in per_split.items():
        pos, neg = sum(s["pos"].values()), sum(s["neg"].values())
        tiles_by_split[split] = pos + neg
        est_mb += (pos + neg) * mb_per_tile
        print(f"\n  [{split}] frames {s['frames']} (true-negative frames {s['frames_negative']})")
        print(f"    tiles  positive {pos}  clean-negative {neg}  rejected(ambiguous) {sum(s['rejected'].values())}")
        print(f"    by level   pos {dict(sorted(s['pos'].items()))}  neg {dict(sorted(s['neg'].items()))}")
        print(f"    label drops (per tile-instance)  {dict(s['drop'])}")
        print(f"    {'class':<18}{'source':>9}{'in tiles':>10}{'whole>=90%':>12}{'<8px':>7}")
        cov = {}
        for c, name in enumerate(CLASSES):
            src, wh = s["labels_src"].get(c, 0), s["whole"].get(c, 0)
            if src:
                print(f"    {name:<18}{src:>9}{s['labels_out'].get(c, 0):>10}{wh / src:>11.1%}{s['tiny'].get(c, 0):>7}")
            cov[name] = {"source": src, "labels_in_tiles": s["labels_out"].get(c, 0),
                         "whole_in_some_tile": wh, "below_min_px": s["tiny"].get(c, 0)}
        manifest["splits"][split] = {
            "frames": s["frames"], "pos_tiles": pos, "neg_tiles": neg,
            "rejected_tiles": sum(s["rejected"].values()), "per_class": cov}
    src = (f"measured on {ntile_m} tiles" if ntile_m else "assumed") + ("" if args.dry_run else ", as written")
    print(f"\n  Tiles: {sum(tiles_by_split.values())} at {mb_per_tile * 1024:.0f} KB/tile ({src}, JPEG q{args.quality})")
    print(f"  Disk, this step (03b): ~{est_mb / 1024:.2f} GB")
    syn = tiles_by_split.get("synthetic", 0)
    train = sum(v for k, v in tiles_by_split.items() if k != "val")
    styled = syn * args.n_styles * mb_per_tile
    aug = (train + syn * (args.n_styles - 1)) * (args.n_aug + 1) * mb_per_tile
    print(f"  Downstream if every stage keeps its output: 04 AdaIN (+{args.n_styles} per synthetic tile) "
          f"~{styled / 1024:.2f} GB, 03 augmentation (x{args.n_aug + 1} train tiles + val) "
          f"~{(aug + tiles_by_split.get('val', 0) * mb_per_tile) / 1024:.2f} GB")
    total = est_mb + styled + aug + tiles_by_split.get("val", 0) * mb_per_tile
    try:
        free = shutil.disk_usage(out if out.exists() else out.parent if out.parent.exists() else Path(".")).free
        print(f"  Total ~{total / 1024:.2f} GB; free on the output drive: {free / 2 ** 30:.1f} GB"
              + ("  <-- NOT ENOUGH" if free < total * 2 ** 20 * 1.15 else ""))
    except OSError:
        print(f"  Total ~{total / 1024:.1f} GB")
    manifest["disk_estimate_gb"] = {"tiles": est_mb / 1024, "adain": styled / 1024, "augmented": aug / 1024,
                                    "kb_per_tile": mb_per_tile * 1024, "measured_tiles": ntile_m}
    if errors:
        print(f"  {len(errors)} unreadable frames, first: {errors[0]}")
    if not args.dry_run:
        (out / "tiling_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  Wrote {out / 'tiling_manifest.json'} and tile_index.jsonl")
        print("  Next: python 04_adain_transfer.py transfer ... (on tiles)  then  "
              f"python 03_augmentation_pipeline.py --input_dir {out}")


if __name__ == "__main__":
    main()
