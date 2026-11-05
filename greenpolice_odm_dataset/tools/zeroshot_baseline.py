#!/usr/bin/env python3
"""
Zero-shot baseline: Ultralytics YOLO-World with one natural-language prompt per class (ATHAR, 2026-10-08).

Runs YOLO-World through the SAME tiling + merge as 07/08 (ewads_tiling.TiledDetector with
ewads_tiling.ultralytics_predict_fn, tile/overlap from the dataset.yaml `ewads:` block) and writes the
detections in 08's cache format (.npz: one (m,6) [x1 y1 x2 y2 conf cls] array per image stem +
'__timing__'), so 08 scores it exactly like the trained model:

  python tools/zeroshot_baseline.py --data_dir data/splits/main/heldout --dataset_yaml data/augmented/main/dataset.yaml \\
      --out_dir eval/zeroshot_heldout
  python 08_tiled_eval.py --data_dir data/splits/main/heldout --cache eval/zeroshot_heldout/detections.npz \\
      --conf <threshold> --meta data/splits/main/heldout/scene_index.jsonl --targets_dataset synthetic_heldout \\
      --out_dir eval/zeroshot_heldout

Weights: the LARGEST local '*-worldv2.pt' in --weights_dir (default: project root and ~/), or --weights.
Nothing is downloaded unless you pass --download <name> (e.g. yolov8x-worldv2.pt); the file actually used
is hashed (SHA-256) into zeroshot_manifest.json together with the prompts file's SHA-256.
Note: YOLO-World's set_classes() encodes the prompts with a CLIP text model, which Ultralytics fetches on
first use (a one-off download, recorded in the manifest as 'text_encoder_note').

Threshold: the zero-shot model is scored with its OWN threshold calibrated on synthetic validation
(08 --calibrate on main/val, like the trained model) -- never on a test set.
The real test set (data/test) is refused unless --final_evaluation is given (Task 8 only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ewads_taxonomy as tx  # noqa: E402
import ewads_tiling as et    # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompts(path: Path) -> list:
    """-> prompts in ewads_taxonomy.CLASSES order (class id i <-> prompt i)."""
    d = json.loads(path.read_text(encoding="utf-8"))["prompts"]
    missing, extra = set(tx.CLASSES) - set(d), set(d) - set(tx.CLASSES)
    if missing or extra:
        raise SystemExit(f"[ERROR] prompts must cover exactly the 18 classes; missing {sorted(missing)}, "
                         f"unknown {sorted(extra)}")
    return [d[c] for c in tx.CLASSES]


def find_weights(weights: str | None, dirs: list, download: str | None) -> Path:
    if weights:
        p = Path(weights)
        if not p.exists():
            raise SystemExit(f"[ERROR] --weights {p} not found")
        return p
    cands = [p for d in dirs if Path(d).exists() for p in Path(d).glob("*-worldv2.pt")]
    if cands:
        return max(cands, key=lambda p: p.stat().st_size)            # largest = biggest model
    if download:
        from ultralytics import YOLOWorld
        dest = Path(dirs[0])
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {download} (explicit --download) ...")
        YOLOWorld(download)                                            # Ultralytics fetches into CWD
        got = Path(download)
        if not got.exists():
            raise SystemExit(f"[ERROR] download of {download} did not produce the file")
        target = dest / got.name
        if got.resolve() != target.resolve():
            got.replace(target)
        return target
    raise SystemExit("[ERROR] no local *-worldv2.pt found in " + ", ".join(map(str, dirs)) +
                     ". Put one there, pass --weights, or pass --download yolov8x-worldv2.pt")


def build_predict_fn(weights: Path, prompts: list, tile: int, min_conf: float, device):
    """YOLO-World with the class prompts, wrapped exactly like 08 wraps an Ultralytics model."""
    from ultralytics import YOLOWorld
    model = YOLOWorld(str(weights))
    model.set_classes(prompts)
    return et.ultralytics_predict_fn(model, tile, conf=min_conf, device=device)


def tile_from_yaml(p):
    if not p or not Path(p).exists():
        return None, None
    import yaml
    e = (yaml.safe_load(open(p, encoding="utf-8")) or {}).get("ewads", {}) or {}
    return e.get("tile"), e.get("overlap")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="full-resolution images/ (+ labels/ for 08)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--weights_dir", nargs="+", default=[str(ROOT), str(Path.home())])
    ap.add_argument("--download", default=None, help="explicit opt-in, e.g. yolov8x-worldv2.pt")
    ap.add_argument("--prompts", default=str(ROOT / "tools" / "zeroshot_prompts.json"))
    ap.add_argument("--dataset_yaml", default=None, help="tile/overlap from its ewads: block (same as 08)")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--overlap", type=int, default=None)
    ap.add_argument("--max_levels", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min_conf", type=float, default=0.001)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--final_evaluation", action="store_true",
                    help="required to run on a test* folder (the real test set: Task 8 only)")
    a = ap.parse_args(argv)

    data_dir, out = Path(a.data_dir), Path(a.out_dir)
    if any(x.lower().startswith("test") for x in data_dir.resolve().parts[-3:]) and not a.final_evaluation:
        raise SystemExit(f"[ERROR] {data_dir} is a test set: only in final evaluation (--final_evaluation)")
    prompts = load_prompts(Path(a.prompts))
    weights = find_weights(a.weights, a.weights_dir, a.download)
    y_tile, y_ov = tile_from_yaml(a.dataset_yaml)
    tile = a.tile or y_tile or et.DEFAULT_TILE
    overlap = a.overlap or y_ov or et.DEFAULT_OVERLAP
    imgs = sorted(p for p in (data_dir / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    if a.max_images:
        imgs = imgs[:a.max_images]
    import ultralytics
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "baseline": "YOLO-World zero-shot",
                "weights": {"path": str(weights.resolve()), "sha256": sha256_file(weights),
                            "bytes": weights.stat().st_size},
                "prompts_file": str(Path(a.prompts).resolve()), "prompts_sha256": sha256_file(Path(a.prompts)),
                "prompts": dict(zip(tx.CLASSES, prompts)), "class_order": tx.CLASSES,
                "tile": tile, "overlap": overlap, "max_levels": a.max_levels, "min_conf": a.min_conf,
                "inference": "ewads_tiling.TiledDetector + ultralytics_predict_fn (same path as 08 --weights)",
                "ultralytics": ultralytics.__version__,
                "text_encoder_note": "set_classes() encodes prompts with Ultralytics' CLIP text model (fetched on first use)",
                "data_dir": str(data_dir.resolve()), "n_images": len(imgs),
                "tool_sha256": sha256_file(Path(__file__))}
    print(f"  YOLO-World {weights.name} sha256 {manifest['weights']['sha256'][:16]}...  tile {tile} overlap {overlap}")
    fn = build_predict_fn(weights, prompts, tile, a.min_conf, a.device)
    det = et.TiledDetector(fn, tile=tile, overlap=overlap, batch=a.batch, conf=a.min_conf, max_levels=a.max_levels)
    out.mkdir(parents=True, exist_ok=True)
    dets, timing = {}, {}
    for i, p in enumerate(imgs, 1):
        t0 = time.time()
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        dets[p.stem] = det(frame).astype(np.float32)
        timing[p.stem] = time.time() - t0
        print(f"    {i}/{len(imgs)}  {p.name}  tiles {det.last_n_tiles}  dets {len(dets[p.stem])}", end="\r")
    cache = out / "detections.npz"
    np.savez_compressed(cache, __timing__=np.array(timing, dtype=object), **dets)
    manifest["detections"] = {"path": str(cache.resolve()), "sha256": sha256_file(cache)}
    (out / "zeroshot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n  detections -> {cache}\n  manifest   -> {out / 'zeroshot_manifest.json'}")
    print(f"  score with: python 08_tiled_eval.py --data_dir {data_dir} --cache {cache} --conf <val-calibrated> ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
