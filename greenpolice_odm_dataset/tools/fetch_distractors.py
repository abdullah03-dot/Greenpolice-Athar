"""
Fetch NON-e-waste clutter models from Objaverse-LVIS for use as unlabelled
distractors in 02_blenderproc_synthetic_v2.py.

Optional: the renderer already scatters procedural clutter (rocks, shrubs,
sticks, tyres, boxes, bags, bottles, rope/hose, blocks). Real scanned/modelled
junk — furniture, mattresses, buckets, shoes, traffic cones — adds shapes that
procedural primitives can't, and is exactly what the detector confuses with
e-waste on real dump sites (the FP < 0.05 target).

LVIS category names are resolved at runtime against
objaverse.load_lvis_annotations() (exact name first, then a token match), and
the mapping actually used is printed and saved — verify it before relying on
it. Categories that are electronics are refused, so a distractor can never be
an unlabelled e-waste item.

Output:
  assets/distractors/<category>/<uid>.glb  +  <uid>.json {"real_size_m": ...}
  assets/distractors/resolved_categories.json

Usage:
  python tools/fetch_distractors.py --per_category 6
  python tools/fetch_distractors.py --list_only            # show the resolved mapping only
Then review thumbnails like any model round:
  blenderproc run tools/render_model_thumbnails.py --models_dir assets/distractors \\
      --out_dir assets/distractors/_review
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

# wanted category -> typical longest dimension (m)
DISTRACTORS = {
    "tire": 0.65, "wheel": 0.6, "bottle": 0.3, "water_bottle": 0.25, "beer_bottle": 0.25,
    "wine_bottle": 0.3, "can": 0.12, "beer_can": 0.12, "plastic_bag": 0.5, "shopping_bag": 0.45,
    "trash_can": 0.8, "bucket": 0.35, "pail": 0.35, "crate": 0.5, "carton": 0.4,
    "traffic_cone": 0.7, "chair": 0.9, "folding_chair": 0.9, "stool": 0.5, "bench": 1.6,
    "table": 1.2, "mattress": 1.9, "pillow": 0.6, "cushion": 0.5, "blanket": 1.4, "towel": 0.8,
    "suitcase": 0.7, "backpack": 0.5, "shoe": 0.3, "boot": 0.35, "barrel": 0.9, "log": 1.6,
    "brick": 0.22, "flowerpot": 0.35, "watering_can": 0.45, "basket": 0.45, "hose": 2.0,
    "ball": 0.22, "soccer_ball": 0.22, "bicycle": 1.7, "helmet": 0.3, "cardboard": 0.5,
    "doormat": 0.7, "rag_doll": 0.4, "teddy_bear": 0.4, "jean": 0.9, "jacket": 0.8,
}
# Never accept these as clutter — they are e-waste (see CLASS_LVIS_CATEGORIES in 02b)
ELECTRONIC_TOKENS = {"television", "monitor", "computer", "keyboard", "printer", "telephone",
                     "cellular", "radio", "toaster", "kettle", "fan", "hair_dryer", "battery",
                     "refrigerator", "washer", "microwave", "lightbulb", "lamp", "laptop",
                     "router", "remote_control", "calculator", "speaker", "stereo", "camera",
                     "projector", "vacuum", "blender", "heater", "air_conditioner", "clock"}


def resolve(wanted: dict, lvis_keys: list) -> dict:
    """wanted name -> actual LVIS key (exact, else unique token match), or None."""
    keys = set(lvis_keys)
    out = {}
    for w in wanted:
        if w in keys:
            out[w] = w
            continue
        hits = [k for k in lvis_keys if w in k.replace("(", "").replace(")", "").split("_")
                or k.startswith(w + "_")]
        out[w] = sorted(hits, key=len)[0] if hits else None
    return {w: k for w, k in out.items()
            if k is None or not any(t in k for t in ELECTRONIC_TOKENS)}


def validate_mesh(glb_path: str) -> bool:
    try:
        import trimesh
        s = trimesh.load(glb_path)
        if isinstance(s, trimesh.Scene):
            return any(hasattr(g, "faces") and len(g.faces) > 10 for g in s.geometry.values())
        return len(getattr(s, "faces", [])) > 10
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./assets/distractors")
    ap.add_argument("--per_category", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--list_only", action="store_true")
    args = ap.parse_args()
    try:
        import objaverse
    except ImportError:
        sys.exit("pip install objaverse trimesh")

    out = Path(args.out_dir)
    lvis = objaverse.load_lvis_annotations()
    mapping = resolve(DISTRACTORS, sorted(lvis))
    print(f"{'wanted':<16} -> LVIS category (n objects)")
    for w, k in mapping.items():
        print(f"  {w:<16} -> {k or '-- not found --'}" + (f" ({len(lvis[k])})" if k else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_categories.json").write_text(json.dumps(mapping, indent=2))
    if args.list_only:
        return

    rng = random.Random(args.seed)
    seen = {p.stem for p in out.glob("*/*.glb")}
    total = 0
    for w, k in mapping.items():
        if not k:
            continue
        cat_dir = out / w
        have = len(list(cat_dir.glob("*.glb"))) if cat_dir.exists() else 0
        pool = [u for u in lvis[k] if u not in seen]
        rng.shuffle(pool)
        while have < args.per_category and pool:
            uid = pool.pop()
            seen.add(uid)
            try:   # serial, one UID at a time: pooled downloads hang on truncated files (HANDOFF)
                got = objaverse.load_objects(uids=[uid], download_processes=1)
            except Exception as e:
                print(f"    ✗ {uid[:12]} download failed ({type(e).__name__})")
                continue
            path = got.get(uid)
            if not path or not validate_mesh(path):
                continue
            cat_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, cat_dir / f"{uid}.glb")
            (cat_dir / f"{uid}.json").write_text(json.dumps(
                {"real_size_m": DISTRACTORS[w], "lvis_category": k, "role": "distractor"}))
            have += 1
            total += 1
            print(f"    ✓ {w:<14} {uid}")
    print(f"\n{total} distractor models added under {out}. Review thumbnails before rendering.")


if __name__ == "__main__":
    main()
