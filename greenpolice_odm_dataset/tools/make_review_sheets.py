"""
Tile the thumbnails from render_model_thumbnails.py into one numbered contact
sheet per class, plus an index.json mapping tile number -> model file.

Usage:
  python tools/make_review_sheets.py --thumbs_dir ./assets/models/_review
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumbs_dir", default="./assets/models/_review")
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()

    thumbs_dir = Path(args.thumbs_dir)
    index = {}
    for cls_dir in sorted(p for p in thumbs_dir.iterdir() if p.is_dir()):
        thumbs = sorted(cls_dir.glob("*.png"))
        if not thumbs:
            continue
        size = Image.open(thumbs[0]).size[0]
        cols = min(args.cols, len(thumbs))
        rows = -(-len(thumbs) // cols)
        header = 36
        sheet = Image.new("RGB", (cols * size, header + rows * size), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 10), f"{cls_dir.name}  ({len(thumbs)} models)", fill="black")
        index[cls_dir.name] = {}
        for n, t in enumerate(thumbs, start=1):
            x, y = ((n - 1) % cols) * size, header + ((n - 1) // cols) * size
            try:
                with Image.open(t) as im:
                    sheet.paste(im.convert("RGB"), (x, y))
            except Exception:
                # Unreadable thumbnail (e.g. half-written) — mark it, keep going
                draw.rectangle([x, y, x + size - 1, y + size - 1], fill=(120, 120, 120))
                draw.text((x + 40, y + size // 2), "thumbnail unreadable", fill="white")
            draw.rectangle([x, y, x + 28, y + 20], fill="black")
            draw.text((x + 5, y + 4), str(n), fill="white")
            index[cls_dir.name][n] = t.stem
        out = thumbs_dir / f"sheet_{cls_dir.name}.jpg"
        sheet.save(out, quality=90)
        print(f"{out}  ({len(thumbs)} tiles)")

    (thumbs_dir / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
