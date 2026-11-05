"""
Checks that Ultralytics' TRAIN-TIME mosaic keeps objects at native scale on the tiled dataset
(so 03's offline --add_mosaic, which halves resolution, is redundant).

Usage: python tests/test_mosaic_scale.py <tiled_dataset_split_dir> [imgsz]
       e.g.  python tests/test_mosaic_scale.py data/tiled/synthetic 768
Builds Ultralytics' RT-DETR training dataset with mosaic=1 and the affine scale jitter 05 uses
(0.25), draws 300 samples and compares object long-side percentiles with the source labels.
"""
import glob
import random
import sys

import numpy as np


def main(split_dir: str, imgsz: int = 768):
    from ultralytics.cfg import get_cfg
    from ultralytics.models.rtdetr.val import RTDETRDataset
    src = []
    for f in glob.glob(f"{split_dir}/labels/*.txt"):
        for line in open(f):
            p = line.split()
            if len(p) >= 5:
                src.append(max(float(p[3]), float(p[4])) * imgsz)
    hyp = get_cfg(overrides=dict(mosaic=1.0, scale=0.25, mixup=0.0, degrees=0, translate=0.1, fliplr=0.5))
    ds = RTDETRDataset(img_path=f"{split_dir}/images", imgsz=imgsz, augment=True, hyp=hyp,
                       data={"names": {i: str(i) for i in range(18)}, "channels": 3}, batch_size=1)
    random.seed(0); np.random.seed(0)
    out = []
    for k in range(300):
        b = ds[k % len(ds)]["bboxes"]
        b = b.numpy() if hasattr(b, "numpy") else np.asarray(b)
        if len(b):
            out += list(np.maximum(b[:, 2], b[:, 3]) * imgsz)
    src, out = np.array(src), np.array(out)
    for q in (5, 10, 25, 50):
        print(f"  p{q:<2}: source {np.percentile(src, q):6.1f} px   after mosaic {np.percentile(out, q):6.1f} px")
    print(f"  share < 8 px: source {(src < 8).mean():.3f}   after mosaic {(out < 8).mean():.3f}")
    ratio = np.percentile(out, 50) / np.percentile(src, 50)
    assert 0.8 < ratio < 1.25, f"mosaic changed object scale (median ratio {ratio:.2f})"
    print(f"OK: median scale ratio {ratio:.2f} (native scale preserved)")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 768)
