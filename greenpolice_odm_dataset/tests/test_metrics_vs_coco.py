"""Check ewads_metrics.evaluate against pycocotools COCOeval on random data (run: python tests/test_metrics_vs_coco.py)."""
import sys, io, contextlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ewads_metrics as M
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def make(seed, n_img=60, nc=6):
    rng = np.random.default_rng(seed)
    gts, dets = {}, {}
    for i in range(n_img):
        n = rng.integers(0, 12)
        c = rng.integers(0, nc, n); w = np.exp(rng.uniform(np.log(8), np.log(400), n)); h = w * rng.uniform(.4, 1.6, n)
        x = rng.uniform(0, 3000, n); y = rng.uniform(0, 2000, n)
        b = np.stack([x, y, x + w, y + h], 1); gts[f"im{i}"] = (c, b)
        d = []
        for k in range(n):                                     # noisy true positives, some missed
            if rng.random() < .8:
                j = rng.normal(0, .06, 4) * np.array([w[k], h[k], w[k], h[k]])
                d.append(np.r_[b[k] + j, rng.uniform(.2, 1), c[k] if rng.random() < .9 else rng.integers(0, nc)])
        for _ in range(rng.integers(0, 15)):                   # false positives
            ww = np.exp(rng.uniform(np.log(8), np.log(300))); xx, yy = rng.uniform(0, 3000), rng.uniform(0, 2000)
            d.append([xx, yy, xx + ww, yy + ww * .8, rng.uniform(.01, .7), rng.integers(0, nc)])
        dets[f"im{i}"] = np.array(d).reshape(-1, 6)
    return gts, dets, nc

def coco_eval(gts, dets, nc, area=None):
    imgs = [{"id": i, "width": 4000, "height": 3000} for i in range(len(gts))]
    ids = {k: i for i, k in enumerate(sorted(gts))}
    anns, k = [], 1
    for im, (c, b) in gts.items():
        for cc, bb in zip(c, b):
            anns.append({"id": k, "image_id": ids[im], "category_id": int(cc) + 1, "bbox": [bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]],
                         "area": float((bb[2]-bb[0])*(bb[3]-bb[1])), "iscrowd": 0}); k += 1
    gt = COCO(); gt.dataset = {"images": imgs, "annotations": anns, "categories": [{"id": i+1} for i in range(nc)]}
    with contextlib.redirect_stdout(io.StringIO()): gt.createIndex()
    res = [{"image_id": ids[im], "category_id": int(d[5]) + 1, "bbox": [d[0], d[1], d[2]-d[0], d[3]-d[1]], "score": float(d[4])}
           for im, dd in dets.items() for d in dd]
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(res)
        E = COCOeval(gt, dt, "bbox"); E.params.maxDets = [1, 10, 100000]; E.params.areaRng = [[0, 1e10]]; E.params.areaRngLbl = ["all"]
        E.evaluate(); E.accumulate()
    p = E.eval["precision"][:, :, :, 0, -1]          # T,R,K
    ap_k = np.array([np.mean(p[:, :, k][p[:, :, k] > -1]) if (p[:, :, k] > -1).any() else np.nan for k in range(nc)])
    return float(np.nanmean(ap_k)), float(np.nanmean(p[0][p[0] > -1])), ap_k

worst = 0
for seed in range(4):
    gts, dets, nc = make(seed)
    r = M.evaluate(gts, dets, nc)
    cm, c50, ck = coco_eval(gts, dets, nc)
    d = max(abs(r["map"] - cm), abs(r["map50"] - c50), np.nanmax(np.abs(r["per_class_ap"] - ck)))
    worst = max(worst, d)
    print(f"seed {seed}: mine mAP {r['map']:.5f} / COCO {cm:.5f} | mAP50 {r['map50']:.5f} / {c50:.5f} | max per-class diff {np.nanmax(np.abs(r['per_class_ap']-ck)):.2e}")
assert worst < 1e-6, worst
print("OK: matches pycocotools to", f"{worst:.1e}")
