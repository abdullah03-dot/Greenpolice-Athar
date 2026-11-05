"""
COCO-style detection metrics in plain numpy (no pycocotools, no torch).

Numerically matches pycocotools' COCOeval (101-point interpolated AP, IoU 0.50:0.05:0.95,
greedy per-image matching, ignore regions for size buckets).  Verified against pycocotools in
tests/test_metrics_vs_coco.py.  Used by 08_tiled_eval.py so the numbers reported for the paper are
computed on FULL frames, after tile merging, in native pixels.

Inputs
  gts : {image_id: (cls (n,) int, boxes (n,4) xyxy native px)}
  dets: {image_id: (m,6) [x1 y1 x2 y2 conf cls]}
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
SIZE_BUCKETS = {            # native long side in px
    "8-16": (8, 16), "16-32": (16, 32), "32-64": (32, 64), "64-128": (64, 128), "128+": (128, 1e9),
}


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n,4) x (m,4) -> (n,m)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def long_side(b: np.ndarray) -> np.ndarray:
    return np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) if len(b) else np.zeros(0)


def _match(dt_boxes, dt_conf, gt_boxes, gt_ign, iou_thrs):
    """COCO greedy matching for ONE image & class.  dets must be sorted by conf desc, gts by
    ignore asc.  Returns dtm (T,m) matched-gt index or -1, dt_ign (T,m)."""
    T, m, n = len(iou_thrs), len(dt_boxes), len(gt_boxes)
    dtm = -np.ones((T, m), int)
    dti = np.zeros((T, m), bool)
    if m == 0 or n == 0:
        return dtm, dti
    ious = box_iou(dt_boxes, gt_boxes)
    for ti, t in enumerate(iou_thrs):
        gtm = -np.ones(n, int)
        for d in range(m):
            best, mi = min(t, 1 - 1e-10), -1
            for g in range(n):
                if gtm[g] >= 0:
                    continue
                if mi > -1 and not gt_ign[mi] and gt_ign[g]:
                    break
                if ious[d, g] < best:
                    continue
                best, mi = ious[d, g], g
            if mi == -1:
                continue
            dti[ti, d] = gt_ign[mi]
            dtm[ti, d] = mi
            gtm[mi] = d
    return dtm, dti


def evaluate(gts: Dict[str, Tuple[np.ndarray, np.ndarray]], dets: Dict[str, np.ndarray],
             nc: int, size_range: Optional[Tuple[float, float]] = None,
             image_ids: Optional[Sequence[str]] = None, iou_thrs=IOU_THRS) -> dict:
    """
    Returns {"ap": (nc,T) array (nan = class absent), "n_gt": (nc,), "map": float, "map50": float,
             "map75": float, "per_class_ap": (nc,), "recall_at_iou50": (nc,)}
    size_range=(lo,hi): only GT with native long side in [lo,hi) count; unmatched detections
    outside the range are ignored (exactly COCO's areaRng behaviour, on long side).
    """
    ids = list(image_ids) if image_ids is not None else sorted(gts)
    T = len(iou_thrs)
    ap = np.full((nc, T), np.nan)
    rec_final = np.full((nc, T), np.nan)
    n_gt = np.zeros(nc, int)
    empty_g = (np.zeros(0, int), np.zeros((0, 4)))
    for c in range(nc):
        scores, tps, fps = [], [], []
        npig = 0
        for im in ids:
            gc, gb = gts.get(im, empty_g)
            d = dets.get(im)
            gsel = gc == c
            g = gb[gsel]
            if size_range is None:
                gi = np.zeros(len(g), bool)
            else:
                ls = long_side(g)
                gi = ~((ls >= size_range[0]) & (ls < size_range[1]))
            order = np.argsort(gi, kind="stable")
            g, gi = g[order], gi[order]
            npig += int((~gi).sum())
            if d is None or len(d) == 0:
                continue
            dd = d[d[:, 5] == c]
            if len(dd) == 0:
                continue
            dd = dd[np.argsort(-dd[:, 4], kind="stable")]
            dtm, dti = _match(dd[:, :4], dd[:, 4], g, gi, iou_thrs)
            if size_range is not None:
                ls = long_side(dd[:, :4])
                out = ~((ls >= size_range[0]) & (ls < size_range[1]))
                dti = dti | ((dtm < 0) & out[None, :])
            scores.append(dd[:, 4])
            matched = dtm >= 0
            tps.append(matched & ~dti)
            fps.append(~matched & ~dti)
        n_gt[c] = npig
        if npig == 0:
            continue
        if not scores:
            ap[c] = 0.0
            rec_final[c] = 0.0
            continue
        sc = np.concatenate(scores)
        tp = np.concatenate(tps, 1)
        fp = np.concatenate(fps, 1)
        inds = np.argsort(-sc, kind="mergesort")
        tp, fp = tp[:, inds], fp[:, inds]
        tpc, fpc = np.cumsum(tp, 1).astype(float), np.cumsum(fp, 1).astype(float)
        for ti in range(T):
            rc = tpc[ti] / npig
            pr = tpc[ti] / (tpc[ti] + fpc[ti] + np.spacing(1))
            rec_final[c, ti] = rc[-1] if len(rc) else 0.0
            pr = pr.tolist()
            for i in range(len(pr) - 1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]
            idx = np.searchsorted(rc, REC_THRS, side="left")
            q = np.zeros(len(REC_THRS))
            for ri, pi in enumerate(idx):
                if pi < len(pr):
                    q[ri] = pr[pi]
            ap[c, ti] = q.mean()
    present = n_gt > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # classes absent from the set -> all-NaN rows
        per_class = np.nanmean(ap, 1)
    return {"ap": ap, "n_gt": n_gt,
            "map": float(np.nanmean(per_class[present])) if present.any() else float("nan"),
            "map50": float(np.nanmean(ap[present, 0])) if present.any() else float("nan"),
            "map75": float(np.nanmean(ap[present, 5])) if present.any() else float("nan"),
            "per_class_ap": per_class, "recall_at_iou50": rec_final[:, 0]}


# ───────────────────── operating point (fixed confidence threshold) ──────────────────────
def operating_point(gts, dets, nc: int, conf: float, iou: float = 0.5,
                    image_ids: Optional[Sequence[str]] = None, min_px: Optional[float] = None) -> dict:
    """TP/FP/FN at one confidence threshold, IoU>=iou, class-aware greedy matching by confidence.
    min_px: GT whose native long side < min_px are IGNORED (a detection matching one is neither TP
    nor FP, missing one is not FN) and so are unmatched detections smaller than min_px -- the same
    rule COCO uses for its size ranges.  Used for the 'assessable envelope' numbers."""
    ids = list(image_ids) if image_ids is not None else sorted(gts)
    tp = np.zeros(nc, int); fp = np.zeros(nc, int); fn = np.zeros(nc, int)
    fp_neg_images, n_neg_images, fp_per_image = 0, 0, []
    for im in ids:
        gc, gb = gts.get(im, (np.zeros(0, int), np.zeros((0, 4))))
        d = dets.get(im, np.zeros((0, 6)))
        d = d[d[:, 4] >= conf]
        d = d[np.argsort(-d[:, 4], kind="stable")] if len(d) else d
        used = np.zeros(len(gc), bool)
        gign = (long_side(gb) < min_px) if (min_px and len(gb)) else np.zeros(len(gc), bool)
        dsmall = (long_side(d[:, :4]) < min_px) if (min_px and len(d)) else np.zeros(len(d), bool)
        img_fp = 0
        ious = box_iou(d[:, :4], gb) if len(d) and len(gb) else np.zeros((len(d), len(gb)))
        for i in range(len(d)):
            c = int(d[i, 5])
            cand = np.where((gc == c) & ~used & (ious[i] >= iou))[0] if len(gb) else []
            if len(cand):
                # prefer a real (non-ignored) GT, as COCO does
                real = [j for j in cand if not gign[j]]
                pool = real if real else list(cand)
                j = pool[int(np.argmax(ious[i, pool]))]
                used[j] = True
                if not gign[j]:
                    tp[c] += 1
            elif not dsmall[i]:
                fp[c] += 1
                img_fp += 1
        for j, c in enumerate(gc):
            if not used[j] and not gign[j]:
                fn[int(c)] += 1
        fp_per_image.append(img_fp)
        if len(gc) == 0:
            n_neg_images += 1
            fp_neg_images += img_fp
    T_, F_, N_ = tp.sum(), fp.sum(), fn.sum()
    return {"conf": conf, "tp": tp, "fp": fp, "fn": fn,
            "recall": float(T_ / max(T_ + N_, 1)), "precision": float(T_ / max(T_ + F_, 1)),
            "fp_share": float(F_ / max(T_ + F_, 1)),               # FP / (TP+FP)  (07's "FP rate")
            "fp_per_image": float(np.mean(fp_per_image)) if fp_per_image else 0.0,
            "fp_per_negative_image": float(fp_neg_images / n_neg_images) if n_neg_images else None,
            "n_negative_images": n_neg_images}


def calibrate_threshold(gts, dets, nc: int, target_recall: float = 0.87,
                        thresholds: Sequence[float] = tuple(np.round(np.arange(0.05, 0.96, 0.01), 2)),
                        image_ids=None) -> dict:
    """Highest confidence threshold whose micro-recall (IoU 0.5) >= target; falls back to the
    lowest threshold if the target is unreachable."""
    sweep = [operating_point(gts, dets, nc, float(t), 0.5, image_ids) for t in thresholds]
    ok = [s for s in sweep if s["recall"] >= target_recall]
    best = max(ok, key=lambda s: s["conf"]) if ok else min(sweep, key=lambda s: s["conf"])
    return {"threshold": best["conf"], "reached_target": bool(ok), "operating_point": best,
            "sweep": [{k: s[k] for k in ("conf", "recall", "precision", "fp_share", "fp_per_image")}
                      for s in sweep]}
