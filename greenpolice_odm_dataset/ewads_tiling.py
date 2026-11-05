"""
EWADS multi-scale pyramid tiling  --  shared by 03b (dataset tiler), 05/06 (training),
07 (edge inference) and 08 (evaluation).  Pure numpy + OpenCV, no torch.

WHY THIS EXISTS
  A 20 MP drone frame squeezed to 640 px loses 8.5x of its linear resolution: a 24 px
  phone becomes 3 px and a CPU chip disappears.  The renderer now labels objects down to
  12 px, so the detector has to SEE them at native resolution.  We therefore detect on
  fixed-size crops (tiles) cut from the full-resolution frame, plus coarser copies of the
  same frame so that objects bigger than a tile are still seen whole.

THE PYRAMID
  level 0 : native pixels (scale 1.0), cut into overlapping T x T tiles (default 768)
  level k : the frame shrunk by <=2x per level, tiled the same way
  last    : the frame shrunk until it fits in ONE tile (scale = T / long side)

  - An object of native size L is seen at scale s with size L*s.  Objects up to `overlap`
    scaled-px are guaranteed to lie completely inside at least one tile of that level,
    and every object is whole in the last level.  Because consecutive scales differ by
    <=2x, the bands [MIN_OBJ_PX, overlap] of neighbouring levels overlap, so every size
    from 8 px upward is covered by a level that sees it whole and at a sensible size.
  - Cost: ~1.3x the tiles of native tiling alone (level 0 dominates).

TRAINING LABEL RULES (tile_labels)
  keep a box if it is >= MIN_VIS_KEEP visible in the tile, its (unclipped) long side
  >= MIN_OBJ_PX scaled px and its clipped long side >= MIN_BOX_PX.
  Any object that is visible but NOT kept makes the tile "contaminated": it can still
  be used as a positive (it has other labels) but is NEVER used as a clean negative,
  otherwise we would teach the model that a visible object is background.

INFERENCE MERGE (merge_detections)
  1. Detections that touch an interior tile border are dropped on non-final levels
     (they are fragments; the neighbouring tile or a coarser level has the whole object).
  2. Class-aware NMS across tiles and levels.
  3. Class-aware intersection-over-smaller suppression for nested duplicates.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_TILE = 768
DEFAULT_OVERLAP = 128
PAD_VALUE = 114
MIN_OBJ_PX = 8.0          # scaled long side below which an object is out of scope
MIN_BOX_PX = 4.0          # clipped long side below which a fragment is dropped (== renderer)
MIN_VIS_KEEP = 0.25       # fraction of the box that must be inside the tile (SAHI uses 0.1)
BORDER_EPS = 2.0          # px: a detection this close to an interior border is a fragment


# ───────────────────────────── geometry ──────────────────────────────────────
@dataclass(frozen=True)
class TileSpec:
    level: int
    scale: float          # frame scale applied before cropping (1.0 = native pixels)
    x0: int               # crop origin inside the SCALED frame
    y0: int
    tile: int
    sw: int               # scaled frame size
    sh: int
    final: bool           # True on the last level (whole frame in one tile)

    @property
    def vw(self) -> int:  # valid (non-padding) width / height of this crop
        return min(self.tile, self.sw - self.x0)

    @property
    def vh(self) -> int:
        return min(self.tile, self.sh - self.y0)

    @property
    def name(self) -> str:
        return f"L{self.level}_{self.x0}_{self.y0}"


def pyramid_scales(w: int, h: int, tile: int) -> List[float]:
    """[1.0, ...halving..., fit-to-one-tile].  Single entry when the frame already fits."""
    long_side = max(w, h)
    scales = [1.0]
    while long_side * scales[-1] > tile:
        nxt = scales[-1] / 2.0
        if long_side * nxt <= tile:
            nxt = tile / long_side            # in [prev/2, prev): last level fits exactly
        scales.append(nxt)
    return scales


def _starts(n: int, tile: int, overlap: int) -> List[int]:
    """Evenly spread crop origins covering [0, n) with overlap >= `overlap`."""
    if n <= tile:
        return [0]
    stride = tile - overlap
    k = int(math.ceil((n - tile) / stride)) + 1
    return sorted({int(round(i * (n - tile) / (k - 1))) for i in range(k)})


def plan_tiles(w: int, h: int, tile: int = DEFAULT_TILE,
               overlap: int = DEFAULT_OVERLAP) -> List[TileSpec]:
    assert 0 <= overlap < tile
    scales = pyramid_scales(w, h, tile)
    specs: List[TileSpec] = []
    for lvl, s in enumerate(scales):
        sw, sh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        for y0 in _starts(sh, tile, overlap):
            for x0 in _starts(sw, tile, overlap):
                specs.append(TileSpec(lvl, s, x0, y0, tile, sw, sh, lvl == len(scales) - 1))
    return specs


def scale_frame(img: np.ndarray, spec: TileSpec) -> np.ndarray:
    if spec.scale == 1.0:
        return img
    return cv2.resize(img, (spec.sw, spec.sh), interpolation=cv2.INTER_AREA)


def crop_tile(scaled: np.ndarray, spec: TileSpec) -> np.ndarray:
    """T x T crop; bottom/right padded with PAD_VALUE where the frame is smaller than T."""
    t = spec.tile
    out = np.full((t, t) + scaled.shape[2:], PAD_VALUE, dtype=scaled.dtype)
    out[:spec.vh, :spec.vw] = scaled[spec.y0:spec.y0 + spec.vh, spec.x0:spec.x0 + spec.vw]
    return out


def iter_tiles(img: np.ndarray, specs: Sequence[TileSpec]):
    """Yield (spec, tile) while resizing each pyramid level only once."""
    cache_level, cache_img = None, None
    for sp in specs:
        if sp.level != cache_level:
            cache_img, cache_level = scale_frame(img, sp), sp.level
        yield sp, crop_tile(cache_img, sp)


# ───────────────────────────── labels ────────────────────────────────────────
def yolo_to_xyxy(labels: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """labels (N,5) [cls cx cy w h] normalised -> (cls int (N,), boxes native xyxy (N,4))."""
    if len(labels) == 0:
        return np.zeros(0, int), np.zeros((0, 4), float)
    cls = labels[:, 0].astype(int)
    cx, cy, bw, bh = (labels[:, i] for i in range(1, 5))
    b = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
    return cls, b


def read_yolo(path) -> np.ndarray:
    try:
        a = np.loadtxt(str(path), ndmin=2)
    except Exception:
        return np.zeros((0, 5))
    return a.reshape(-1, 5) if a.size else np.zeros((0, 5))


def tile_labels(boxes: np.ndarray, spec: TileSpec, min_obj_px: float = MIN_OBJ_PX,
                min_vis: float = MIN_VIS_KEEP, min_box_px: float = MIN_BOX_PX) -> dict:
    """
    boxes: native-pixel xyxy (N,4).  Returns
      keep   : bool (N,)   label this box in this tile
      clipped: (N,4)       boxes in TILE pixels, clipped to the valid region
      contaminated: bool   a visible object is NOT labelled -> not a clean negative
      vis    : (N,)        fraction of each box inside the tile (0 if not in the tile)
      reason : dict of counts for the dropped ones
    """
    n = len(boxes)
    if n == 0:
        return dict(keep=np.zeros(0, bool), clipped=np.zeros((0, 4)), contaminated=False,
                    vis=np.zeros(0), reason=dict(subscope=0, partial=0, sliver=0))
    b = boxes * spec.scale - np.array([spec.x0, spec.y0, spec.x0, spec.y0], float)
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    c = np.stack([np.clip(b[:, 0], 0, spec.vw), np.clip(b[:, 1], 0, spec.vh),
                  np.clip(b[:, 2], 0, spec.vw), np.clip(b[:, 3], 0, spec.vh)], 1)
    cw, ch = c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]
    in_tile = (cw > 0.5) & (ch > 0.5)
    vis = (cw * ch) / np.maximum(w * h, 1e-9)
    long_full, long_clip = np.maximum(w, h), np.maximum(cw, ch)
    ok_scope = long_full >= min_obj_px
    ok_vis = vis >= min_vis
    ok_size = long_clip >= min_box_px
    keep = in_tile & ok_scope & ok_vis & ok_size
    visible_unlabelled = in_tile & ~keep
    reason = dict(subscope=int((in_tile & ~ok_scope).sum()),
                  partial=int((in_tile & ok_scope & ~ok_vis).sum()),
                  sliver=int((in_tile & ok_scope & ok_vis & ~ok_size).sum()))
    return dict(keep=keep, clipped=c, contaminated=bool(visible_unlabelled.any()),
                vis=np.where(in_tile, vis, 0.0), reason=reason)


def to_yolo_lines(cls: np.ndarray, clipped: np.ndarray, tile: int) -> List[str]:
    out = []
    for k, (x1, y1, x2, y2) in zip(cls, clipped):
        out.append(f"{int(k)} {(x1 + x2) / 2 / tile:.7f} {(y1 + y2) / 2 / tile:.7f} "
                   f"{(x2 - x1) / tile:.7f} {(y2 - y1) / tile:.7f}")
    return out


# ───────────────────────────── merging ───────────────────────────────────────
def nms_classwise(dets: np.ndarray, iou_thr: float = 0.5, ios_thr: float = 0.85,
                  max_pre: int = 4000) -> np.ndarray:
    """dets (N,6) [x1 y1 x2 y2 conf cls].  Greedy class-aware NMS + IoS nested-duplicate removal."""
    if len(dets) == 0:
        return dets
    dets = dets[np.argsort(-dets[:, 4])][:max_pre]
    keep: List[int] = []
    for c in np.unique(dets[:, 5]):
        idx = np.where(dets[:, 5] == c)[0]                 # already sorted by conf
        b = dets[idx, :4]
        area = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
        alive = np.ones(len(idx), bool)
        for i in range(len(idx)):
            if not alive[i]:
                continue
            keep.append(idx[i])
            rest = np.where(alive)[0]
            rest = rest[rest > i]
            if len(rest) == 0:
                continue
            xx1 = np.maximum(b[i, 0], b[rest, 0]); yy1 = np.maximum(b[i, 1], b[rest, 1])
            xx2 = np.minimum(b[i, 2], b[rest, 2]); yy2 = np.minimum(b[i, 3], b[rest, 3])
            inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
            iou = inter / (area[i] + area[rest] - inter + 1e-9)
            ios = inter / (np.minimum(area[i], area[rest]) + 1e-9)
            alive[rest[(iou >= iou_thr) | (ios >= ios_thr)]] = False
    return dets[sorted(keep, key=lambda k: -dets[k, 4])]


def merge_detections(per_tile: Iterable[Tuple[TileSpec, np.ndarray]], iou_thr: float = 0.5,
                     ios_thr: float = 0.85, drop_fragments: bool = True) -> np.ndarray:
    """per_tile: (spec, dets (n,6) in TILE pixels).  Returns (M,6) in NATIVE frame pixels."""
    out = []
    for spec, d in per_tile:
        if d is None or len(d) == 0:
            continue
        d = np.asarray(d, float).copy()
        d[:, [0, 2]] = d[:, [0, 2]].clip(0, spec.vw)
        d[:, [1, 3]] = d[:, [1, 3]].clip(0, spec.vh)
        d = d[((d[:, 2] - d[:, 0]) > 0.5) & ((d[:, 3] - d[:, 1]) > 0.5)]
        if drop_fragments and not spec.final and len(d):
            cut = np.zeros(len(d), bool)
            if spec.x0 > 0:
                cut |= d[:, 0] <= BORDER_EPS
            if spec.y0 > 0:
                cut |= d[:, 1] <= BORDER_EPS
            if spec.x0 + spec.vw < spec.sw:
                cut |= d[:, 2] >= spec.vw - BORDER_EPS
            if spec.y0 + spec.vh < spec.sh:
                cut |= d[:, 3] >= spec.vh - BORDER_EPS
            d = d[~cut]
        if len(d) == 0:
            continue
        d[:, [0, 2]] = (d[:, [0, 2]] + spec.x0) / spec.scale
        d[:, [1, 3]] = (d[:, [1, 3]] + spec.y0) / spec.scale
        out.append(d)
    if not out:
        return np.zeros((0, 6))
    return nms_classwise(np.concatenate(out), iou_thr, ios_thr)


# ───────────────────────────── tiled detector ────────────────────────────────
PredictFn = Callable[[List[np.ndarray]], List[np.ndarray]]


class TiledDetector:
    """
    Wrap any tile-level detector.
      predict_fn(list of T x T x 3 RGB uint8 tiles) -> list of (n,6) arrays
                 [x1 y1 x2 y2 conf cls] in tile pixels
    Call with an RGB or BGR frame (set `bgr=True` if BGR; tiles are converted for you).
    """

    def __init__(self, predict_fn: PredictFn, tile: int = DEFAULT_TILE,
                 overlap: int = DEFAULT_OVERLAP, batch: int = 8, conf: float = 0.001,
                 nms_iou: float = 0.5, ios: float = 0.85, max_levels: Optional[int] = None):
        self.predict_fn, self.tile, self.overlap, self.batch = predict_fn, tile, overlap, batch
        self.conf, self.nms_iou, self.ios, self.max_levels = conf, nms_iou, ios, max_levels
        self.last_n_tiles = 0

    def plan(self, w: int, h: int) -> List[TileSpec]:
        specs = plan_tiles(w, h, self.tile, self.overlap)
        if self.max_levels:   # cheaper mode: keep the first N levels + the final one
            keep = {s.level for s in specs if s.level < self.max_levels} | \
                   {s.level for s in specs if s.final}
            specs = [s for s in specs if s.level in keep]
        return specs

    def __call__(self, frame: np.ndarray, bgr: bool = True) -> np.ndarray:
        h, w = frame.shape[:2]
        specs = self.plan(w, h)
        self.last_n_tiles = len(specs)
        results: List[Tuple[TileSpec, np.ndarray]] = []
        buf_specs, buf_tiles = [], []

        def flush():
            if not buf_tiles:
                return
            preds = self.predict_fn(list(buf_tiles))
            for sp, d in zip(buf_specs, preds):
                d = np.asarray(d, float).reshape(-1, 6)
                results.append((sp, d[d[:, 4] >= self.conf]))
            buf_specs.clear(); buf_tiles.clear()

        for sp, t in iter_tiles(frame, specs):
            buf_specs.append(sp)
            buf_tiles.append(cv2.cvtColor(t, cv2.COLOR_BGR2RGB) if bgr else t)
            if len(buf_tiles) >= self.batch:
                flush()
        flush()
        return merge_detections(results, self.nms_iou, self.ios)


# ───────────────────────────── model adapters ────────────────────────────────
def ultralytics_predict_fn(model, imgsz: int, conf: float = 0.001, device=None) -> PredictFn:
    """Adapter for an Ultralytics RTDETR / YOLO model (expects RGB tiles)."""
    def fn(tiles: List[np.ndarray]) -> List[np.ndarray]:
        # Ultralytics treats numpy input as BGR; flip so the model sees the RGB it trained on.
        res = model.predict([np.ascontiguousarray(t[:, :, ::-1]) for t in tiles], imgsz=imgsz, conf=conf,
                            iou=1.0, verbose=False, device=device, max_det=300)
        return [r.boxes.data.cpu().numpy() if r.boxes is not None else np.zeros((0, 6)) for r in res]
    return fn


def _xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def parse_raw_output(out: np.ndarray, imgsz: int, conf: float = 0.001, fmt: str = "auto") -> np.ndarray:
    """
    Decode a raw ONNX/TensorRT output for ONE tile to (n,6) [x1 y1 x2 y2 conf cls] in tile px.

      (1,N,6)    end-to-end heads.  Ultralytics' YOLOv10 writes  x1 y1 x2 y2 conf cls  but the
                 current RT-DETR export writes  cx cy w h conf cls  (pixels).  Both are (N,6),
                 so the box format is decided by `fmt`: "xyxy" | "xywh" | "auto".
                 "auto" checks ALL N raw rows (before any confidence filtering): a genuine xyxy
                 head never has x2 < x1 or y2 < y1, while 300 xywh rows always contain some.
      (1,N,4+nc) older RT-DETR export: cx cy w h + per-class sigmoid scores (normalised 0..1)
      (1,4+nc,N) classic YOLOv8-style head with no NMS inside (cx cy w h px + scores)
    Verified against Ultralytics' own predict() in tests/test_onnx_parity.py.
    """
    p = np.asarray(out)
    p = p[0] if p.ndim == 3 else p
    classic = False
    if p.shape[-1] != 6 and p.shape[0] < p.shape[1]:        # (4+nc, N) -> (N, 4+nc)
        p, classic = p.T, True
    if p.shape[-1] == 6:
        p = p.astype(float)
        live = p[np.abs(p[:, :4]).sum(1) > 0]               # ignore all-zero padding rows
        if fmt == "auto":
            ok = ((live[:, 2] >= live[:, 0]) & (live[:, 3] >= live[:, 1])).mean() if len(live) else 1.0
            fmt = "xyxy" if ok >= 0.999 else "xywh"
        d = p[p[:, 4] >= conf].copy()
        if len(d) and d[:, :4].max() <= 2.0:                # normalised variant
            d[:, :4] *= imgsz
        if fmt == "xywh" and len(d):
            d[:, :4] = _xywh_to_xyxy(d[:, :4])
        return d
    boxes, scores = p[:, :4].astype(float), p[:, 4:]
    cls = scores.argmax(1)
    cf = scores[np.arange(len(scores)), cls]
    m = cf >= conf
    boxes, cls, cf = boxes[m], cls[m], cf[m]
    if len(boxes) and boxes.max() <= 2.0:
        boxes = boxes * imgsz
    d = np.concatenate([_xywh_to_xyxy(boxes) if len(boxes) else boxes.reshape(0, 4),
                        cf[:, None], cls[:, None]], 1)
    if classic and len(d):                                  # head has no NMS: do it per tile
        d = nms_classwise(d, 0.7, 1.01, max_pre=2000)
    return d


# ───────────────────────────── repeat-factor sampling ────────────────────────
def repeat_factors(image_classes: Dict[str, set], t: float = 0.1, cap: float = 8.0
                   ) -> Tuple[Dict[str, float], Dict[int, float]]:
    """LVIS repeat-factor sampling.  f_c = fraction of images containing c;
    r_c = max(1, sqrt(t / f_c)); r_i = max_c in i r_c.  Returns ({image: r_i}, {class: r_c})."""
    n = max(1, len(image_classes))
    cnt: Dict[int, int] = {}
    for cs in image_classes.values():
        for c in cs:
            cnt[c] = cnt.get(c, 0) + 1
    r_c = {c: min(cap, max(1.0, math.sqrt(t / (k / n)))) for c, k in cnt.items()}
    r_i = {im: max([r_c[c] for c in cs], default=1.0) for im, cs in image_classes.items()}
    return r_i, r_c


def stochastic_repeats(r_i: Dict[str, float], seed: int = 0) -> List[str]:
    """Image list where each image appears floor(r) times + 1 more with prob frac(r)."""
    rng = random.Random(seed)
    out: List[str] = []
    for im in sorted(r_i):
        r = r_i[im]
        out += [im] * (int(r) + (1 if rng.random() < r - int(r) else 0))
    return out
