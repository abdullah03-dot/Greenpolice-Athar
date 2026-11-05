"""
EWADS evidence fusion -- the offline (RTX 5090) half of the system.

The Jetson must decide in real time, with one INT8 model.  The evidence that goes to court does not
have to: after landing, every evidence-pass frame can be re-analysed with
  1. TEST-TIME AUGMENTATION   the same model on 4-8 exactly-invertible views of the frame
  2. MODEL ENSEMBLE           2-3 independently trained models
  3. WEIGHTED BOX FUSION      averaging the boxes of all those runs (not NMS, which keeps ONE box
                              and throws the rest away)
  4. MULTI-VIEW FUSION        the same ground object seen in several overlapping frames is
                              associated across frames (image registration, GPS as a fallback) and
                              re-scored by how consistently it was detected

What each step can and cannot buy (read before you quote a gain)
  * 1-3 improve LOCALISATION (the 0.75-0.95 IoU part of mAP@0.5:0.95) and calibration, because box
    errors that are independent across runs average out (~1/sqrt(N)).  Errors the model makes the
    same way in every view (a bottle cap that "is" a battery from every angle) do NOT average out.
    Expect a few mAP points, not tens.  Measure on YOUR val set (11_evidence_pipeline --data_dir).
  * 4 suppresses detections that are NOT reproduced in the other frames covering the same ground
    (random false positives, glints, motion-blurred frames), and can carry a confirmed object into
    a frame where it was missed (--fill_missed).  It cannot fix a systematic confusion either.

D4 test-time augmentation is valid HERE because the camera looks straight down (no "up" in the
image) and 03 trains with HorizontalFlip + VerticalFlip + RandomRotate90 -- every D4 view is inside
the training distribution, and axis-aligned boxes map to axis-aligned boxes EXACTLY (unlike free
rotation, which loosens boxes; see 03's header).

Pure numpy + OpenCV (+ scipy for the assignment).  Detections everywhere are (n,6+) arrays
[x1 y1 x2 y2 conf cls ...] in native frame pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import ewads_taxonomy as tx

# ════════════════════════════ 1. D4 test-time augmentation ════════════════════════════
# k in 0..7:  flip = k // 4 (horizontal mirror first), rot = k % 4 (then rot90 counter-clockwise)
D4_NAMES = ["id", "rot90", "rot180", "rot270", "hflip", "hflip_rot90", "vflip", "hflip_rot270"]
TTA_SETS = {"none": [0], "flips": [0, 4, 6, 2], "d4": list(range(8))}


def d4_image(img: np.ndarray, k: int) -> np.ndarray:
    if k // 4:
        img = img[:, ::-1]
    return np.ascontiguousarray(np.rot90(img, k % 4))


def d4_points_forward(pts: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """(n,2) continuous pixel coords in the ORIGINAL frame -> coords in the transformed frame."""
    p = np.asarray(pts, float).copy()
    if k // 4:
        p[:, 0] = w - p[:, 0]
    cw, ch = w, h
    for _ in range(k % 4):                       # np.rot90 (CCW):  (x, y) -> (y, W - x)
        p = np.stack([p[:, 1], cw - p[:, 0]], 1)
        cw, ch = ch, cw
    return p


def d4_points_inverse(pts: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """inverse of d4_points_forward; w, h are the ORIGINAL frame size."""
    p = np.asarray(pts, float).copy()
    r = k % 4
    cw, ch = (h, w) if r % 2 else (w, h)         # size of the transformed frame
    for _ in range(r):                           # undo one CCW rot:  (x', y') -> (H' - y', x')
        p = np.stack([ch - p[:, 1], p[:, 0]], 1)
        cw, ch = ch, cw
    if k // 4:
        p[:, 0] = w - p[:, 0]
    return p


def d4_boxes_inverse(dets: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """dets (n,>=4) detected on d4_image(frame, k) -> boxes in the original frame (exact)."""
    if len(dets) == 0 or k == 0:
        return np.asarray(dets, float).copy()
    d = np.asarray(dets, float).copy()
    a = d4_points_inverse(d[:, 0:2], k, w, h)
    b = d4_points_inverse(d[:, 2:4], k, w, h)
    d[:, 0] = np.minimum(a[:, 0], b[:, 0]); d[:, 2] = np.maximum(a[:, 0], b[:, 0])
    d[:, 1] = np.minimum(a[:, 1], b[:, 1]); d[:, 3] = np.maximum(a[:, 1], b[:, 1])
    return d


@dataclass(frozen=True)
class TTARun:
    k: int = 0               # D4 element
    scale: float = 1.0       # frame resized by this before tiling (bicubic; >1 helps small objects)

    @property
    def name(self) -> str:
        return D4_NAMES[self.k] + ("" if self.scale == 1.0 else f"@x{self.scale:g}")


def tta_plan(tta: str = "flips", scales: Sequence[float] = (1.0,)) -> List[TTARun]:
    ks = TTA_SETS[tta] if isinstance(tta, str) else list(tta)
    return [TTARun(k, float(s)) for s in scales for k in ks]


def run_tta(frame: np.ndarray, detect: Callable[[np.ndarray], np.ndarray],
            plan: Sequence[TTARun]) -> List[np.ndarray]:
    """detect(frame) -> (n,6) native boxes of THAT frame.  Returns one (n,6) array per run, all in
    the ORIGINAL frame's pixel coordinates."""
    h, w = frame.shape[:2]
    out = []
    for r in plan:
        f = frame
        if r.scale != 1.0:
            f = cv2.resize(frame, (int(round(w * r.scale)), int(round(h * r.scale))),
                           interpolation=cv2.INTER_CUBIC if r.scale > 1 else cv2.INTER_AREA)
        sw, sh = f.shape[1], f.shape[0]
        d = np.asarray(detect(d4_image(f, r.k)), float).reshape(-1, 6)
        d = d4_boxes_inverse(d, r.k, sw, sh)
        if r.scale != 1.0:
            d[:, :4] /= np.array([sw / w, sh / h, sw / w, sh / h])
        out.append(d)
    return out


# ════════════════════════════ 2. Weighted box fusion ════════════════════════════
def weighted_boxes_fusion(runs: Sequence[np.ndarray], weights: Optional[Sequence[float]] = None,
                          iou_thr: float = 0.55, skip_thr: float = 0.0, conf_type: str = "avg",
                          one_per_run: bool = True, pad_px: float = 0.0) -> np.ndarray:
    """
    Weighted Box Fusion (Solovyev et al. 2021).  runs: list of (n,6) [x1 y1 x2 y2 conf cls]
    (any pixel scale -- IoU is scale-free).  Returns (m,7) [x1 y1 x2 y2 conf cls n_runs], sorted
    by conf, where n_runs = how many runs (views x models) produced the box -- a number a reviewer
    can understand ("found in 7 of 8 views").

    one_per_run=False reproduces the reference `ensemble_boxes.weighted_boxes_fusion` exactly
    (tests/test_fusion.py).  one_per_run=True (default) lets each run add at most ONE box to a
    cluster: in an e-waste pile, two adjacent batteries detected by the same run are two objects,
    and the reference algorithm can average them into one box between them.
    conf_type: 'avg' (reference default: mean score x min(R, n)/R, i.e. absent runs count as 0)
               or 'max'.
    pad_px: boxes are compared (never fused) after growing each side by this many pixels.  For a
            12 px object, +/-1.5 px of jitter between views drops the IoU of two boxes of the SAME
            object below 0.55, so plain WBF splits it into 2-3 clusters = duplicate boxes = false
            positives (measured in tests/test_fusion.py).  1-2 px of padding absorbs pixel jitter and
            is negligible for objects > 50 px.  0 = reference behaviour.
    """
    R = len(runs)
    if R == 0:
        return np.zeros((0, 7))
    w = np.ones(R) if weights is None else np.asarray(weights, float)
    rows = []
    for t, d in enumerate(runs):
        d = np.asarray(d, float).reshape(-1, 6)
        d = d[d[:, 4] >= skip_thr]
        d = d[((d[:, 2] - d[:, 0]) > 0) & ((d[:, 3] - d[:, 1]) > 0)]
        if len(d):
            rows.append(np.column_stack([d[:, 5], d[:, 4] * w[t], np.full(len(d), w[t]),
                                         np.full(len(d), t), d[:, :4]]))
    if not rows:
        return np.zeros((0, 7))
    allb = np.concatenate(rows)                  # [label, score*w, w, run, x1 y1 x2 y2]
    pad4 = np.array([-pad_px, -pad_px, pad_px, pad_px])
    out = []
    for lab in np.unique(allb[:, 0]):
        b = allb[allb[:, 0] == lab]
        b = b[b[:, 1].argsort()[::-1]]           # same (unstable) order as the reference
        n = len(b)
        fused = np.zeros((n, 4)); fconf = np.zeros(n); k = 0
        members: List[List[int]] = []
        runs_in: List[set] = []
        csum = np.zeros(n); bsum = np.zeros((n, 4)); cmax = np.zeros(n)
        for j in range(n):
            idx = -1
            if k:
                fb = fused[:k] + pad4 if pad_px else fused[:k]
                bj = b[j, 4:8] + pad4 if pad_px else b[j, 4:8]
                xx1 = np.maximum(fb[:, 0], bj[0]); yy1 = np.maximum(fb[:, 1], bj[1])
                xx2 = np.minimum(fb[:, 2], bj[2]); yy2 = np.minimum(fb[:, 3], bj[3])
                inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
                a1 = (fb[:, 2] - fb[:, 0]) * (fb[:, 3] - fb[:, 1])
                a2 = (bj[2] - bj[0]) * (bj[3] - bj[1])
                iou = inter / (a1 + a2 - inter)
                if one_per_run:
                    for ci in range(k):
                        if b[j, 3] in runs_in[ci]:
                            iou[ci] = -1
                bi = int(np.argmax(iou))
                if iou[bi] > iou_thr:
                    idx = bi
            if idx < 0:
                idx = k; k += 1
                members.append([]); runs_in.append(set())
            members[idx].append(j); runs_in[idx].add(b[j, 3])
            csum[idx] += b[j, 1]; bsum[idx] += b[j, 1] * b[j, 4:8]; cmax[idx] = max(cmax[idx], b[j, 1])
            fused[idx] = bsum[idx] / csum[idx]
            fconf[idx] = csum[idx] / len(members[idx]) if conf_type == "avg" else cmax[idx]
        for ci in range(k):
            m = len(members[ci])
            c = fconf[ci] * min(R, m) / w.sum() if conf_type == "avg" else fconf[ci] / w.max()
            out.append([*fused[ci], c, lab, len(runs_in[ci])])
    out = np.asarray(out, float)
    return out[np.argsort(-out[:, 4], kind="stable")]


# ════════════════════════════ 3. Frame-to-frame geometry ════════════════════════════
def _thumb(gray: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    s = min(1.0, max_side / max(gray.shape[:2]))
    if s < 1.0:
        gray = cv2.resize(gray, (int(round(gray.shape[1] * s)), int(round(gray.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)
    return gray, s


def register_pair(prev_gray: np.ndarray, cur_gray: np.ndarray, max_side: int = 1600,
                  min_inliers: int = 25) -> Tuple[Optional[np.ndarray], dict]:
    """
    Homography mapping CURRENT-frame pixels -> PREVIOUS-frame pixels (full resolution), from SIFT
    matches + RANSAC on thumbnails.  Over flat ground seen straight down a homography is exact up to
    lens distortion and relief.  Falls back to phase correlation (translation only) on low-texture
    ground.  Returns (None, info) if neither is trustworthy -- the caller must then NOT link frames
    by image content.
    """
    a, s = _thumb(prev_gray, max_side)          # each frame keeps its OWN thumbnail scale: frames of
    b, sb = _thumb(cur_gray, max_side)          # different sizes (3:2 and 4:3 stills) must not share one
    info = {"method": None, "inliers": 0, "rmse_px": None}
    try:
        det = cv2.SIFT_create(nfeatures=4000)
        ka, da = det.detectAndCompute(a, None)
        kb, db = det.detectAndCompute(b, None)
    except cv2.error:
        ka = kb = da = db = None
    if da is not None and db is not None and len(ka) >= min_inliers and len(kb) >= min_inliers:
        m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
        good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) >= min_inliers:
            pb = np.float32([kb[g.queryIdx].pt for g in good])
            pa = np.float32([ka[g.trainIdx].pt for g in good])
            H, mask = cv2.findHomography(pb, pa, cv2.RANSAC, 3.0)
            if H is not None and mask is not None and int(mask.sum()) >= min_inliers \
                    and mask.mean() >= 0.2 and _sane(H):
                inl = mask.ravel().astype(bool)
                proj = cv2.perspectiveTransform(pb[inl][None], H)[0]
                rmse = float(np.sqrt(((proj - pa[inl]) ** 2).sum(1).mean())) / s
                Hf = np.diag([1 / s, 1 / s, 1.0]) @ H @ np.diag([sb, sb, 1.0])   # cur full -> cur thumb -> prev thumb -> prev full
                if not _sane(Hf):
                    return None, info
                info.update(method="sift_homography", inliers=int(inl.sum()), rmse_px=round(rmse, 2))
                return Hf / Hf[2, 2], info
    # low texture: translation by phase correlation
    if a.shape == b.shape and abs(s - sb) < 1e-9:
        win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), win)
        if resp >= 0.15:
            # sign convention verified in tests/test_fusion.py (crop b = crop a moved by (+dx,+dy)
            # gives phaseCorrelate(a, b) = (-dx, -dy) here, and a point x of b sits at x + (dx,dy) in a).
            Hf = np.array([[1, 0, -dx / s], [0, 1, -dy / s], [0, 0, 1]], float)
            info.update(method="phase_correlation", inliers=0, rmse_px=None, response=round(float(resp), 3))
            return Hf, info
    return None, info


def _sane(H: np.ndarray) -> bool:
    A = H[:2, :2] / H[2, 2]
    det = float(np.linalg.det(A))
    return 0.5 < det < 2.0 and abs(H[2, 0] / H[2, 2]) < 1e-3 and abs(H[2, 1] / H[2, 2]) < 1e-3


EARTH_R = 6371008.8


def latlon_to_en(lat, lon, lat0, lon0) -> Tuple[float, float]:
    """local east/north metres (equirectangular; fine over a few km)."""
    return (math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0)),
            math.radians(lat - lat0) * EARTH_R)


def en_to_latlon(e, n, lat0, lon0) -> Tuple[float, float]:
    return (lat0 + math.degrees(n / EARTH_R),
            lon0 + math.degrees(e / (EARTH_R * math.cos(math.radians(lat0)))))


def pixel_to_en_matrix(meta: dict, wh: Tuple[int, int], origin: Tuple[float, float]) -> Optional[np.ndarray]:
    """
    3x3 affine: frame pixel -> local east/north metres, for a NADIR frame with known GPS, heading and
    GSD.  meta: lat, lon, yaw_deg (camera heading, clockwise from north = direction of the image's
    top edge), gsd_m.  None if any is missing.  Accuracy is the GPS's (non-RTK Phantom: metres).
    """
    if not meta or any(meta.get(k) is None for k in ("lat", "lon", "yaw_deg", "gsd_m")):
        return None
    e0, n0 = latlon_to_en(meta["lat"], meta["lon"], *origin)
    g, p = meta["gsd_m"], math.radians(meta["yaw_deg"])
    c, s = math.cos(p), math.sin(p)
    cx, cy = wh[0] / 2.0, wh[1] / 2.0
    # E = g(dx cos - dy sin),  N = g(-dx sin - dy cos)   with dx, dy = pixel offsets from the centre
    return np.array([[g * c, -g * s, e0 - g * (c * cx - s * cy)],
                     [-g * s, -g * c, n0 - g * (-s * cx - c * cy)],
                     [0, 0, 1]], float)


def warp_boxes(boxes: np.ndarray, H: np.ndarray) -> np.ndarray:
    """(n,4) xyxy -> axis-aligned bounding boxes of the 4 warped corners."""
    if len(boxes) == 0:
        return np.zeros((0, 4))
    b = np.asarray(boxes, float)[:, :4]
    corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], 1).reshape(-1, 1, 2)
    w = cv2.perspectiveTransform(corners.astype(np.float64), H).reshape(-1, 4, 2)
    return np.concatenate([w.min(1), w.max(1)], 1)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


# ════════════════════════════ 4. Multi-view fusion ════════════════════════════
@dataclass
class _Frame:
    fid: str
    wh: Tuple[int, int]
    seg: int
    H: np.ndarray                    # this frame's pixels -> segment mosaic pixels
    unc: float                       # accumulated registration uncertainty (mosaic px) since segment start
    link: dict
    meta: Optional[dict]


@dataclass
class Track:
    tid: int
    seg: int
    obs: List[dict] = field(default_factory=list)   # {f, box, conf, cls, support}
    mosaic_box: Optional[np.ndarray] = None
    # filled by finalize()
    covering: List[int] = field(default_factory=list)
    score: float = 0.0               # cross-view consistency: sum(conf) / frames covering the object
    box_score: float = 0.0           # best per-box score sqrt(conf x consistency) -- same scale as rescored()
    cls: int = -1
    class_votes: Dict[int, float] = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return len(self.obs)


class MultiViewFuser:
    """
    Associate per-frame detections of STATIC ground objects across overlapping frames.

    Geometry:  consecutive frames are linked by image registration (SIFT homography, else phase
    correlation); when that fails and both frames carry GPS + heading + GSD, by GPS (metres of error
    on a non-RTK Phantom, so GPS links are only used where they are unambiguous); otherwise a new
    segment starts and nothing is linked across the gap.

    Association:  Hungarian matching of each new frame's detections to the tracks of its segment,
    projected into the frame.  Gate = IoU >= iou_gate OR centre distance <= registration
    uncertainty + 25% of the box size.  Different fine classes may link inside one taxonomy group
    (tx.GROUPS); across groups only with strong overlap (IoU >= cross_group_iou).

    Score (finalize):  a track's score is  sum(conf of its detections) / (frames whose footprint
    fully contained the object).  A detection that the other views of the same ground did not
    reproduce is diluted; one seen everywhere keeps its confidence.  Per-frame boxes are re-scored
    with sqrt(own conf x track score) (see rescored).  Untracked detections (below track_conf) pass
    through unchanged.
    """

    def __init__(self, iou_gate: float = 0.3, cross_group_iou: float = 0.6, reg_err_px: float = 3.0,
                 gps_err_m: float = 2.0, edge_margin_px: float = 4.0, window: int = 12,
                 track_conf: float = 0.01, min_views: int = 2, use_gps: bool = True,
                 register_max_side: int = 1600):
        self.iou_gate, self.cross_group_iou = iou_gate, cross_group_iou
        self.reg_err_px, self.gps_err_m, self.edge_margin = reg_err_px, gps_err_m, edge_margin_px
        self.window, self.track_conf, self.min_views, self.use_gps = window, track_conf, min_views, use_gps
        self.reg_side = register_max_side
        self.frames: List[_Frame] = []
        self.tracks: List[Track] = []
        self.untracked: Dict[int, np.ndarray] = {}
        self._prev_gray: Optional[np.ndarray] = None
        self._origin: Optional[Tuple[float, float]] = None
        self._seg = -1

    # ── geometry ──
    def _link(self, gray, meta, wh) -> Tuple[Optional[np.ndarray], dict, float]:
        """-> (H cur->prev, info, added uncertainty in px) or (None, info, 0)."""
        prev = self.frames[-1]
        if gray is not None and self._prev_gray is not None:
            H, info = register_pair(self._prev_gray, gray, self.reg_side)
            if H is not None:
                return H, info, self.reg_err_px
        else:
            info = {"method": None}
        if self.use_gps and meta and prev.meta:
            if self._origin is None and meta.get("lat") is not None:
                self._origin = (meta["lat"], meta["lon"])
            if self._origin is not None:
                Ac = pixel_to_en_matrix(meta, wh, self._origin)
                Ap = pixel_to_en_matrix(prev.meta, prev.wh, self._origin)
                if Ac is not None and Ap is not None:
                    H = np.linalg.inv(Ap) @ Ac
                    return H, {"method": "gps", "image_registration": info.get("method")}, \
                        self.gps_err_m / meta["gsd_m"]
        return None, info, 0.0

    def add_frame(self, fid: str, dets: np.ndarray, wh: Tuple[int, int],
                  gray: Optional[np.ndarray] = None, meta: Optional[dict] = None) -> dict:
        dets = np.asarray(dets, float)
        if dets.ndim != 2 or len(dets) == 0:
            dets = np.zeros((0, 7))
        if dets.shape[1] == 6:
            dets = np.column_stack([dets, np.ones(len(dets))])
        fi = len(self.frames)
        if meta and self._origin is None and meta.get("lat") is not None:
            self._origin = (meta["lat"], meta["lon"])
        H_prev, info, du = (None, {"method": "first_frame"}, 0.0) if fi == 0 else self._link(gray, meta, wh)
        if H_prev is None:
            self._seg += 1
            fr = _Frame(fid, wh, self._seg, np.eye(3), 0.0, {**info, "new_segment": True}, meta)
        else:
            p = self.frames[-1]
            fr = _Frame(fid, wh, p.seg, p.H @ H_prev, p.unc + du, info, meta)
        self.frames.append(fr)
        self._prev_gray = gray

        keep = dets[:, 4] >= self.track_conf
        self.untracked[fi] = dets[~keep]
        D = dets[keep]
        cand = [t for t in self.tracks if t.seg == fr.seg and fi - t.obs[-1]["f"] <= self.window]
        matched_t, matched_d = [], []
        if len(D) and cand:
            Hinv = np.linalg.inv(fr.H)
            proj = np.array([warp_boxes(t.mosaic_box[None], Hinv)[0] for t in cand])
            rel = np.array([fr.unc - self.frames[t.obs[-1]["f"]].unc for t in cand])
            iou = _iou(proj, D[:, :4])
            cp = (proj[:, :2] + proj[:, 2:4]) / 2; cd = (D[:, :2] + D[:, 2:4]) / 2
            dist = np.linalg.norm(cp[:, None] - cd[None], axis=2)
            size = np.maximum(D[:, 2] - D[:, 0], D[:, 3] - D[:, 1])[None]
            gate = (iou >= self.iou_gate) | (dist <= rel[:, None] + 0.25 * size)
            tcls = np.array([t.obs[-1]["cls"] for t in cand])
            same_g = np.array([[tx.same_group(a, b) for b in D[:, 5]] for a in tcls])
            gate &= same_g | (iou >= self.cross_group_iou)
            loose = rel[:, None] > 0.5 * size        # GPS-grade uncertainty: demand uniqueness
            if loose.any():
                uniq = (gate.sum(0, keepdims=True) == 1) & (gate.sum(1, keepdims=True) == 1)
                gate &= ~loose | uniq
            # prefer the confident box when a frame holds near-duplicates of one object
            cost = np.where(gate, 1 - iou + dist / (size + rel[:, None] + 1e-9) + 0.5 * (1 - D[None, :, 4]), 1e6)
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(cost)
            for r_, c_ in zip(ri, ci):
                if cost[r_, c_] < 1e5:
                    matched_t.append(r_); matched_d.append(c_)
        for r_, c_ in zip(matched_t, matched_d):
            self._observe(cand[r_], fi, D[c_])
        for j in sorted(set(range(len(D))) - set(matched_d)):
            t = Track(len(self.tracks), fr.seg)
            self.tracks.append(t)
            self._observe(t, fi, D[j])
        return {"frame": fid, "segment": fr.seg, "link": fr.link, "n_tracked": int(len(D)),
                "n_matched": len(matched_d)}

    def _observe(self, t: Track, fi: int, d: np.ndarray):
        t.obs.append({"f": fi, "box": d[:4].copy(), "conf": float(d[4]), "cls": int(d[5]),
                      "support": int(d[6]) if len(d) > 6 else 1})
        t.mosaic_box = warp_boxes(d[None, :4], self.frames[fi].H)[0]

    # ── scoring ──
    def _project_into(self, t: Track, fi: int) -> Tuple[np.ndarray, float]:
        """track box in frame fi, from the observation nearest in time (least drift)."""
        o = min(t.obs, key=lambda o: abs(o["f"] - fi))
        H = np.linalg.inv(self.frames[fi].H) @ self.frames[o["f"]].H
        return warp_boxes(o["box"][None], H)[0], abs(self.frames[fi].unc - self.frames[o["f"]].unc)

    def finalize(self) -> List[Track]:
        for t in self.tracks:
            obs_f = {o["f"] for o in t.obs}
            lo, hi = min(obs_f) - self.window, max(obs_f) + self.window
            cov = set(obs_f)
            for fi, fr in enumerate(self.frames):
                if fi in cov or fr.seg != t.seg or not (lo <= fi <= hi):
                    continue
                b, unc = self._project_into(t, fi)
                m = self.edge_margin + unc
                if b[0] >= m and b[1] >= m and b[2] <= fr.wh[0] - m and b[3] <= fr.wh[1] - m:
                    cov.add(fi)
            t.covering = sorted(cov)
            t.score = sum(o["conf"] for o in t.obs) / len(cov)
            t.box_score = max(math.sqrt(o["conf"] * t.score) for o in t.obs)
            votes: Dict[int, float] = {}
            for o in t.obs:
                votes[o["cls"]] = votes.get(o["cls"], 0.0) + o["conf"]
            t.class_votes = votes
            t.cls = max(votes, key=votes.get)
        return self.tracks

    def rescored(self, fill_missed: bool = False, fill_factor: float = 0.9, mode: str = "geo",
                 class_from: str = "track") -> Dict[str, np.ndarray]:
        """Per-frame detections after multi-view fusion, (n,10):
             [x1 y1 x2 y2 score cls track_id propagated observed_cls observed_conf]
        score  mode 'geo' (default) = sqrt(observed conf x track score): the box keeps its own quality
                     signal (a poor duplicate box of a well-seen object must not inherit the object's
                     score) and gains / loses by how consistently the other views saw the object.
               mode 'track' = the track score for every observation (pure cross-view consistency).
        cls    class_from 'track' = the track's confidence-weighted class vote over all its views (a
               multi-view benefit: one frame's confusion is outvoted); 'observed' = this frame's own.
               observed_cls / observed_conf always carry what the detector said IN THIS FRAME, so the
               audit record never claims a frame-level detection that did not happen.
        fill_missed: add a CONFIRMED track (>= min_views detections) to the covering frames where the
        detector missed it (box projected from the nearest view, propagated=1, observed_cls=-1,
        observed_conf=0), scored fill_factor x what its median observation would get."""
        per = {i: [] for i in range(len(self.frames))}
        for t in self.tracks:
            for o in t.obs:
                sc = math.sqrt(o["conf"] * t.score) if mode == "geo" else t.score
                c = t.cls if class_from == "track" else o["cls"]
                per[o["f"]].append([*o["box"], sc, c, t.tid, 0, o["cls"], o["conf"]])
            if fill_missed and t.n_obs >= self.min_views:
                seen = {o["f"] for o in t.obs}
                med = float(np.median([o["conf"] for o in t.obs]))
                base = math.sqrt(med * t.score) if mode == "geo" else t.score
                for fi in t.covering:
                    if fi not in seen:
                        b, _ = self._project_into(t, fi)
                        per[fi].append([*b, base * fill_factor, t.cls, t.tid, 1, -1, 0.0])
        out = {}
        for fi, fr in enumerate(self.frames):
            u = self.untracked.get(fi, np.zeros((0, 7)))
            u10 = np.column_stack([u[:, :6], -np.ones(len(u)), np.zeros(len(u)), u[:, 5], u[:, 4]]) \
                if len(u) else np.zeros((0, 10))
            a = np.asarray(per[fi], float).reshape(-1, 10)
            allr = np.concatenate([a, u10]) if len(u10) else a
            out[fr.fid] = allr[np.argsort(-allr[:, 4], kind="stable")] if len(allr) else allr
        return out

    def candidate_objects(self, min_score: float, iou: float = 0.3, ios: float = 0.7,
                          mode: str = "geo") -> List[Tuple[Track, List[Track]]]:
        """Tracks whose best box passes min_score ON THE SAME SCALE as rescored(mode) -- so the
        'multiview' operating threshold from a VAL ablation applies directly -- with near-duplicate
        tracks of the SAME ground object merged (a detector can emit two boxes on one object, e.g.
        from two pyramid levels, and each box then forms its own consistent track).  Two tracks are
        one object when, projected into the stronger track's best frame, their boxes overlap by
        IoU >= iou or intersection-over-smaller >= ios, and their classes share a taxonomy group.
        Touching neighbours (two batteries side by side) overlap by ~0 and stay separate.
        Returns [(primary, [absorbed...])] strongest first."""
        key = (lambda t: t.box_score) if mode == "geo" else (lambda t: t.score)
        cand = sorted([t for t in self.tracks if key(t) >= min_score], key=lambda t: -key(t))
        kept: List[Tuple[Track, List[Track]]] = []
        for t in cand:
            dup_of = None
            for k, absorbed in kept:
                if k.seg != t.seg or not tx.same_group(k.cls, t.cls):
                    continue
                kb = max(k.obs, key=lambda o: o["conf"])
                b, _ = self._project_into(t, kb["f"])
                a = kb["box"]
                inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                aa, ab = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
                if inter / (aa + ab - inter + 1e-9) >= iou or inter / (min(aa, ab) + 1e-9) >= ios:
                    dup_of = absorbed
                    break
            if dup_of is None:
                kept.append((t, []))
            else:
                dup_of.append(t)
        return kept

    def ground_position(self, t: Track) -> Optional[dict]:
        """lat/lon of the track from its most confident view (needs GPS + heading + GSD)."""
        if self._origin is None:
            return None
        o = max(t.obs, key=lambda o: o["conf"])
        fr = self.frames[o["f"]]
        A = pixel_to_en_matrix(fr.meta or {}, fr.wh, self._origin)
        if A is None:
            return None
        cx, cy = (o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2
        e, n, _ = A @ np.array([cx, cy, 1.0])
        lat, lon = en_to_latlon(e, n, *self._origin)
        return {"lat": round(lat, 7), "lon": round(lon, 7), "from_frame": fr.fid,
                "accuracy_note": f"aircraft GPS (non-RTK: assume +/- {self.gps_err_m:g} m or worse)"}
