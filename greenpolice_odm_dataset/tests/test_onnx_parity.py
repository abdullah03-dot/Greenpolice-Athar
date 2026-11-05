"""Parity between Ultralytics predict() and the raw-ONNX path used on the Jetson
(ewads_tiling.parse_raw_output).  Needs: pip install ultralytics onnx onnxruntime.
Random-weight models are fine: the point is that BOTH paths decode the same tensor identically
(box format, scale, class, score).  Run:  python tests/test_onnx_parity.py"""
import sys, tempfile, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ewads_tiling as et
from ultralytics import RTDETR, YOLO
import onnxruntime as ort

rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:768, 0:768]
rgb = np.stack([(xx * 255 // 767), (yy * 255 // 767), ((xx + yy) % 255)], -1).astype(np.uint8)
rgb[100:200, 300:420] = (250, 40, 40)
rgb = np.clip(rgb.astype(int) + rng.integers(-20, 20, rgb.shape), 0, 255).astype(np.uint8)
os.chdir(tempfile.mkdtemp())

def spread_scores(model):
    """Random-init YOLO heads output ~0 scores everywhere (ties).  Randomise the cls bias so the
    top detections are distinct and both paths can be compared."""
    import torch
    head = model.model.model[-1]
    with torch.no_grad():
        for branch in ("cv3", "one2one_cv3"):
            for seq in getattr(head, branch, []):
                seq[-1].bias.copy_(torch.randn_like(seq[-1].bias) * 0.7 - 2.0)
    return model


def run(name, model, min_distinct=None):
    """min_distinct: refuse a degenerate random-init output (every query the same saturated box --
    then xyxy and xywh are indistinguishable and parity means nothing). Returns False if refused."""
    path = model.export(format="onnx", imgsz=768, opset=17, simplify=False, device="cpu", verbose=False)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None]
    raw = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    if min_distinct:
        n_distinct = len(np.unique(np.round(raw.reshape(-1, raw.shape[-1])[:, :4], 2), axis=0))
        if n_distinct < min_distinct:
            print(f"{name:<12} degenerate random init ({n_distinct} distinct boxes) -- trying next seed")
            return False
    u = et.ultralytics_predict_fn(model, 768, conf=1e-4)([rgb])[0]
    o = et.parse_raw_output(raw, 768, 1e-4)
    o[:, :4] = o[:, :4].clip(0, 768); u[:, :4] = u[:, :4].clip(0, 768)   # YOLO clips to the image, RT-DETR does not; merge_detections clips both
    u, o = u[np.argsort(-u[:, 4], kind="stable")], o[np.argsort(-o[:, 4], kind="stable")]
    # Tie-robust parity: each of Ultralytics' top-K detections must exist in the ONNX decode (same class,
    # score within 1e-3, box within 0.5 px).  Comparing only "top-1 vs top-1" failed whenever several
    # boxes tie on score (random-init RT-DETR scores cluster near 0.06) and the two sorts broke the tie
    # differently -- a test artefact, not a decode difference.  (changed 2026-10-08)
    K = min(10, len(u))
    worst = 0.0
    for row in u[:K]:
        cand = o[(o[:, 5] == row[5]) & (np.abs(o[:, 4] - row[4]) < 1e-3)]
        bd_row = float(np.abs(cand[:, :4] - row[:4]).max(1).min()) if len(cand) else float("inf")
        worst = max(worst, bd_row)
    print(f"{name:<12} onnx out {raw.shape}  top-1 ultralytics {np.round(u[0], 2).tolist()}  onnx {np.round(o[0], 2).tolist()}"
          f"  top-{K} worst matched box diff {worst:.3f}px")
    assert worst < 0.5, name
    return True

# Untrained YOLO heads give position-independent scores (ties), so parity is only meaningful for a
# real checkpoint.  Pass one:  python tests/test_onnx_parity.py path/to/yolo_best.pt
import torch
for seed in range(10):        # deterministic random init; skip degenerate ones (changed 2026-10-08)
    torch.manual_seed(seed)
    if run(f"rtdetr-l/s{seed}", RTDETR("rtdetr-l.yaml"), min_distinct=50):
        break
else:
    raise AssertionError("rtdetr-l: every seed 0-9 gave a degenerate random-init output")
for ck in sys.argv[1:]:
    run(Path(ck).stem, YOLO(ck))

# Pure-numpy format checks (no model needed)
xyxy = np.array([[10, 20, 50, 70, .9, 3], [100, 120, 180, 150, .8, 1]] + [[5, 5, 9, 9, .001, 0]] * 40, float)
xywh = np.array([[30, 45, 40, 50, .9, 3], [140, 135, 80, 30, .8, 1]] + [[7, 7, 4, 4, .001, 0]] * 40, float)
assert np.allclose(et.parse_raw_output(xyxy[None], 768, .5)[:, :4], xyxy[:2, :4])
assert np.allclose(et.parse_raw_output(xywh[None], 768, .5, "xywh")[:, :4], xyxy[:2, :4])
big = np.vstack([xywh, [[600, 700, 20, 20, .9, 2], [300, 500, 10, 10, .7, 5], [90, 80, 4, 6, .6, 1]] * 30])   # many rows: some have w<cx
assert np.allclose(et.parse_raw_output(big[None], 768, .5)[:2, :4], xyxy[:2, :4]), "auto should pick xywh"
print("format checks OK")
print("OK")
