"""End-to-end check of tile -> detect -> merge -> evaluate with a pixel-based fake detector
(finds coloured rectangles).  It produces genuine border FRAGMENTS, so it exercises the merge
rules, unlike an oracle.  Run: python tests/test_tiled_pipeline.py"""
import sys, time
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ewads_tiling as et, ewads_metrics as M

COL = np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255]], np.uint8)
rng = np.random.default_rng(1)

def make_frame(W, H, n=40):
    img = np.full((H, W, 3), 128, np.uint8); img += rng.integers(-6, 6, img.shape, dtype=np.int16).astype(np.uint8) * 0  # flat
    cls, boxes, placed = [], [], []
    tries = 0
    while len(cls) < n and tries < 2000:
        tries += 1
        L = float(np.exp(rng.uniform(np.log(10), np.log(min(W, H) * .6))))
        w, h = (int(L), int(L * rng.uniform(.4, 1.0))) if rng.random() < .5 else (int(L * rng.uniform(.4, 1.0)), int(L))
        w, h = max(w, 6), max(h, 6)
        x, y = int(rng.uniform(0, W - w)), int(rng.uniform(0, H - h))
        b = (x - 6, y - 6, x + w + 6, y + h + 6)
        if any(not (b[2] < p[0] or b[0] > p[2] or b[3] < p[1] or b[1] > p[3]) for p in placed): continue
        placed.append(b); c = int(rng.integers(0, 6))
        img[y:y + h, x:x + w] = COL[c]; cls.append(c); boxes.append([x, y, x + w, y + h])
    return img, np.array(cls), np.array(boxes, float)

def fake_predict(tiles):
    out = []
    for t in tiles:                                        # t is RGB
        d = []
        for c in range(6):
            m = (np.abs(t.astype(int) - COL[c].astype(int)).sum(2) < 90).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
            for i in range(1, n):
                x, y, w, h, a = st[i]
                if max(w, h) >= et.MIN_OBJ_PX and a >= 0.5 * w * h:
                    d.append([x, y, x + w, y + h, 0.9, c])
        out.append(np.array(d).reshape(-1, 6))
    return out

for (W, H) in [(3840, 2160), (5472, 3648), (1920, 1080)]:
    gts, dets, t_inf, ntiles = {}, {}, 0, 0
    det = et.TiledDetector(fake_predict, batch=8)
    for k in range(4 if W < 5000 else 2):
        img, cls, boxes = make_frame(W, H)
        t0 = time.time(); d = det(img, bgr=False); t_inf += time.time() - t0; ntiles += det.last_n_tiles
        gts[f"f{k}"] = (cls, boxes); dets[f"f{k}"] = d
    r = M.evaluate(gts, dets, 6); op = M.operating_point(gts, dets, 6, 0.5)
    print(f"{W}x{H}: tiles/frame {ntiles/len(gts):.0f} | mAP50 {r['map50']:.3f} mAP50-95 {r['map']:.3f} | "
          f"recall {op['recall']:.3f} FP {op['fp'].sum()} | {t_inf/len(gts):.1f}s/frame (fake det)")
    assert op["recall"] > 0.97 and op["fp"].sum() <= 2, (op["recall"], op["fp"].sum())
print("OK")
