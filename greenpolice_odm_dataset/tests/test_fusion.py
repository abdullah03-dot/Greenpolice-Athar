"""
Tests for ewads_fusion.py and 11_evidence_pipeline.py  (CPU only, no trained model needed).

  1. D4 test-time views: boxes come back EXACTLY (checked on real pixels, all 8 elements).
  2. Weighted box fusion == the reference `ensemble_boxes` package (one_per_run=False), and the
     default one_per_run=True keeps two adjacent same-class objects of one run apart.
  3. Registration: translation, rotation+scale homography, phase-correlation sign, refusal on
     unrelated frames.  GPS pixel->ground matrices agree with each other across two frames.
  4. END TO END: a synthetic flight (painted objects on textured ground, overlapping frames) run
     through 11 with a FAKE detector that has the failure modes real ones have:
       - box jitter independent per run          -> TTA/ensemble should tighten boxes (AP75 up)
       - random false positives per run/frame    -> WBF + multi-view should suppress them
       - size-dependent misses per view          -> multi-view fill should recover some
       - a SYSTEMATIC false positive (a grey slab the model always calls 'battery') that no fusion
         can remove -- the test asserts it SURVIVES, so nobody believes fusion fixes confusion.
     The gains printed are for this toy detector only; they show the mechanisms work, not what a
     real model will gain.  Measure that with 11 --data_dir on real labelled sequences.
Run:  python tests/test_fusion.py
"""
import csv
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ewads_fusion as F   # noqa: E402
import ewads_taxonomy as tx  # noqa: E402


# ───────────────────────────── 1. D4 ─────────────────────────────
def test_d4():
    W, H = 97, 61
    for k in range(8):
        img = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = 13, 7, 40, 22
        img[y1:y2, x1:x2] = 255
        t = F.d4_image(img, k)
        ys, xs = np.nonzero(t)
        b = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.9, 3]], float)
        back = F.d4_boxes_inverse(b, k, W, H)[0]
        assert np.allclose(back[:4], [x1, y1, x2, y2]) and back[4] == 0.9 and back[5] == 3, (k, back)
        fwd = F.d4_points_forward(np.array([[x1, y1], [x2, y2]], float), k, W, H)
        assert np.allclose(sorted(fwd[:, 0]), [b[0, 0], b[0, 2]]), k
    print("  ok  D4 views: all 8 boxes recovered exactly from real pixels")


# ───────────────────────────── 2. WBF ─────────────────────────────
def test_wbf():
    rng = np.random.default_rng(0)
    try:
        from ensemble_boxes import weighted_boxes_fusion as ref
    except ImportError:
        ref = None
        print("  --  ensemble_boxes not installed: parity check skipped (pip install ensemble-boxes)")
    if ref:
        for trial in range(60):
            R = int(rng.integers(1, 6))
            base = rng.uniform(0, 0.8, (12, 2))
            runs = []
            for _ in range(R):
                n = int(rng.integers(0, 15))
                idx = rng.integers(0, 12, n)
                xy = (base[idx] + rng.normal(0, 0.01, (n, 2))).clip(0, 1)
                wh = rng.uniform(0.02, 0.15, (n, 2))
                runs.append(np.column_stack([xy, np.minimum(xy + wh, 1), rng.uniform(0.001, 1, n),
                                             rng.integers(0, 3, n)]))
            wts = rng.uniform(0.5, 2, R) if trial % 2 else None
            for ct in ("avg", "max"):
                mine = F.weighted_boxes_fusion(runs, wts, 0.55, 0.0, ct, one_per_run=False)
                b, s, l = ref([r[:, :4] for r in runs], [r[:, 4] for r in runs], [r[:, 5] for r in runs],
                              weights=wts, iou_thr=0.55, conf_type=ct)
                A = np.column_stack([b, s, l]); A = A[np.lexsort((A[:, 0], A[:, 4]))]
                B = mine[:, :6]; B = B[np.lexsort((B[:, 0], B[:, 4]))]
                assert A.shape == B.shape and np.allclose(A, B, atol=2e-5), (trial, ct)
        print("  ok  WBF matches ensemble_boxes exactly (60 random cases, avg + max, with weights)")
    # adjacent same-class objects, two runs: reference merges a run's 2 boxes, one_per_run does not
    a = np.array([[0, 0, 10, 10, .9, 3], [6, 0, 16, 10, .8, 3]], float)        # IoU 0.25
    b = np.array([[3, 0, 13, 10, .85, 3]], float)                             # in the middle
    r1 = F.weighted_boxes_fusion([a, b], iou_thr=0.2, one_per_run=False)
    r2 = F.weighted_boxes_fusion([a, b], iou_thr=0.2, one_per_run=True)
    assert len(r1) == 1 and len(r2) == 2, (r1, r2)
    assert r2[:, 6].max() <= 2
    print("  ok  one_per_run keeps two adjacent objects of one run apart (reference merges them)")


# ───────────────────────────── 3. geometry ─────────────────────────────
def _ground(h=3000, w=4500, seed=1):
    rng = np.random.default_rng(seed)
    g = cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (0, 0), 2.5)
    return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)


def test_registration():
    big = _ground()
    W, H = 1600, 900
    x0, y0, dx, dy = 600, 500, 240, -90
    prev = big[y0:y0 + H, x0:x0 + W]
    cur = big[y0 + dy:y0 + dy + H, x0 + dx:x0 + dx + W]
    Hm, info = F.register_pair(prev, cur)
    assert info["method"] == "sift_homography" and np.allclose(Hm[:2, 2], [dx, dy], atol=0.5), (info, Hm)
    Hp, info = F.register_pair(prev, cur, min_inliers=10 ** 9)
    assert info["method"] == "phase_correlation" and np.allclose(Hp[:2, 2], [dx, dy], atol=1.0), (info, Hp)
    # rotation 7 deg + scale 1.05 about the centre
    A = cv2.getRotationMatrix2D((W / 2, H / 2), 7, 1.05)
    rot = cv2.warpAffine(prev, A, (W, H), borderMode=cv2.BORDER_REFLECT)
    Hr, info = F.register_pair(prev, rot)
    pts = np.array([[[400, 300]], [[1200, 700]], [[800, 450]]], np.float64)
    truth = cv2.transform(pts, cv2.invertAffineTransform(A))
    got = cv2.perspectiveTransform(pts, Hr)
    assert np.abs(got - truth).max() < 1.0, (info, got, truth)
    # frames of DIFFERENT sizes (3:2 and 4:3 stills in one folder): each keeps its own thumbnail scale
    pa = big[0:1800, 0:3200]; pb = big[50:1850, 300:2700]
    Hd, info = F.register_pair(pa, pb)
    assert Hd is not None and np.allclose(Hd[:2, :2], np.eye(2), atol=2e-3) and np.allclose(Hd[:2, 2], [300, 50], atol=1.0), Hd
    # unrelated frames must NOT register
    other = _ground(seed=7)[0:H, 0:W]
    Hn, info = F.register_pair(prev, other)
    assert Hn is None, info
    # GPS: a ground point seen in two frames maps to the same east/north
    origin = (25.2, 55.3)
    m1 = {"lat": 25.2, "lon": 55.3, "yaw_deg": 30.0, "gsd_m": 0.01}
    e_off, n_off = 4.0, 3.0                           # aircraft moved 4 m east, 3 m north
    lat2, lon2 = F.en_to_latlon(e_off, n_off, *origin)
    m2 = {"lat": lat2, "lon": lon2, "yaw_deg": 30.0, "gsd_m": 0.01}
    A1, A2 = F.pixel_to_en_matrix(m1, (W, H), origin), F.pixel_to_en_matrix(m2, (W, H), origin)
    p_in_1 = np.array([1000.0, 300.0, 1.0])
    gpt = A1 @ p_in_1
    p_in_2 = np.linalg.inv(A2) @ gpt
    assert np.allclose(A2 @ p_in_2, gpt) and np.linalg.norm(p_in_2[:2] - p_in_1[:2]) > 100
    # heading convention: image top edge points along the yaw
    top = A1 @ np.array([W / 2, 0, 1]); ctr = A1 @ np.array([W / 2, H / 2, 1])
    v = top[:2] - ctr[:2]
    assert abs(math.degrees(math.atan2(v[0], v[1])) - 30.0) < 1e-6
    print("  ok  registration: translation 0.5 px, rotation+scale < 1 px, phase-corr sign, rejects "
          "unrelated frames; GPS matrices consistent, heading convention checked")


# ───────────────────────────── 4. end to end ─────────────────────────────
CLASS_COLORS = {0: (40, 40, 220), 2: (40, 200, 40), 3: (220, 60, 40), 6: (200, 40, 200), 9: (40, 220, 220)}
DISTRACTOR = (150, 90, 20)         # a slab (colour unlike the grey ground) the fake model ALWAYS calls 'battery' (class 3)
GLINT = (250, 250, 250)            # sun glint: in ONE frame only (depends on view angle); called crt_monitor


def make_flight(root: Path, n_frames=18, W=1600, H=900, step=150, seed=3):
    rng = np.random.default_rng(seed)
    gh, gw = H + 80, W + step * (n_frames - 1) + 40
    ground = cv2.cvtColor(_ground(gh, gw, seed), cv2.COLOR_GRAY2BGR)
    ground = (ground.astype(np.float32) * 0.5 + 60).astype(np.uint8)        # keep paint colours distinct
    objs = []
    for _ in range(70):
        c = int(rng.choice(list(CLASS_COLORS)))
        s = float(rng.choice([12, 18, 26, 40, 64]))
        w_, h_ = s, s * rng.uniform(0.5, 1.0)
        x, y = rng.uniform(20, gw - 20 - w_), rng.uniform(20, gh - 20 - h_)
        box = np.array([x, y, x + w_, y + h_])
        if objs and np.any([(min(box[2], o[1][2]) - max(box[0], o[1][0]) > -6) and
                            (min(box[3], o[1][3]) - max(box[1], o[1][1]) > -6) for o in objs]):
            continue
        objs.append((c, box))
        cv2.rectangle(ground, (int(round(x)), int(round(y))), (int(round(x + w_)) - 1, int(round(y + h_)) - 1),
                      CLASS_COLORS[c], -1)
    dis = []
    for _ in range(4):
        x, y = rng.uniform(50, gw - 100), rng.uniform(50, gh - 100)
        ok = all(not (x < o[1][2] + 10 and x + 30 > o[1][0] - 10 and y < o[1][3] + 10 and y + 20 > o[1][1] - 10)
                 for o in objs)
        if ok:
            cv2.rectangle(ground, (int(x), int(y)), (int(x) + 29, int(y) + 19), DISTRACTOR, -1)
            dis.append((x, y))
    (root / "images").mkdir(parents=True); (root / "labels").mkdir()
    for i in range(n_frames):
        ox, oy = 20 + i * step, 40 + int(round(25 * math.sin(i / 3)))
        fr = ground[oy:oy + H, ox:ox + W].copy()
        for _ in range(int(rng.poisson(2.0))):                    # transient glints: every TTA view of
            gx, gy = int(rng.uniform(0, W - 30)), int(rng.uniform(0, H - 30))   # this frame sees them,
            cv2.rectangle(fr, (gx, gy), (gx + 17, gy + 13), GLINT, -1)    # the next frame does not
        cv2.imwrite(str(root / "images" / f"frame_{i:03d}.png"), fr)
        lines = []
        for c, b in objs:
            bb = (b - [ox, oy, ox, oy]).clip([0, 0, 0, 0], [W, H, W, H])        # objects cut by the frame
            if (bb[2] - bb[0]) * (bb[3] - bb[1]) >= 20 and max(bb[2] - bb[0], bb[3] - bb[1]) >= 4:  # edge are
                cx, cy, ww, hh = (bb[0] + bb[2]) / 2 / W, (bb[1] + bb[3]) / 2 / H, (bb[2] - bb[0]) / W, (bb[3] - bb[1]) / H
                lines.append(f"{c} {cx:.7f} {cy:.7f} {ww:.7f} {hh:.7f}")         # labelled clipped (as 02 does)
        (root / "labels" / f"frame_{i:03d}.txt").write_text("\n".join(lines))
    return objs, dis


class FakeModel:
    """Finds the painted rectangles by colour (so geometry, tiling, TTA and fusion are exercised for
    real) and then behaves like an imperfect detector."""

    def __init__(self, seed, jitter=0.07, bias_px=0.0, fp_rate=0.6, miss_k=10.0):
        self.rng = np.random.default_rng(seed)
        self.jitter, self.bias, self.fp_rate, self.miss_k = jitter, bias_px, fp_rate, miss_k

    def predict(self, tiles):
        out = []
        for t in tiles:
            bgr = t[:, :, ::-1].astype(np.int16)
            dets = []
            for c, col in list(CLASS_COLORS.items()) + [(3, DISTRACTOR), (0, GLINT)]:
                m = (np.abs(bgr - np.array(col)).max(2) < 12).astype(np.uint8)
                n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
                for x, y, w, h, area in st[1:]:
                    if area < 20:
                        continue
                    s = max(w, h)
                    if col in (DISTRACTOR, GLINT):
                        conf = 0.55 + 0.1 * self.rng.random()                      # consistent across runs
                    else:
                        if self.rng.random() < self.miss_k / (s * s / 4 + self.miss_k):   # small = missed more
                            continue
                        conf = float(np.clip(0.45 + 0.4 * self.rng.random(), 0, 1))
                    j = self.jitter * s + 0.6
                    b = np.array([x, y, x + w, y + h], float) + self.rng.normal(0, j, 4) + [-self.bias, -self.bias, self.bias, self.bias]
                    dets.append([*b, conf, c])
            for _ in range(self.rng.poisson(self.fp_rate)):                           # random false positives
                s = self.rng.uniform(10, 40); x, y = self.rng.uniform(0, t.shape[1] - s), self.rng.uniform(0, t.shape[0] - s)
                dets.append([x, y, x + s, y + s * 0.7, float(self.rng.uniform(0.05, 0.6)), float(self.rng.choice(list(CLASS_COLORS)))])
            out.append(np.asarray(dets, float).reshape(-1, 6))
        return out


class FakeEnsemble:
    """drop-in for 11's Ensemble: one FakeModel per --weights entry, through the real TiledDetector."""

    def __init__(self, paths, args, b08):
        import ewads_tiling as et
        self.models = [FakeModel(11), FakeModel(23, bias_px=0.4)][:len(paths)]
        self.dets = [et.TiledDetector(m.predict, tile=512, overlap=96, batch=8, conf=0.001)
                     for m in self.models]

    def run(self, frame, plan):
        res = []
        for d in self.dets:
            res += F.run_tta(frame, lambda f: d(f), plan)
        return res


def _load_11():
    spec = importlib.util.spec_from_file_location("ev11", ROOT / "11_evidence_pipeline.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    ev.Ensemble = FakeEnsemble
    return ev


def test_end_to_end():
    ev = _load_11()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        objs, dis = make_flight(td / "seq")
        for n in ("a.pt", "b.pt"):
            (td / n).write_bytes(n.encode())
        out = td / "abl"
        sys.argv = ["11", "--weights", str(td / "a.pt"), str(td / "b.pt"), "--data_dir", str(td / "seq"),
                    "--tta", "flips", "--out_dir", str(out), "--tile", "512", "--report_conf", "0.2"]
        ev.main()
        rd = out / (out / "LATEST_RUN.txt").read_text()
        rep = json.loads((rd / "evidence_report.json").read_text())
        st = rep["stages"]
        s, t, e, mv, fl = (st[k] for k in ("single", "tta", "tta+ensemble", "multiview", "multiview+fill"))
        assert t["map75"] > s["map75"] and e["map75"] >= t["map75"] - 0.01, "TTA/ensemble did not tighten boxes"
        assert e["map5095"] > s["map5095"], "fusion did not raise mAP"
        assert t["fp_share"] < s["fp_share"], "WBF did not suppress per-run false positives"
        # glints (in ONE frame, in every TTA view of it): WBF cannot remove them, multi-view should
        imgs = {f.stem: cv2.imread(str(f)) for f in (td / "seq" / "images").iterdir()}

        def n_glints(stage):
            z = np.load(rd / f"detections_{stage.replace('+', '_')}.npz")
            n = 0
            for fid, d in ((k, z[k]) for k in z.files):
                for x1, y1, x2, y2 in d[d[:, 4] >= st[stage]["conf"], :4]:
                    p = imgs[fid][int(max(0, y1 + 2)):int(y2 - 2), int(max(0, x1 + 2)):int(x2 - 2)]
                    if p.size and (np.abs(p.reshape(-1, 3).astype(int) - GLINT).max(1) < 12).mean() > 0.6:
                        n += 1
            return n
        g_e, g_mv = n_glints("tta+ensemble"), n_glints("multiview")
        assert g_mv < 0.5 * g_e, (g_e, g_mv)
        assert rep["chain"]["valid"]
        # candidate objects: every ground object found once, few duplicates (track merging)
        offs = {f"frame_{i:03d}": np.array([20 + i * 150, 40 + int(round(25 * math.sin(i / 3)))] * 2)
                for i in range(len(list((td / "seq" / "images").iterdir())))}
        B = np.array([np.array(o["best_box_xyxy"]) + offs[o["best_frame"]] for o in rep["objects"]])
        G = np.array([b for _, b in objs])
        iou = F._iou(B, G)
        hit = iou.max(1) >= 0.3
        found = set(iou.argmax(1)[hit])
        dups = int(hit.sum()) - len(found)
        assert len(found) >= 0.9 * len(objs), (len(found), len(objs))
        assert dups <= 0.05 * len(objs), dups
        # the systematic 'battery' slab must still be there (fusion cannot fix a confusion)
        rows = list(csv.DictReader(open(rd / "review.csv")))
        assert rows and all((rd / "review_crops" / r["crop"]).exists() for r in rows)
        recs = [json.loads(l) for l in open(rd / "evidence_audit.jsonl")]
        fdet = [d for r in recs if r["context"]["record_type"] == "frame" for d in r["detections"]]
        assert fdet and all("observed_conf" in d and "track_class" in d for d in fdet)
        assert all(d["observed_conf"] is not None for d in fdet if not d["propagated"])
        assert all(not o.get("crop_error") for o in rep["objects"])
        slab_hits = 0
        frames = {f.stem: f for f in (td / "seq" / "images").iterdir()}
        for r in rows:
            img = cv2.imread(str(frames[r["best_frame"]]))
            o = [o for o in rep["objects"] if o["track_id"] == int(r["track_id"])][0]
            x1, y1, x2, y2 = [int(round(v)) for v in o["best_box_xyxy"]]
            patch = img[max(0, y1 + 2):y2 - 2, max(0, x1 + 2):x2 - 2]
            if patch.size and np.abs(patch.reshape(-1, 3).astype(int) - DISTRACTOR).max(1).mean() < 12:
                slab_hits += int(o["class_name"] == "battery" and o["confirmed"])
        assert slab_hits >= 1, "systematic false positive vanished -- the toy is not testing what it claims"
        n_conf = sum(r["confirmed"] == "True" for r in rows)
        print(f"  ok  end to end on a {len(list(frames))}-frame synthetic flight ({len(objs)} objects, "
              f"{len(dis)} systematic distractors, glints): {len(rep['objects'])} candidates ({n_conf} confirmed) "
              f"cover {len(found)}/{len(objs)} objects with {dups} duplicates; {slab_hits} systematic "
              f"distractor(s) correctly NOT removed; audit chain valid ({rep['chain']['records']} records)")
        print("      mAP .5:.95  single %.3f -> TTA %.3f -> TTA+ensemble %.3f -> multi-view %.3f  (toy detector!)"
              % (s["map5095"], t["map5095"], e["map5095"], mv["map5095"]))
        print("      FP share at recall>=0.87:  %.3f -> %.3f -> %.3f -> %.3f;  +fill %.3f"
              % (s["fp_share"], t["fp_share"], e["fp_share"], mv["fp_share"], fl["fp_share"]))
        print(f"      single-frame glints above the operating threshold: TTA+ensemble {g_e} -> multi-view {g_mv}")
        # re-fusion from cache without models
        sys.argv = ["11", "--data_dir", str(td / "seq"), "--out_dir", str(out), "--reuse_cache",
                    "--tta", "flips", "--wbf_iou", "0.6", "--stage_thresholds", str(rd / "stage_thresholds.json")]
        import time as _t
        _t.sleep(1.1)                                       # new run folder (timestamped name)
        ev.main()
        rd2 = out / (out / "LATEST_RUN.txt").read_text()
        assert rd2 != rd and (rd / "review.csv").exists() and (rd / "evidence_audit.jsonl").exists(), \
            "re-fusion overwrote the first run's evidence"
        rep2 = json.loads((rd2 / "evidence_report.json").read_text())
        assert all(not v["calibrated_here"] for v in rep2["stages"].values())
        # cache made with other settings must be refused, not silently mixed
        z = dict(np.load(out / "cache" / "frame_003.npz"))
        z["key"] = np.array("deadbeef"); np.savez_compressed(out / "cache" / "frame_003.npz", **z)
        try:
            ev.main()
            raise AssertionError("mixed cache accepted")
        except SystemExit as e:
            assert "different models/settings" in str(e)
        print("  ok  --reuse_cache re-fuses without models into a NEW run folder (earlier evidence untouched); "
              "mixed caches refused; --stage_thresholds freezes operating points and warns on changed settings")


def test_reload_verification():
    """review crops are only cut from the exact bytes that were analysed"""
    import hashlib
    ev = _load_11()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.png"
        cv2.imwrite(str(p), np.full((40, 60, 3), 77, np.uint8))
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        img, why = ev._reload({"image_path": str(p), "image_sha256": h})
        assert img is not None and why is None
        cv2.imwrite(str(p), np.full((40, 60, 3), 78, np.uint8))          # source edited after analysis
        img, why = ev._reload({"image_path": str(p), "image_sha256": h})
        assert img is None and "changed" in why
    print("  ok  review crops refuse a source file that changed after analysis")


def test_video():
    """--video path: lossless video of the flight, no telemetry (image-motion scheduling), crops
    re-read from the video must be byte-identical to the frames that were analysed."""
    import hashlib
    ev = _load_11()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        objs, _ = make_flight(td / "seq", n_frames=12)
        frames = sorted((td / "seq" / "images").iterdir())
        first = cv2.imread(str(frames[0]))
        vw = cv2.VideoWriter(str(td / "flight.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 2.0,
                             (first.shape[1], first.shape[0]))
        for f in frames:
            vw.write(cv2.imread(str(f)))
        vw.release()
        (td / "a.pt").write_bytes(b"a")
        out = td / "vid"
        sys.argv = ["11", "--weights", str(td / "a.pt"), "--video", str(td / "flight.avi"), "--tta", "none",
                    "--out_dir", str(out), "--tile", "512", "--report_conf", "0.3", "--views", "3"]
        ev.main()
        rd = out / (out / "LATEST_RUN.txt").read_text()
        rep = json.loads((rd / "evidence_report.json").read_text())
        recs = [json.loads(l) for l in open(rd / "evidence_audit.jsonl")]
        fr = [r for r in recs if r["context"]["record_type"] == "frame"]
        assert len(fr) >= 10, len(fr)                                   # 150 px moves -> nearly every frame
        assert rep["chain"]["valid"] and len(rep["objects"]) >= 0.8 * len(objs), (len(rep["objects"]), len(objs))
        # crops: raw PNG pixels == the analysed frame's pixels at that place
        cap = cv2.VideoCapture(str(td / "flight.avi"))
        allf = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            allf.append(f)
        cap.release()                                                  # Windows: release before the temp dir is deleted
        for o in rep["objects"][:10]:
            idx = int(o["best_frame"][1:])
            crop = cv2.imread(str(rd / "review_crops" / o["crop_raw"]))
            x, y = o["crop_offset_xy"]
            assert np.array_equal(crop, allf[idx][y:y + crop.shape[0], x:x + crop.shape[1]])
            assert hashlib.sha256(crop.tobytes()).hexdigest() == o["crop_pixels_sha256"]
            assert hashlib.sha256(allf[idx].tobytes()).hexdigest() == \
                next(r for r in fr if r["context"]["frame_index"] == idx)["image_sha256"]
        print(f"  ok  video path: {len(fr)} frames scheduled from {len(frames)}, {len(rep['objects'])} candidates, "
              f"crops byte-identical to the analysed frames, frame hashes reproducible")


if __name__ == "__main__":
    test_d4()
    test_wbf()
    test_registration()
    test_reload_verification()
    test_end_to_end()
    test_video()
    print("\nfusion tests passed")
