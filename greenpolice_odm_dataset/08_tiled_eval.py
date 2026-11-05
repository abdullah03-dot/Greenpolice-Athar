"""
EWADS -- Step 8: full-frame evaluation with tiled inference  (the numbers that go in the paper)
==============================================================================================
Ultralytics' own `val` on the training tiles is only a training-time proxy: it never merges
tiles, splits objects at tile borders and has no notion of altitude.  This script evaluates the
way the drone will actually be used:

    full-resolution frame -> pyramid tiles -> model -> merge (fragment drop + NMS) -> native boxes
    -> COCO mAP@0.5:0.95 (numerically identical to pycocotools), recall, FP, per class,
       per object size and per altitude band.

Usage
  # PyTorch weights (RT-DETR or YOLO checkpoint), full-frame test set:
  python 08_tiled_eval.py --weights runs/ewads/Stage3_HardFinetune/weights/best.pt \\
      --data_dir data/test --meta data/test/scene_index.jsonl --out_dir eval/test

  # Calibrate the deployment threshold on VAL, then report on TEST at that threshold:
  python 08_tiled_eval.py --weights best.pt --data_dir data/val_full --calibrate --out_dir eval/val
  python 08_tiled_eval.py --weights best.pt --data_dir data/test --conf 0.31 --out_dir eval/test

  # TensorRT engine on the Jetson (same backend as 07; Ultralytics or trtexec engines):
  python 08_tiled_eval.py --engine best_fp16.engine --data_dir data/test --conf 0.31 --out_dir eval/fp16

  # ONNX model (fixed tile size read from the graph):
  python 08_tiled_eval.py --onnx ewads_yolov10n_fp32.onnx --data_dir data/test --out_dir eval/onnx

  # Re-score cached detections without re-running the model:
  python 08_tiled_eval.py --data_dir data/test --cache eval/test/detections.npz --conf 0.31

--data_dir has images/ and labels/ at FULL resolution (never tiled, never augmented, never used
for calibration).  Labels are YOLO-normalised; an empty label file = true-negative frame.
--meta (optional): scene_index.jsonl from the renderer (image_path + altitude_m) or a JSON/CSV
mapping image stem -> altitude in metres; enables the per-altitude table.

Two sets of headline numbers are printed: ALL labelled objects, and the ASSESSABLE ENVELOPE
(objects >= --assess_px native pixels, COCO-style ignore for the rest).  The envelope is what the
system claims (07 logs per frame which classes were assessable); report both.

Then (ewads_hier_eval):
  * HIERARCHY  fine (18) / group (8, ewads_taxonomy) / binary ("e-waste") mAP, recall, FP share
  * TIER-1     the toxic classes on their own
  * MATRIX     mAP by altitude band x object size (needs --meta)
  * SITES      survey view: dump sites found, false alerts per frame
  * PRE-REGISTERED TARGETS from --targets (default targets_v1.json), each section scored on the frames
    inside its altitude scope, with the freeze status (tools/freeze_targets.py) printed next to it.

  # score the 11 evidence pipeline's output (its 08-compatible cache) against the evidence targets:
  python 08_tiled_eval.py --data_dir data/test_seq --cache abl/test/<run>/detections_multiview.npz \
      --conf 0.34 --meta data/test_seq/scene_index.jsonl --targets_section evidence --out_dir eval/evidence
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ewads_hier_eval as HE
import ewads_metrics as M
import ewads_taxonomy as tx
import ewads_tiling as et

CLASSES = tx.CLASSES                       # single source of truth (ewads_taxonomy)
ALT_BANDS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]
TARGETS = {"map5095": 0.68, "recall": 0.87, "fp_share": 0.05}      # from the project brief


# ───────────────────────── backends ─────────────────────────
def load_ultralytics(path: str):
    import torch
    from ultralytics import RTDETR, YOLO
    ck = torch.load(path, map_location="cpu", weights_only=False)
    name = type(ck.get("model") or ck.get("ema")).__name__
    return RTDETR(path) if "RTDETR" in name else YOLO(path)


def _load_07():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench07", Path(__file__).resolve().parent / "07_edge_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_predict_fn(args, tile_hint):
    """-> (predict_fn, tile, batch).  ONNX and TensorRT go through 07's backends, so this script
    scores EXACTLY the code path the drone runs (incl. batching / static-batch padding)."""
    if args.onnx or args.engine:
        b07 = _load_07()
        inf = b07.ONNXInferencer(Path(args.onnx)) if args.onnx else b07.TensorRTInferencer(Path(args.engine))
        tile = inf.input_size or tile_hint or et.DEFAULT_TILE
        batch = max(1, int(getattr(inf, "max_batch", 1) or 1))

        def fn(tiles):
            x = np.concatenate([inf.preprocess_tile(t) for t in tiles])
            raw = inf.infer(x)[0]
            return [et.parse_raw_output(raw[i:i + 1], tile, 0.0) for i in range(len(tiles))]
        print(f"  {'ONNX' if args.onnx else 'TensorRT'} {Path(args.onnx or args.engine).name}: tile {tile}, batch {batch}")
        return fn, tile, batch
    model = load_ultralytics(args.weights)
    tile = args.tile or tile_hint or et.DEFAULT_TILE
    print(f"  {type(model).__name__} {Path(args.weights).name}: tile {tile}")
    return et.ultralytics_predict_fn(model, tile, conf=args.min_conf, device=args.device), tile, args.batch


def tile_from_yaml(p):
    if not p or not Path(p).exists():
        return None, None
    try:
        import yaml
        e = (yaml.safe_load(open(p)) or {}).get("ewads", {})
        return e.get("tile"), e.get("overlap")
    except Exception:
        return None, None


# ───────────────────────── data ─────────────────────────
def load_gt(data_dir: Path, max_images=None, return_sizes=False):
    imgs = sorted(p for p in (data_dir / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if max_images:
        imgs = imgs[:max_images]
    gts, sizes = {}, {}
    for p in imgs:
        from PIL import Image
        with Image.open(p) as im:
            w, h = im.size
        sizes[p.stem] = (w, h)
        gts[p.stem] = et.yolo_to_xyxy(et.read_yolo(data_dir / "labels" / f"{p.stem}.txt"), w, h)
    return (imgs, gts, sizes) if return_sizes else (imgs, gts)


def load_meta(path):
    """stem -> altitude_m"""
    if not path or not Path(path).exists():
        return {}
    p = Path(path)
    alt = {}
    if p.suffix == ".jsonl":
        for line in p.read_text().splitlines():
            r = json.loads(line)
            stem = Path(r.get("image_path", "")).stem or r.get("stem")
            if stem and "altitude_m" in r:
                alt[stem] = float(r["altitude_m"])
    elif p.suffix == ".json":
        alt = {k: float(v) for k, v in json.loads(p.read_text()).items()}
    elif p.suffix == ".csv":
        for r in csv.DictReader(open(p)):
            alt[r["stem"]] = float(r["altitude_m"])
    return alt


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--weights", help="Ultralytics .pt (RT-DETR or YOLO)")
    src.add_argument("--onnx", help="ONNX model (YOLOv10 e2e or RT-DETR output)")
    src.add_argument("--engine", help="TensorRT engine (trtexec or Ultralytics .engine) -- on the Jetson")
    ap.add_argument("--data_dir", required=True, help="full-resolution images/ + labels/")
    ap.add_argument("--out_dir", default="./eval")
    ap.add_argument("--dataset_yaml", default=None, help="read tile/overlap from its ewads: block")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--overlap", type=int, default=None)
    ap.add_argument("--max_levels", type=int, default=None,
                    help="cheaper inference: only the first N pyramid levels + the final one")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--min_conf", type=float, default=0.001, help="lowest score kept (needed for mAP)")
    ap.add_argument("--conf", type=float, default=None, help="operating threshold for recall/FP")
    ap.add_argument("--calibrate", action="store_true", help="pick the highest conf reaching --target_recall")
    ap.add_argument("--target_recall", type=float, default=TARGETS["recall"])
    ap.add_argument("--meta", default=None, help="scene_index.jsonl / json / csv with altitude_m")
    ap.add_argument("--cache", default=None, help="detections.npz: load if it exists, else write")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--assess_px", type=float, default=16,
                    help="'assessable envelope': objects whose native long side >= this (02's SPAWN_MIN_PX)")
    ap.add_argument("--targets", default=str(Path(__file__).resolve().parent / "targets_v2.json"),
                    help="pre-registered targets file (see tools/freeze_targets.py); '' to skip. "
                         "targets_v1.json (schema 1) still works")
    ap.add_argument("--allow_selection_on_heldout", action="store_true",
                    help="override the guard that refuses --calibrate on a held-out/test folder (recorded)")
    ap.add_argument("--targets_dataset", default=None,
                    choices=["synthetic_heldout", "sim_to_sim_loeo", "sim_to_real"],
                    help="schema-v2 targets: which dataset THIS run scores. Only that dataset's sections "
                         "are checked (required for v2; ignored for v1)")
    ap.add_argument("--targets_section", default="auto",
                    help="which section applies to this data (v1: evidence | survey; v2: e.g. "
                         "synthetic_heldout.evidence). auto = every selected section, each on the frames "
                         "inside its altitude scope; naming one asserts the scope")
    ap.add_argument("--camera", default="phantom4pro", help="for altitude -> GSD (site linking in metres)")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="image-level bootstrap resamples for 95%% CIs (0 = off)")
    ap.add_argument("--bootstrap_seed", type=int, default=2026)
    args = ap.parse_args()

    data_dir, out = Path(args.data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    imgs, gts, sizes = load_gt(data_dir, args.max_images, return_sizes=True)
    n_gt = sum(len(c) for c, _ in gts.values())
    print(f"\n  {len(imgs)} frames, {n_gt} labelled objects "
          f"({sum(len(c) == 0 for c, _ in gts.values())} negative frames)")

    cache = Path(args.cache) if args.cache else out / "detections.npz"
    dets, timing = {}, {}
    if cache.exists() and not (args.weights or args.onnx or args.engine):
        with np.load(cache, allow_pickle=True) as z:          # close the file (Windows keeps it locked)
            dets = {k: z[k] for k in z.files if k != "__timing__"}
            timing = z["__timing__"].item() if "__timing__" in z.files else {}
        print(f"  loaded cached detections: {cache}")
    else:
        if not (args.weights or args.onnx or args.engine):
            sys.exit("Provide --weights, --onnx or --engine (or an existing --cache).")
        y_tile, y_ov = tile_from_yaml(args.dataset_yaml)
        fn, tile, batch = build_predict_fn(args, args.tile or y_tile)
        overlap = args.overlap or y_ov or et.DEFAULT_OVERLAP
        det = et.TiledDetector(fn, tile=tile, overlap=overlap, batch=batch, conf=args.min_conf,
                               max_levels=args.max_levels)
        t_all = time.time()
        n_tiles = []
        for i, p in enumerate(imgs, 1):
            t0 = time.time()
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            dets[p.stem] = det(frame).astype(np.float32)
            n_tiles.append(det.last_n_tiles)
            timing[p.stem] = time.time() - t0
            print(f"    {i}/{len(imgs)}  {p.name}  tiles {det.last_n_tiles}  dets {len(dets[p.stem])}", end="\r")
        print(f"\n  inference {time.time() - t_all:.0f}s | {np.mean(n_tiles):.1f} tiles/frame | "
              f"{np.mean(list(timing.values())):.2f}s/frame (incl. disk read + tiling + merge)")
        np.savez_compressed(cache, __timing__=np.array(timing, dtype=object), **dets)
        print(f"  cached detections -> {cache}")

    nc = len(CLASSES)
    # ─────────── overall + per class ───────────
    r = M.evaluate(gts, dets, nc)
    print(f"\n  mAP@0.5:0.95 {r['map']:.4f}   mAP@0.5 {r['map50']:.4f}   mAP@0.75 {r['map75']:.4f}")

    # ─────────── operating point ───────────
    conf = args.conf
    cal = None
    if args.calibrate:
        why = HE.selection_data_problem(args.data_dir)
        if why and not args.allow_selection_on_heldout:
            sys.exit(f"[ERROR] --calibrate refused: {why}")
        if why:
            print(f"  ⚠  OVERRIDE (--allow_selection_on_heldout): {why}")
    if args.calibrate or conf is None:
        cal = M.calibrate_threshold(gts, dets, nc, args.target_recall)
        conf = cal["threshold"] if args.conf is None else args.conf
        flag = "" if cal["reached_target"] else "  (TARGET NOT REACHABLE -- lowest threshold shown)"
        print(f"  calibrated threshold for recall >= {args.target_recall}: {cal['threshold']:.2f}{flag}")
        if args.conf is None and not args.calibrate:
            print("  (no --conf given: used the calibrated value ON THIS SET; for the paper calibrate on "
                  "val and pass --conf to the test run)")
    op = M.operating_point(gts, dets, nc, conf)
    print(f"  @conf {conf:.2f}: recall {op['recall']:.4f} | precision {op['precision']:.4f} | "
          f"FP/(TP+FP) {op['fp_share']:.4f} | FP/frame {op['fp_per_image']:.3f}"
          + (f" | FP/negative frame {op['fp_per_negative_image']:.3f}" if op['fp_per_negative_image'] is not None else ""))

    print(f"\n  {'class':<18}{'n_gt':>7}{'AP':>8}{'AP50':>8}{'recall@conf':>13}")
    rows = []
    for c, name in enumerate(CLASSES):
        if r["n_gt"][c] == 0:
            continue
        rc = op["tp"][c] / max(op["tp"][c] + op["fn"][c], 1)
        print(f"  {name:<18}{r['n_gt'][c]:>7}{r['per_class_ap'][c]:>8.3f}{r['ap'][c, 0]:>8.3f}{rc:>13.3f}")
        rows.append([name, int(r["n_gt"][c]), r["per_class_ap"][c], r["ap"][c, 0], rc])

    # ─────────── by object size (native px) ───────────
    by_size = {}
    print(f"\n  by object size (native long side, px):  mAP@.5:.95   mAP@.5   n_gt")
    for lab, rng in M.SIZE_BUCKETS.items():
        rs = M.evaluate(gts, dets, nc, size_range=rng)
        by_size[lab] = {"map": rs["map"], "map50": rs["map50"], "n_gt": int(rs["n_gt"].sum())}
        if rs["n_gt"].sum():
            print(f"    {lab:>8}  {rs['map']:>22.3f}{rs['map50']:>10.3f}{int(rs['n_gt'].sum()):>8}")

    # ─────────── by altitude ───────────
    by_alt = {}
    alt = load_meta(args.meta)
    if alt:
        print(f"\n  by flight altitude:   mAP@.5:.95   mAP@.5   recall@conf   n_frames   n_gt")
        for lo, hi in ALT_BANDS:
            ids = [s for s in gts if s in alt and lo <= alt[s] < hi]
            if not ids:
                continue
            ra = M.evaluate(gts, dets, nc, image_ids=ids)
            oa = M.operating_point(gts, dets, nc, conf, image_ids=ids)
            lab = f"{lo:g}-{hi:g}m" if hi < 1e8 else f">{lo:g}m"
            by_alt[lab] = {"map": ra["map"], "map50": ra["map50"], "recall": oa["recall"],
                           "n_frames": len(ids), "n_gt": int(ra["n_gt"].sum()),
                           "per_class_ap": {CLASSES[c]: float(ra["per_class_ap"][c])
                                            for c in range(nc) if ra["n_gt"][c]}}
            print(f"    {lab:>8}  {ra['map']:>12.3f}{ra['map50']:>10.3f}{oa['recall']:>13.3f}"
                  f"{len(ids):>10}{int(ra['n_gt'].sum()):>7}")
        miss = [s for s in gts if s not in alt]
        if miss:
            print(f"    ({len(miss)} frames had no altitude in --meta)")

    # ─────────── assessable envelope ───────────
    # Objects below --assess_px are physically below what the system claims to detect (the
    # renderer never spawns them, and the forensic log marks such classes 'not assessable').
    # Report BOTH numbers: the envelope one is what the deployed system promises; the all-object
    # one shows what the envelope hides.  Never quote the envelope number without saying so.
    env_rng = (args.assess_px, 1e9)
    re_ = M.evaluate(gts, dets, nc, size_range=env_rng)
    ope = M.operating_point(gts, dets, nc, conf, min_px=args.assess_px)
    n_small = n_gt - int(re_["n_gt"].sum())
    print(f"\n  ASSESSABLE ENVELOPE (objects >= {args.assess_px:g} px; {n_small} of {n_gt} smaller objects ignored):")
    print(f"    mAP@0.5:0.95 {re_['map']:.4f}   mAP@0.5 {re_['map50']:.4f}   @conf {conf:.2f}: recall "
          f"{ope['recall']:.4f} | FP/(TP+FP) {ope['fp_share']:.4f}")
    envelope = {"assess_px": args.assess_px, "n_ignored_small_gt": n_small, "map5095": re_["map"],
                "map50": re_["map50"], "recall": ope["recall"], "fp_share": ope["fp_share"],
                "per_class_ap": {CLASSES[c]: float(re_["per_class_ap"][c]) for c in range(nc) if re_["n_gt"][c]}}

    # ─────────── hierarchy: fine / group / binary ───────────
    hier = {}
    print(f"\n  HIERARCHY (ewads_taxonomy; @conf {conf:.2f}; group/binary dedup = IoU-0.7 NMS inside the mapped class)")
    print(f"    {'level':<8}{'classes':>8}{'mAP':>8}{'AP50':>8}{'env mAP':>9}{'recall':>8}{'FP share':>10}")
    for lvl in tx.LEVELS:
        row = {k: HE.metric(gts, dets, metric=k, level=lvl, conf=conf)[0]
               for k in ("map5095", "map50", "recall", "fp_share")}
        row["env_map5095"] = HE.metric(gts, dets, metric="map5095", level=lvl, min_px=args.assess_px)[0]
        hier[lvl] = row
        print(f"    {lvl:<8}{tx.n_classes(lvl):>8}{row['map5095']:>8.3f}{row['map50']:>8.3f}{row['env_map5095']:>9.3f}"
              f"{row['recall']:>8.3f}{row['fp_share']:>10.3f}")
    print("    (group/binary >> fine means the errors are confusions between similar classes, not misses)")
    grp_ap = {}
    gg, gd, gnc = HE.to_level(gts, dets, "group")
    rg = M.evaluate(gg, gd, gnc)
    for i, gname in enumerate(tx.GROUP_NAMES):
        if rg["n_gt"][i]:
            grp_ap[gname] = {"ap5095": float(rg["per_class_ap"][i]), "ap50": float(rg["ap"][i, 0]),
                             "n_gt": int(rg["n_gt"][i]), "members": tx.GROUPS[gname]}

    # ─────────── tier 1 (toxic) ───────────
    t1 = {k: HE.metric(gts, dets, metric=k, classes="tier1", conf=conf, **kw)[0]
          for k, kw in (("map5095", {}), ("map50", {}), ("recall", {}), ("fp_share", {}))}
    t1["env_recall"] = HE.metric(gts, dets, metric="recall", classes="tier1", conf=conf, min_px=args.assess_px)[0]
    print(f"\n  TIER-1 ({', '.join(sorted(tx.TIER_1))}): mAP {t1['map5095']:.3f}  AP50 {t1['map50']:.3f}  "
          f"recall {t1['recall']:.3f} (envelope {t1['env_recall']:.3f})  FP share {t1['fp_share']:.3f}")

    # ─────────── altitude x size ───────────
    matrix = {}
    if alt:
        matrix = HE.size_altitude_matrix(gts, dets, alt, ALT_BANDS)
        bl = list(M.SIZE_BUCKETS)
        print(f"\n  mAP@.5:.95 BY ALTITUDE x OBJECT SIZE (native long side px; '-' = no objects)")
        print(f"    {'':<9}" + "".join(f"{b:>9}" for b in bl))
        for lab, row in matrix.items():
            print(f"    {lab:<9}" + "".join(f"{row['cells'][b]['map5095']:>9.3f}" if b in row["cells"] else f"{'-':>9}"
                                       for b in bl) + f"   ({row['n_frames']} frames)")

    # ─────────── survey: sites ───────────
    tspec = HE.load_targets(args.targets) if args.targets and Path(args.targets).exists() else None
    site_cfg = {k: v for k, v in ((tspec or {}).get("spec", {}).get("site_definition", {}) or {}).items()
                if k in ("link_m", "link_px_if_no_altitude", "margin_px")}
    import ewads_survey as es
    cam = es.CAMERAS.get(args.camera)
    gsd = {s_: es.gsd_m(alt[s_], sizes[s_][0], cam) for s_ in gts if s_ in alt and alt[s_] > 0} if (alt and cam) else {}
    sites = HE.site_metrics(gts, dets, conf, sizes=sizes, gsd=gsd, **site_cfg)
    print(f"\n  SITES (survey view, class-agnostic, link {sites['link_m']:g} m"
          f"{'' if not sites['frames_without_altitude'] else ', %d frames without altitude used %g px' % (sites['frames_without_altitude'], site_cfg.get('link_px_if_no_altitude', 60))}): "
          f"{sites['sites_found']}/{sites['n_sites']} sites found (recall {sites['site_recall']:.3f}) | "
          f"false alerts {sites['false_alerts']} ({sites['false_alerts_per_frame']:.3f} per frame)")

    # ─────────── bootstrap 95% CIs (image-level) ───────────
    ci = {}
    if args.bootstrap > 0:
        t_b = time.time()
        ci = HE.bootstrap_ci(gts, dets, HE.DEFAULT_CI_SPECS, conf, n_boot=args.bootstrap,
                             seed=args.bootstrap_seed, site_cfg=site_cfg, gsd=gsd)
        print(f"\n  95% CI (image-level bootstrap, {args.bootstrap} resamples, seed {args.bootstrap_seed}, "
              f"@conf {conf:.2f}; {time.time() - t_b:.0f}s)")
        for k, c in ci.items():
            rng_ = "n/a" if c["lo"] is None else f"[{c['lo']:.4f}, {c['hi']:.4f}]"
            print(f"    {k:<14}{c['point']:>9.4f}  {rng_}")
    ci_cfg = {"n_boot": args.bootstrap, "seed": args.bootstrap_seed}

    # ─────────── pre-registered targets ───────────
    prereg = None
    names, sel_note = [], ""
    if tspec:
        try:
            names, sel_note = HE.select_sections(tspec["spec"], args.targets_section, args.targets_dataset)
        except ValueError as e:
            print(f"\n  PRE-REGISTERED TARGETS skipped: {e}")
            tspec = None
    if tspec:
        secs = tspec["spec"]["sections"]
        print(f"\n  PRE-REGISTERED TARGETS  {Path(tspec['file']).name}  sha256 {tspec['sha256'][:12]}...  "
              f"status: {tspec['status']}"
              + (f" (locked {tspec['lock']['frozen_utc']})" if tspec['lock'] else "") + f"  [{sel_note}]")
        if tspec["status"] != "FROZEN":
            print("    ⚠  not frozen / changed after freezing: these PASS/FAIL lines are NOT pre-registered results")
        prereg = {"file": tspec["file"], "sha256": tspec["sha256"], "status": tspec["status"],
                  "lock": tspec["lock"], "selection": sel_note,
                  "schema_version": HE.schema_version(tspec["spec"]), "sections": {}}
        for nm in names:
            res = HE.check_targets(secs[nm], gts, dets, conf, alt=alt, sizes=sizes, gsd=gsd, site_cfg=site_cfg,
                                   scope_asserted=args.targets_section != "auto", ci=ci_cfg)
            prereg["sections"][nm] = res
            print(f"    [{nm}] {secs[nm].get('scope', '')} -- {res[0]['scope'] if res else ''}")
            for r_ in res:
                lim = []
                if r_.get("min_px"):
                    lim.append(f">={r_['min_px']:g}px")
                if r_.get("classes"):
                    lim.append(str(r_["classes"]))
                if r_.get("level", "fine") != "fine":
                    lim.append(r_["level"])
                v = r_["value_measured"]
                verdict = ("no verdict (scope unknown)" if v is not None else "n/a") if r_["pass"] is None \
                    else ("PASS" if r_["pass"] else "FAIL")
                c_ = r_.get("ci95") or {}
                ci_txt = f"  95% CI [{c_['lo']:.4f}, {c_['hi']:.4f}]" if c_.get("lo") is not None else ""
                print(f"      {r_['id']:<4}{r_['metric']:<24}{' '.join(lim):<18}{r_['op']:>3} {r_['value']:<6}"
                      f"{'' if v is None else f'{v:.4f}':>9}  {verdict}{ci_txt}"
                      + (f"  ({r_['error']})" if r_.get("error") else ""))
        if args.conf is None:
            print("    ⚠  threshold calibrated on THIS set: operating-point targets are optimistic. "
                  "Calibrate on VAL and pass --conf.")

    # ─────────── targets ───────────
    chk = {"map5095": (r["map"], r["map"] >= TARGETS["map5095"]),
           "recall": (op["recall"], op["recall"] >= TARGETS["recall"]),
           "fp_share": (op["fp_share"], op["fp_share"] < TARGETS["fp_share"])}
    chk_env = {"map5095": (re_["map"], re_["map"] >= TARGETS["map5095"]),
               "recall": (ope["recall"], ope["recall"] >= TARGETS["recall"]),
               "fp_share": (ope["fp_share"], ope["fp_share"] < TARGETS["fp_share"])}
    print("\n  Brief's single-number targets (legacy reading, kept for comparison):  mAP@.5:.95 >= 0.68 | "
          "recall >= 0.87 | FP < 0.05 (as FP/(TP+FP))")
    print(f"    {'':<9} {'all objects':>16} {'envelope >= %gpx' % args.assess_px:>20}")
    for k in chk:
        (v, ok), (ve, oke) = chk[k], chk_env[k]
        print(f"    {k:<9} {v:>8.4f} {'PASS' if ok else 'FAIL':>6}  {ve:>12.4f} {'PASS' if oke else 'FAIL':>6}")

    report = {"data_dir": str(data_dir), "n_frames": len(imgs), "n_objects": n_gt,
              "weights": args.weights or args.onnx, "map5095": r["map"], "map50": r["map50"],
              "map75": r["map75"], "conf": conf, "operating_point": {
                  k: v for k, v in op.items() if k not in ("tp", "fp", "fn")},
              "per_class_ap": {CLASSES[c]: float(r["per_class_ap"][c]) for c in range(nc) if r["n_gt"][c]},
              "per_class_n": {CLASSES[c]: int(r["n_gt"][c]) for c in range(nc)},
              "ci95": ci, "bootstrap": {"n_boot": args.bootstrap, "seed": args.bootstrap_seed,
                                        "unit": "image", "method": "percentile"},
              "by_size": by_size, "by_altitude": by_alt,
              "calibration": ({k: v for k, v in cal.items() if k != "operating_point"} if cal else None),
              "mean_seconds_per_frame": float(np.mean(list(timing.values()))) if timing else None,
              "targets": {k: {"value": v, "pass": bool(ok)} for k, (v, ok) in chk.items()},
              "envelope": envelope,
              "targets_envelope": {k: {"value": v, "pass": bool(ok)} for k, (v, ok) in chk_env.items()},
              "hierarchy": hier, "group_ap": grp_ap, "tier1": t1, "size_x_altitude": matrix,
              "sites": {k: v for k, v in sites.items() if k != "per_frame"},
              "preregistered_targets": prereg}
    (out / "eval_report.json").write_text(json.dumps(report, indent=2, default=_json))
    with open(out / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "n_gt", "ap5095", "ap50", f"recall@{conf:.2f}"])
        w.writerows(rows)
    print(f"\n  Wrote {out / 'eval_report.json'} and per_class.csv\n")


if __name__ == "__main__":
    main()
