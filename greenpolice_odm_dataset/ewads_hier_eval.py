"""
EWADS evaluation extensions used by 08_tiled_eval.py (and testable without a model).

  * HIERARCHY   the same detections scored at three levels (ewads_taxonomy):
                  fine   18 classes                      ("is it a CRT or a flat screen?")
                  group  8 groups                        ("is it a display?")
                  binary 1 class                         ("is it e-waste?")
  * SITES       the survey question: was each DUMP SITE flagged, and how many false alerts per
                frame?  Ground-truth objects closer than `link_m` form one site; a site is found
                when any detection (any class) touches any of its objects.  False detections close
                together count as ONE false alert (that is what an operator experiences).
  * MATRIX      mAP by altitude band x object size, so "0.68" can never hide where it comes from.
  * TARGETS     pre-registered targets (targets_v1.json), checked against a freeze lock
                (tools/freeze_targets.py) so it is visible whether they were changed after results.

All functions take 08's structures: gts {stem: (cls (n,), boxes (n,4))}, dets {stem: (m,>=6)}.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import ewads_metrics as M
import ewads_taxonomy as tx
import ewads_tiling as et


# ───────────────────────────── hierarchy ─────────────────────────────
def to_level(gts: dict, dets: dict, level: str, dedup_iou: float = 0.7) -> Tuple[dict, dict, int]:
    """Map fine ids to `level`.  Detections are de-duplicated inside each mapped class with plain IoU
    NMS (dedup_iou): a model that puts a 'crt_monitor' AND a 'flat_screen' box on one display has
    found ONE display, not one display and one false positive.  Nested boxes are NOT suppressed
    (a cpu_chip on a circuit_board is two board_chip objects)."""
    if level == "fine":
        return gts, dets, len(tx.CLASSES)
    g2 = {k: (tx.map_ids(c, level), b) for k, (c, b) in gts.items()}
    d2 = {}
    for k, d in dets.items():
        d = np.asarray(d, float)[:, :6].copy()
        if len(d):
            d[:, 5] = tx.map_ids(d[:, 5].astype(int), level)
            d = et.nms_classwise(d, iou_thr=dedup_iou, ios_thr=1.01, max_pre=len(d) + 1)
        d2[k] = d
    return g2, d2, tx.n_classes(level)


def class_subset(spec, level: str) -> Optional[List[int]]:
    """'tier1' | list of names at `level` | None (all)."""
    if spec in (None, "all"):
        return None
    names = tx.level_names(level)
    if spec == "tier1":
        if level != "fine":
            raise ValueError("'tier1' is defined on fine classes")
        return [names.index(c) for c in sorted(tx.TIER_1)]
    return [names.index(c) for c in spec]


def metric(gts, dets, *, metric: str, level: str = "fine", conf: Optional[float] = None,
           min_px: Optional[float] = None, max_px: Optional[float] = None, classes=None,
           image_ids: Optional[Sequence[str]] = None, site_cfg: Optional[dict] = None,
           sizes: Optional[dict] = None, gsd: Optional[dict] = None) -> Tuple[float, dict]:
    """One number + details.  metric: map5095 | map50 | map75 | recall | precision | fp_share |
    fp_per_image | site_recall | false_alerts_per_frame.  Size limits use the native long side
    (operating-point metrics support min_px only, as COCO-style ignore)."""
    g, d, nc = to_level(gts, dets, level)
    ids = list(image_ids) if image_ids is not None else sorted(g)
    sub = class_subset(classes, level)
    n_frames = len(ids)
    if metric in ("map5095", "map50", "map75"):
        rng = None if (min_px is None and max_px is None) else (min_px or 0.0, max_px or 1e9)
        r = M.evaluate(g, d, nc, size_range=rng, image_ids=ids)
        if sub is None:
            v = {"map5095": r["map"], "map50": r["map50"], "map75": r["map75"]}[metric]
        else:
            col = {"map5095": None, "map50": 0, "map75": 5}[metric]
            vals = [(np.nanmean(r["ap"][c]) if col is None else r["ap"][c, col]) for c in sub if r["n_gt"][c] > 0]
            v = float(np.mean(vals)) if vals else float("nan")
        return float(v), {"n_gt": int(r["n_gt"].sum() if sub is None else r["n_gt"][sub].sum()), "n_frames": n_frames}
    if conf is None:
        raise ValueError(f"{metric} needs an operating threshold (conf)")
    if metric in ("recall", "precision", "fp_share", "fp_per_image"):
        op = M.operating_point(g, d, nc, conf, image_ids=ids, min_px=min_px)
        if sub is None:
            v = op[metric]
            tp, fp, fn = int(op["tp"].sum()), int(op["fp"].sum()), int(op["fn"].sum())
        else:
            tp, fp, fn = int(op["tp"][sub].sum()), int(op["fp"][sub].sum()), int(op["fn"][sub].sum())
            v = {"recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
                 "fp_share": fp / max(tp + fp, 1), "fp_per_image": fp / max(n_frames, 1)}[metric]
        return float(v), {"tp": tp, "fp": fp, "fn": fn, "n_frames": n_frames}
    if metric in ("site_recall", "false_alerts_per_frame"):
        s = site_metrics(gts, dets, conf, image_ids=ids, sizes=sizes, gsd=gsd, **(site_cfg or {}))
        return float(s[metric]), {k: v for k, v in s.items() if k != "per_frame"}
    raise ValueError(metric)


# ───────────────────────────── bootstrap CIs ─────────────────────────────
def bootstrap_ci(gts, dets, specs: List[dict], conf: Optional[float], *, n_boot: int = 1000, seed: int = 2026,
                 image_ids: Optional[Sequence[str]] = None, site_cfg: Optional[dict] = None,
                 gsd: Optional[dict] = None, alpha: float = 0.05) -> Dict[str, dict]:
    """Image-level non-parametric bootstrap (2026-10-08).  Each resample draws len(ids) IMAGES with
    replacement (a duplicated image counts twice: it gets a distinct key), every spec is recomputed
    with metric() -- the same code as the point value -- and the percentile interval is returned.
    spec = {"name", "metric", "level"?, "classes"?, "min_px"?, "max_px"?}.
    -> {name: {"point", "lo", "hi", "n_boot", "n_boot_valid", "n_images", "seed", "method"}}"""
    ids = list(image_ids) if image_ids is not None else sorted(gts)
    empty = np.zeros((0, 6), np.float32)

    def one(g, d, ids_, gs, s):
        return metric(g, d, metric=s["metric"], level=s.get("level", "fine"), conf=conf, min_px=s.get("min_px"),
                      max_px=s.get("max_px"), classes=s.get("classes"), image_ids=ids_, site_cfg=site_cfg,
                      gsd=gs)[0]

    out = {s["name"]: {"point": one(gts, dets, ids, gsd, s), "samples": []} for s in specs}
    if not ids:
        return {k: {"point": v["point"], "lo": None, "hi": None, "n_boot": 0, "n_boot_valid": 0, "n_images": 0,
                    "seed": seed, "method": "image-level percentile bootstrap"} for k, v in out.items()}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        g2, d2, gs2 = {}, {}, {}
        for j, k in enumerate(pick):
            key = f"{ids[k]}#b{j}"
            g2[key] = gts[ids[k]]
            d2[key] = dets.get(ids[k], empty)
            if gsd and ids[k] in gsd:
                gs2[key] = gsd[ids[k]]
        keys = list(g2)
        for s in specs:
            v = one(g2, d2, keys, gs2, s)
            if v is not None and np.isfinite(v):
                out[s["name"]]["samples"].append(v)
    res = {}
    for name, o in out.items():
        smp = np.asarray(o["samples"], float)
        res[name] = {"point": o["point"],
                     "lo": float(np.percentile(smp, 100 * alpha / 2)) if len(smp) else None,
                     "hi": float(np.percentile(smp, 100 * (1 - alpha / 2))) if len(smp) else None,
                     "n_boot": n_boot, "n_boot_valid": int(len(smp)), "n_images": len(ids), "seed": seed,
                     "method": "image-level percentile bootstrap"}
    return res


def paired_bootstrap_diff(gts, dets_a, dets_b, specs: List[dict], conf_a: Optional[float],
                          conf_b: Optional[float], *, n_boot: int = 1000, seed: int = 2026,
                          image_ids: Optional[Sequence[str]] = None, site_cfg: Optional[dict] = None,
                          gsd: Optional[dict] = None, alpha: float = 0.05) -> Dict[str, dict]:
    """Two models scored on the SAME images (2026-10-08): each resample draws one set of images and
    evaluates both models on it, so the CI is for the difference b - a (paired), not two separate CIs.
    Each model keeps its own operating threshold (conf_a / conf_b, both fixed on validation).
    -> {name: {"a", "b", "diff", "diff_lo", "diff_hi", "p_diff_le_0", ...}}"""
    ids = list(image_ids) if image_ids is not None else sorted(gts)
    empty = np.zeros((0, 6), np.float32)

    def one(g, d, ids_, gs, s, conf):
        return metric(g, d, metric=s["metric"], level=s.get("level", "fine"), conf=conf, min_px=s.get("min_px"),
                      max_px=s.get("max_px"), classes=s.get("classes"), image_ids=ids_, site_cfg=site_cfg,
                      gsd=gs)[0]

    pts = {s["name"]: (one(gts, dets_a, ids, gsd, s, conf_a), one(gts, dets_b, ids, gsd, s, conf_b)) for s in specs}
    diffs = {s["name"]: [] for s in specs}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot if ids else 0):
        pick = rng.integers(0, len(ids), len(ids))
        g2, da, db, gs2 = {}, {}, {}, {}
        for j, k in enumerate(pick):
            key = f"{ids[k]}#b{j}"
            g2[key], da[key], db[key] = gts[ids[k]], dets_a.get(ids[k], empty), dets_b.get(ids[k], empty)
            if gsd and ids[k] in gsd:
                gs2[key] = gsd[ids[k]]
        keys = list(g2)
        for s in specs:
            va, vb = one(g2, da, keys, gs2, s, conf_a), one(g2, db, keys, gs2, s, conf_b)
            if np.isfinite(va) and np.isfinite(vb):
                diffs[s["name"]].append(vb - va)
    out = {}
    for name, (va, vb) in pts.items():
        d = np.asarray(diffs[name], float)
        out[name] = {"a": va, "b": vb, "diff": vb - va,
                     "diff_lo": float(np.percentile(d, 100 * alpha / 2)) if len(d) else None,
                     "diff_hi": float(np.percentile(d, 100 * (1 - alpha / 2))) if len(d) else None,
                     "p_diff_le_0": float((d <= 0).mean()) if len(d) else None,
                     "n_boot": n_boot, "n_boot_valid": int(len(d)), "n_images": len(ids), "seed": seed,
                     "method": "paired image-level percentile bootstrap"}
    return out


DEFAULT_CI_SPECS = [
    {"name": "map50", "metric": "map50"},
    {"name": "map5095", "metric": "map5095"},
    {"name": "recall", "metric": "recall"},
    {"name": "tier1_recall", "metric": "recall", "classes": "tier1"},
    {"name": "fp_share", "metric": "fp_share"},
    {"name": "site_recall", "metric": "site_recall"},
]


# ───────────────────────────── sites ─────────────────────────────
def _components(boxes: np.ndarray, link_px: float) -> List[List[int]]:
    """connected components of boxes whose gap (box-to-box distance) <= link_px."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            gx = max(0.0, max(boxes[i, 0], boxes[j, 0]) - min(boxes[i, 2], boxes[j, 2]))
            gy = max(0.0, max(boxes[i, 1], boxes[j, 1]) - min(boxes[i, 3], boxes[j, 3]))
            if np.hypot(gx, gy) <= link_px:
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _touch(a: np.ndarray, b: np.ndarray, margin: float) -> np.ndarray:
    """(n,4) x (m,4) -> bool: boxes intersect after growing a by margin."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), bool)
    return ((a[:, None, 0] - margin) <= b[None, :, 2]) & ((a[:, None, 2] + margin) >= b[None, :, 0]) & \
           ((a[:, None, 1] - margin) <= b[None, :, 3]) & ((a[:, None, 3] + margin) >= b[None, :, 1])


def site_metrics(gts, dets, conf: float, image_ids=None, link_m: float = 3.0,
                 link_px_if_no_altitude: float = 60.0, margin_px: float = 4.0,
                 sizes: Optional[dict] = None, gsd: Optional[dict] = None) -> dict:
    """Survey-level scoring (class-agnostic).  gsd: {stem: metres per pixel} -> link_m converted per
    frame; frames without it use link_px_if_no_altitude."""
    ids = list(image_ids) if image_ids is not None else sorted(gts)
    n_sites = found = alerts = 0
    per_frame = {}
    used_fallback = 0
    for im in ids:
        _, gb = gts.get(im, (np.zeros(0, int), np.zeros((0, 4))))
        d = np.asarray(dets.get(im, np.zeros((0, 6))), float)
        d = d[d[:, 4] >= conf][:, :4] if len(d) else np.zeros((0, 4))
        g_ = (gsd or {}).get(im)
        link = link_m / g_ if g_ else link_px_if_no_altitude
        used_fallback += int(not g_)
        sites = _components(gb, link) if len(gb) else []
        hit = _touch(d, gb, margin_px)                    # det x gt
        f_found = sum(bool(hit[:, s].any()) for s in sites) if len(d) else 0
        false_d = d[~hit.any(1)] if len(d) else d
        f_alerts = len(_components(false_d, link)) if len(false_d) else 0
        n_sites += len(sites); found += f_found; alerts += f_alerts
        per_frame[im] = {"sites": len(sites), "found": f_found, "false_alerts": f_alerts}
    return {"site_recall": found / n_sites if n_sites else float("nan"), "n_sites": n_sites,
            "sites_found": found, "false_alerts": alerts,
            "false_alerts_per_frame": alerts / max(len(ids), 1), "n_frames": len(ids),
            "frames_without_altitude": used_fallback, "link_m": link_m, "per_frame": per_frame}


# ───────────────────────────── altitude x size ─────────────────────────────
def size_altitude_matrix(gts, dets, alt: Dict[str, float], bands, buckets=None, level="fine") -> dict:
    buckets = buckets or M.SIZE_BUCKETS
    g, d, nc = to_level(gts, dets, level)
    out = {}
    for lo, hi in bands:
        ids = [s for s in g if s in alt and lo <= alt[s] < hi]
        if not ids:
            continue
        lab = f"{lo:g}-{hi:g}m" if hi < 1e8 else f">{lo:g}m"
        row = {}
        for bl, rng in buckets.items():
            r = M.evaluate(g, d, nc, size_range=rng, image_ids=ids)
            if r["n_gt"].sum():
                row[bl] = {"map5095": r["map"], "map50": r["map50"], "n_gt": int(r["n_gt"].sum())}
        out[lab] = {"n_frames": len(ids), "cells": row}
    return out


# ───────────────────────────── pre-registered targets ─────────────────────────────
def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_targets(path) -> dict:
    """-> {"spec", "sha256", "lock": {...} | None, "status": FROZEN | NOT FROZEN | MODIFIED AFTER FREEZE}"""
    p = Path(path)
    spec = json.loads(p.read_text())
    sha = sha256_file(p)
    lock_p = p.with_suffix(".lock.json")
    lock = json.loads(lock_p.read_text()) if lock_p.exists() else None
    if lock is None:
        status = "NOT FROZEN"
    elif lock.get("targets_sha256") != sha:
        status = "MODIFIED AFTER FREEZE"
    elif lock.get("taxonomy_sha256") != tx.taxonomy_sha256():
        status = "TAXONOMY MODIFIED AFTER FREEZE"
    else:
        status = "FROZEN"
    return {"spec": spec, "sha256": sha, "lock": lock, "status": status, "file": str(p)}


def selection_data_problem(path) -> Optional[str]:
    """Model-selection guard (synthetic-only regime, 2026-10-08): thresholds, gates, checkpoints and
    AdaIN arms are chosen on synthetic VALIDATION only.  -> reason string if `path` looks like a
    held-out / test set (a 'test*' or '*heldout*' folder among its last three path parts), else None."""
    parts = [x.lower() for x in Path(path).resolve().parts[-3:]]
    hit = [x for x in parts if x.startswith("test") or "heldout" in x]
    if hit:
        return (f"{path} looks like a held-out/test set ({hit}): model selection (thresholds, INT8 gate, "
                f"stage thresholds) must use synthetic VALIDATION, e.g. data/splits/main/val")
    return None


METRICS_08 = {"map5095", "map50", "map75", "recall", "precision", "fp_share", "fp_per_image",
              "site_recall", "false_alerts_per_frame"}
EVALUATOR_08 = "08_tiled_eval.py"
OPS = (">=", ">", "<=", "<")


def schema_version(spec: dict) -> int:
    return int(spec.get("schema_version", 1))


def validate_spec(spec: dict) -> List[str]:
    """-> list of problems (empty = valid).  v1: id/metric/op/value per target.  v2 additionally:
    every section has dataset + evaluated_by; every target has scope + why; ids unique; sections
    scored by 08 use only metrics 08 can compute."""
    errs, seen = [], set()
    v = schema_version(spec)
    secs = spec.get("sections")
    if not isinstance(secs, dict) or not secs:
        return ["no 'sections'"]
    need_t = {"id", "metric", "op", "value"} | ({"scope", "why"} if v >= 2 else set())
    for nm, sec in secs.items():
        if v >= 2:
            for k in ("dataset", "evaluated_by", "targets"):
                if k not in sec:
                    errs.append(f"section {nm}: missing '{k}'")
        for t in sec.get("targets", []):
            miss = need_t - set(t)
            if miss:
                errs.append(f"section {nm} target {t.get('id', '?')}: missing {sorted(miss)}")
            if t.get("op") not in OPS:
                errs.append(f"section {nm} target {t.get('id', '?')}: op {t.get('op')!r} not in {OPS}")
            if t.get("id") in seen:
                errs.append(f"duplicate target id {t.get('id')}")
            seen.add(t.get("id"))
            if sec.get("evaluated_by", EVALUATOR_08) == EVALUATOR_08 and t.get("metric") not in METRICS_08:
                errs.append(f"section {nm} target {t.get('id')}: metric {t.get('metric')!r} is not computable by 08")
    return errs


def select_sections(spec: dict, section: str = "auto", dataset: Optional[str] = None) -> Tuple[List[str], str]:
    """Which sections 08 scores -> (names, note).
    v1: 'auto' = every section; else the named one.
    v2: only sections evaluated by 08 and whose 'dataset' == `dataset` (required: scoring sim_to_real
    targets on synthetic data, or the reverse, would be meaningless); `section` may name one of them.
    Raises ValueError on an impossible selection."""
    secs = spec["sections"]
    if schema_version(spec) < 2:
        if section == "auto":
            return list(secs), "all sections"
        if section not in secs:
            raise ValueError(f"unknown section {section!r}; have {list(secs)}")
        return [section], f"section {section} (asserted)"
    mine = {n: s for n, s in secs.items() if s.get("evaluated_by") == EVALUATOR_08}
    datasets = sorted({s["dataset"] for s in mine.values()})
    if not dataset:
        raise ValueError(f"{spec.get('id', 'targets')} is schema v2: pass --targets_dataset "
                         f"(one of {datasets}) so targets are scored only on the data they were set for")
    if dataset not in datasets:
        raise ValueError(f"--targets_dataset {dataset!r} not in {datasets}")
    names = [n for n, s in mine.items() if s["dataset"] == dataset]
    if section != "auto":
        if section not in names:
            raise ValueError(f"section {section!r} is not a {dataset} section; have {names}")
        return [section], f"dataset {dataset}, section {section} (asserted)"
    return names, f"dataset {dataset}"


def check_targets(section: dict, gts, dets, conf, *, alt: Optional[dict] = None, sizes=None, gsd=None,
                  site_cfg=None, scope_asserted: bool = False, ci: Optional[dict] = None) -> List[dict]:
    """Evaluate one section's targets on the frames inside its altitude scope.  Without altitudes the
    scope cannot be enforced: values are still computed, but pass=None ('scope unknown') unless the
    operator asserted the scope (08 --targets_section <name>), which the result records."""
    ids = sorted(gts)
    scope_note = "all frames"
    if section.get("altitude_m"):
        lo, hi = section["altitude_m"]
        if alt:
            ids = [s for s in ids if s in alt and lo <= alt[s] < hi]
            scope_note = f"{len(ids)} frames with altitude in [{lo}, {hi}) m"
        elif scope_asserted:
            scope_note = (f"altitude unknown: scope [{lo}, {hi}) m ASSERTED by the operator "
                          f"(--targets_section), all {len(ids)} frames used")
        else:
            scope_note = f"ALTITUDE UNKNOWN: scope [{lo}, {hi}) m NOT enforced -- no verdict"
    res = []
    for t in section["targets"]:
        if "scope" in t:   # v2: keep the declared scope; "scope" below is the scope actually applied
            t = {**t, "scope_declared": t["scope"]}
        if not ids:
            res.append({**t, "value_measured": None, "pass": None, "scope": scope_note})
            continue
        try:
            v, det = metric(gts, dets, metric=t["metric"], level=t.get("level", "fine"), conf=conf,
                            min_px=t.get("min_px"), max_px=t.get("max_px"), classes=t.get("classes"),
                            image_ids=ids, site_cfg=site_cfg, sizes=sizes, gsd=gsd)
        except ValueError as e:
            res.append({**t, "value_measured": None, "pass": None, "error": str(e), "scope": scope_note})
            continue
        ok = None if not np.isfinite(v) else (v >= t["value"] if t["op"] == ">=" else
                                              v > t["value"] if t["op"] == ">" else
                                              v <= t["value"] if t["op"] == "<=" else v < t["value"])
        if section.get("altitude_m") and not alt and not scope_asserted:
            ok = None
        row = {**t, "value_measured": v, "pass": ok, "details": det, "scope": scope_note}
        if ci and ci.get("n_boot", 0) > 0:      # CI on exactly this target's frames / filters
            spec = {k: t[k] for k in ("metric", "level", "classes", "min_px", "max_px") if k in t}
            row["ci95"] = bootstrap_ci(gts, dets, [{"name": "v", **spec}], conf, n_boot=ci["n_boot"],
                                       seed=ci.get("seed", 2026), image_ids=ids, site_cfg=site_cfg, gsd=gsd)["v"]
        res.append(row)
    return res
