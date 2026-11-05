#!/usr/bin/env python3
"""
Results bundle for the paper (ATHAR Task 9, added 2026-10-08).

Collects every result file named in tools/results_bundle_config.json into results_bundle/, writes the
figures, environment.txt, git_commit.txt (or a code SHA-256 manifest), SUMMARY.md and MANIFEST.sha256.

Rules (enforced here):
  * Nothing is computed or estimated.  Every number in SUMMARY.md is copied from a source file, and
    every verdict (PASS / FAIL / no verdict) is the one that source computed; each row names its
    source file.  The only logic applied is the pre-registered 'each_fold' rule for LOEO (all folds
    must pass), and the runbook's pre-stated AdaIN rule (higher mAP@.5 on synthetic validation).
  * A missing input is reported as NOT RUN (and listed under 'Missing inputs'), never filled in.
  * No model is run.  The real-test figure only draws detections the final evaluation already saved.
  * The targets' freeze status heads the summary; an unfrozen source is flagged.

Usage
  python tools/export_results_bundle.py                      # config: tools/results_bundle_config.json
  python tools/export_results_bundle.py --config <json> --bundle results_bundle
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import ewads_hier_eval as HE    # noqa: E402
import ewads_survey as es       # noqa: E402
import ewads_taxonomy as tx     # noqa: E402
import ewads_tiling as et       # noqa: E402

TIER1_BGR, OTHER_BGR, GT_BGR = (52, 104, 235), (214, 120, 42), (235, 235, 235)   # #eb6834, #2a78d6 (BGR)
CODE_GLOBS = ["*.py", "ewads_*.py", "tools/*.py", "tools/*.json", "tools/reviews/*.json", "tests/*.py",
              "targets_*.json", "environment*.yml"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(p: Optional[Path]):
    return json.loads(p.read_text(encoding="utf-8")) if p and p.is_file() else None


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "NaN" if f != f else f"{f:.{nd}f}"


def ci_txt(c: Optional[dict]) -> str:
    if not c or c.get("lo") is None:
        return "—"
    return f"[{fmt(c['lo'])}, {fmt(c['hi'])}]"


def verdict(p) -> str:
    return "no verdict" if p is None else ("PASS" if p else "FAIL")


class Bundle:
    def __init__(self, root: Path, bundle: Path, cfg: dict):
        self.root, self.dir, self.cfg = root, bundle, cfg
        self.missing, self.sources = [], {}

    def path(self, rel: Optional[str]) -> Optional[Path]:
        return (self.root / rel) if rel else None

    def take(self, key: str, rel: Optional[str], sub: str = "sources") -> Optional[Path]:
        """Copy a source into the bundle (unless it already lives there) -> its bundle path."""
        p = self.path(rel)
        if not p or not p.is_file():
            self.missing.append(f"{key}: {rel}")
            return None
        try:
            p.resolve().relative_to(self.dir.resolve())
            dst = p                                         # already inside the bundle
        except ValueError:
            dst = (self.dir / p.name) if sub == "." else (self.dir / sub / key.replace(":", "_") / p.name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        self.sources[key] = dst
        return dst

    def rel(self, p: Path) -> str:
        return p.resolve().relative_to(self.dir.resolve()).as_posix()


# ───────────────────────── figures ─────────────────────────
def _label(img, x, y, text, col):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y0 = max(th + 4, y)
    cv2.rectangle(img, (x, y0 - th - 4), (x + tw + 4, y0), (20, 20, 20), -1)
    cv2.putText(img, text, (x + 2, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def _wrap(text, width_px, scale=0.55):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width_px - 16 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def _band(img, lines_spec):
    """Caption band ABOVE the image (never over the pixels or boxes).
    lines_spec: [(text, bgr), ...]; long lines wrap."""
    rows = [(l, col) for text, col in lines_spec for l in _wrap(text, img.shape[1])]
    h = 10 + 24 * len(rows)
    band = np.full((h, img.shape[1], 3), 20, np.uint8)
    for i, (l, col) in enumerate(rows):
        cv2.putText(band, l, (8, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    return np.vstack([band, img])


def _fit(img, width=1600):
    return img if img.shape[1] <= width else cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])),
                                                       interpolation=cv2.INTER_AREA)


def fig_synthetic(b: Bundle, out: Path) -> list:
    d = b.path(b.cfg["figures"].get("synthetic_heldout_dir"))
    idx = d / "scene_index.jsonl" if d else None
    if not idx or not idx.is_file():
        b.missing.append("figure synthetic GT overlays: no scene_index.jsonl in synthetic_heldout_dir")
        return []
    recs = [json.loads(l) for l in idx.read_text(encoding="utf-8").splitlines() if l.strip()]
    made = []
    for env in ("desert", "european_urban", "european_rural"):
        cand = sorted((r for r in recs if r["environment"] == env and not r.get("negative")),
                      key=lambda r: (-int(r.get("n_annotations", 0)), r["image_path"]))
        if not cand:
            b.missing.append(f"figure synthetic {env}: no frame")
            continue
        r = cand[0]
        ip = d / r["image_path"]
        img = cv2.imread(str(ip))
        if img is None:
            b.missing.append(f"figure synthetic {env}: unreadable {ip}")
            continue
        H, W = img.shape[:2]
        cls, boxes = et.yolo_to_xyxy(et.read_yolo(d / r["label_path"]), W, H)
        lw = max(2, W // 900)
        for c, bx in zip(cls, boxes):
            name = tx.CLASSES[int(c)]
            col = TIER1_BGR if name in tx.TIER_1 else OTHER_BGR
            x1, y1, x2, y2 = (int(v) for v in bx)
            cv2.rectangle(img, (x1 - lw, y1 - lw), (x2 + lw, y2 + lw), col, lw)
            _label(img, x1, y1 - lw - 2, name, col)
        img = _band(_fit(img), [(f"synthetic held-out | {env} | {r.get('altitude_m')} m | {r.get('capture_mode', '')} | "
                                 f"ground truth ({len(cls)} objects; orange = tier-1) [{Path(r['image_path']).name}]",
                                 (245, 245, 245))])
        p = out / f"synthetic_{env}_ground_truth.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        made.append(p)
    return made


def fig_real(b: Bundle, out: Path) -> Optional[Path]:
    rt = b.cfg["figures"].get("real_test") or {}
    data, cache, rep = b.path(rt.get("data_dir")), b.path(rt.get("cache")), load_json(b.path(rt.get("report")))
    if not (data and cache and cache.is_file() and rep):
        b.missing.append("figure real test frame: final real-test evaluation (cache + report) not found")
        return None
    conf = rep.get("conf")
    with np.load(cache, allow_pickle=True) as z:
        dets = {k: z[k] for k in z.files if k != "__timing__"}
    stem = max(sorted(dets), key=lambda k: int((dets[k][:, 4] >= conf).sum()) if len(dets[k]) else 0)
    ip = next((p for p in (data / "images").iterdir() if p.stem == stem), None)
    img = cv2.imread(str(ip)) if ip else None
    if img is None:
        b.missing.append(f"figure real test frame: image {stem} not found")
        return None
    H, W = img.shape[:2]
    lw = max(2, W // 900)
    lp = data / "labels" / f"{stem}.txt"
    if lp.exists():
        for bx in et.yolo_to_xyxy(et.read_yolo(lp), W, H)[1]:
            x1, y1, x2, y2 = (int(v) for v in bx)
            cv2.rectangle(img, (x1, y1), (x2, y2), GT_BGR, 1)
    for x1, y1, x2, y2, s, c in dets[stem][dets[stem][:, 4] >= conf]:
        name = tx.CLASSES[int(c)]
        col = TIER1_BGR if name in tx.TIER_1 else OTHER_BGR
        cv2.rectangle(img, (int(x1) - lw, int(y1) - lw), (int(x2) + lw, int(y2) + lw), col, lw)
        _label(img, int(x1), int(y1) - lw - 2, f"{name} {s:.2f}", col)
    hrow = None
    hc = b.path(rt.get("heights_csv"))
    if hc and hc.is_file():
        hrow = next((r for r in csv.DictReader(open(hc, encoding="utf-8")) if r["stem"] == stem), None)
    gsd = None
    if hrow:
        import gsd_validation as GV
        f_px, _, _, _ = GV.focal_px(ip, W, H, (hrow.get("camera") or "exif").strip(), None)
        gsd = float(hrow["height_m"]) / f_px if f_px else None
    # same assessability statement as 07's forensic overlay (ewads_survey.resolvability at this GSD)
    cls_ = es.resolvability_summary(gsd)
    na = cls_["not_assessable"] + cls_["unknown"]
    assess = (f"GSD {gsd * 100:.2f} cm/px" if gsd else "GSD unknown") + \
             (f" | NOT assessable: {', '.join(na)}" if na else " | all classes assessable")
    img = _band(_fit(img), [(f"real test | {stem} | predictions @ conf {fmt(conf, 3)} (val-calibrated) | thin white = "
                             f"ground truth | orange = tier-1" + (f" | height {hrow['height_m']} m" if hrow else
                                                                   " | height unknown"), (245, 245, 245)),
                            (assess, (0, 165, 255))])
    p = out / "real_test_predictions_assessability.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return p


def fig_evidence(b: Bundle, out: Path) -> Optional[Path]:
    pat = b.cfg["figures"].get("evidence_crops_glob")
    hits = sorted(glob.glob(str(b.root / pat))) if pat else []
    if not hits:
        b.missing.append(f"figure evidence crop: nothing matches {pat}")
        return None
    p = out / ("evidence_review_crop" + Path(hits[0]).suffix)
    shutil.copy2(hits[0], p)
    (out / "evidence_review_crop.source.txt").write_text(hits[0] + "\n", encoding="utf-8")
    return p


# ───────────────────────── environment / code ─────────────────────────
def environment_txt(b: Bundle) -> Path:
    from importlib import metadata
    lines = [f"created_utc: {datetime.now(timezone.utc).isoformat()}",
             f"python: {sys.version.split()[0]}", f"platform: {platform.platform()}"]
    for pkg in ("torch", "torchvision", "ultralytics", "albumentations", "opencv-python", "opencv-python-headless",
                "numpy", "onnx", "onnxruntime-gpu", "onnxruntime", "blenderproc", "cryptography", "pycocotools"):
        try:
            lines.append(f"{pkg}: {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{pkg}: not installed")
    try:
        import torch
        lines.append(f"cuda (torch build): {torch.version.cuda}")
        lines.append(f"cudnn: {torch.backends.cudnn.version()}")
        for i in range(torch.cuda.device_count()):
            lines.append(f"gpu[{i}]: {torch.cuda.get_device_name(i)}  capability {torch.cuda.get_device_capability(i)}")
        if not torch.cuda.device_count():
            lines.append("gpu: none visible to torch")
    except Exception as e:
        lines.append(f"torch/cuda: unavailable ({type(e).__name__})")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        lines += [f"nvidia-smi: {l}" for l in smi.stdout.strip().splitlines()] or ["nvidia-smi: no output"]
    except Exception:
        lines.append("nvidia-smi: not available")
    blender = sorted((Path.home() / "blender").glob("blender-*")) if (Path.home() / "blender").exists() else []
    lines.append(f"blender (BlenderProc install): {', '.join(p.name for p in blender) or 'not found on this machine'}")
    eng = load_json(b.path(b.cfg["decision_records"].get("engine_decision")))
    jet = (eng or {}).get("env") or (eng or {}).get("environment") or (eng or {}).get("jetson")
    lines.append(f"tensorrt / jetpack (from engine_decision.json): {json.dumps(jet) if jet else 'not available (no engine_decision.json)'}")
    p = b.dir / "environment.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def code_manifest(b: Bundle) -> Path:
    if (b.root / ".git").exists():
        try:
            head = subprocess.run(["git", "-C", str(b.root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(b.root), "status", "--porcelain"], capture_output=True, text=True).stdout
            p = b.dir / "git_commit.txt"
            p.write_text(f"commit: {head}\nuncommitted changes: {'yes' if dirty.strip() else 'no'}\n{dirty}", encoding="utf-8")
            return p
        except Exception:
            pass
    files = sorted({p for g in CODE_GLOBS for p in b.root.glob(g) if p.is_file()})
    man = b.dir / "code_manifest.sha256"
    man.write_text("".join(f"{sha256_file(p)}  {p.relative_to(b.root).as_posix()}\n" for p in files), encoding="utf-8")
    p = b.dir / "git_commit.txt"
    p.write_text(f"not a git repository; code identified by code_manifest.sha256 ({len(files)} files)\n"
                 f"code_manifest.sha256 sha256: {sha256_file(man)}\n", encoding="utf-8")
    return p


# ───────────────────────── SUMMARY.md ─────────────────────────
def target_rows(rep: dict, src: str) -> list:
    pr = (rep or {}).get("preregistered_targets") or {}
    rows = []
    for sec, res in (pr.get("sections") or {}).items():
        for t in res:
            rows.append({"section": sec, "id": t["id"], "metric": t["metric"],
                         "scope": t.get("scope_declared") or t.get("scope", ""), "op": t["op"], "target": t["value"],
                         "measured": t.get("value_measured"), "ci": t.get("ci95"), "pass": t.get("pass"),
                         "applied_scope": t.get("scope", ""), "src": src})
    return rows


def summary_md(b: Bundle, freeze: dict) -> str:
    L = ["# ATHAR results summary", "",
         f"Generated {datetime.now(timezone.utc).isoformat()} by `tools/export_results_bundle.py`. Every number below is "
         "copied from the named source file; every verdict is the one that file computed. Nothing is estimated; "
         "missing inputs are marked **NOT RUN**.", ""]
    L += ["## Pre-registration", "",
          f"- Targets file: `{freeze['file']}` sha256 `{freeze['sha256']}` — status **{freeze['status']}**"
          + (f" (locked {freeze['lock'].get('frozen_utc')} by {freeze['lock'].get('frozen_by') or 'unnamed'})" if freeze.get("lock") else ""),
          f"- Freeze record: {'`freeze_record.txt`' if freeze.get('record') else '**NOT FOUND**'}"]
    if freeze["status"] != "FROZEN":
        L.append("- ⚠ **The targets are not frozen (or changed after freezing): no PASS/FAIL below is a pre-registered result.**")
    L.append("")

    # H1
    L += ["## H1 — detection (synthetic-only training)", ""]
    evals = b.cfg["evals"]
    headline = ["| run | frames | objects | mAP@.5 [95% CI] | mAP@.5:.95 [95% CI] | recall [95% CI] | tier-1 recall [95% CI] "
                "| FP share [95% CI] | site recall [95% CI] | conf | targets status | source |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    trows, fold_rows = [], {}
    for e in evals:
        p = b.sources.get(f"eval:{e['name']}")
        rep = load_json(p)
        if not rep:
            headline.append(f"| {e['name']} | NOT RUN | | | | | | | | | | `{e['report']}` |")
            continue
        src = b.rel(p)
        ci = rep.get("ci95") or {}
        def cell(k):
            c = ci.get(k)
            return f"{fmt(c['point'])} {ci_txt(c)}" if c else "—"
        st = (rep.get("preregistered_targets") or {}).get("status", "no targets")
        headline.append(f"| {e['name']}{' (baseline)' if 'baseline' in e['hypothesis'] else ''} | {rep.get('n_frames')} | "
                        f"{rep.get('n_objects')} | {cell('map50')} | {cell('map5095')} | {cell('recall')} | {cell('tier1_recall')} | "
                        f"{cell('fp_share')} | {cell('site_recall')} | {fmt(rep.get('conf'), 3)} | {st} | `{src}` |")
        for r in target_rows(rep, src):
            r["run"] = e["name"]
            trows.append(r)
            if e.get("fold"):
                fold_rows.setdefault(r["id"], []).append(r)
    L += headline + ["", "Pre-registered targets (verdict as computed by 08 on the target's own scope):", "",
                     "| run | section | id | metric | declared scope | target | measured | 95% CI | verdict | source |",
                     "|---|---|---|---|---|---|---|---|---|---|"]
    if not trows:
        L.append("| NOT RUN | | | | | | | | | |")
    for r in trows:
        L.append(f"| {r['run']} | {r['section']} | {r['id']} | {r['metric']} | {r['scope']} | {r['op']} {r['target']} | "
                 f"{fmt(r['measured'])} | {ci_txt(r['ci'])} | {verdict(r['pass'])} | `{r['src']}` |")
    n_folds = sum(1 for e in evals if e.get("fold"))
    if n_folds:
        L += ["", f"LOEO 'each_fold' rule (targets_v2: every one of the {n_folds} folds must pass):", "",
              "| id | folds with a verdict | all folds PASS |", "|---|---|---|"]
        for tid, rs in sorted(fold_rows.items()):
            got = [r["pass"] for r in rs]
            allp = "incomplete" if len(got) < n_folds or any(g is None for g in got) else ("PASS" if all(got) else "FAIL")
            L.append(f"| {tid} | {sum(g is not None for g in got)}/{n_folds} | {allp} |")
        if not fold_rows:
            L.append("| NOT RUN | | |")
    L.append("")

    # decision records
    L += ["### Decision records", ""]
    dr = b.cfg["decision_records"]
    a_no, a_yes = load_json(b.sources.get("decision:adain_no")), load_json(b.sources.get("decision:adain_yes"))
    if a_no and a_yes:
        ch = "AdaIN" if a_yes["map50"] > a_no["map50"] else "no AdaIN"
        L.append(f"- AdaIN arm (rule: higher mAP@.5 on synthetic validation): no-AdaIN {fmt(a_no['map50'])} "
                 f"(`{b.rel(b.sources['decision:adain_no'])}`), AdaIN {fmt(a_yes['map50'])} "
                 f"(`{b.rel(b.sources['decision:adain_yes'])}`) → **{ch}**")
    else:
        L.append(f"- AdaIN arm: **NOT RUN** (`{dr.get('adain_no')}`, `{dr.get('adain_yes')}`)")
    ab = load_json(b.sources.get("decision:ablation_camera"))
    if ab:
        L += [f"- Camera-mismatch ablation (`{b.rel(b.sources['decision:ablation_camera'])}`, paired bootstrap, "
              f"{ab['n_images']} images, difference = {ab['difference']}):", "",
              f"  | metric | {ab['a']['name']} | {ab['b']['name']} | diff | 95% CI of diff |", "  |---|---|---|---|---|"]
        for k, r in ab["metrics"].items():
            ci = "—" if r["diff_lo"] is None else f"[{fmt(r['diff_lo'])}, {fmt(r['diff_hi'])}]"
            L.append(f"  | {k} | {fmt(r['a'])} | {fmt(r['b'])} | {fmt(r['diff'])} | {ci} |")
    else:
        L.append(f"- Camera-mismatch ablation: **NOT RUN** (`{dr.get('ablation_camera')}`)")
    eng = load_json(b.sources.get("decision:engine_decision"))
    L.append(f"- INT8 vs FP16 (`{b.rel(b.sources['decision:engine_decision'])}`):\n\n  ```json\n  "
             + json.dumps(eng, indent=1, default=str).replace("\n", "\n  ") + "\n  ```" if eng else
             f"- INT8 vs FP16: **NOT RUN** (`{dr.get('engine_decision')}`)")
    jb = load_json(b.sources.get("decision:jetson_benchmark"))
    L.append(f"- Jetson benchmark (`{b.rel(b.sources['decision:jetson_benchmark'])}`): latency section:\n\n  ```json\n  "
             + json.dumps(jb.get("latency"), indent=1, default=str).replace("\n", "\n  ") + "\n  ```" if jb else
             f"- Jetson benchmark: **NOT RUN** (`{dr.get('jetson_benchmark')}`)")
    L.append("")

    # H2
    L += ["## H2 — ground-sampling-distance / assessability geometry", ""]
    h2 = load_json(b.sources.get("h2"))
    if h2:
        src = b.rel(b.sources["h2"])
        s = h2["summary"]["all"]
        L += [f"Source `{src}` — {s.get('n', 0)} photos scored, {len(h2.get('skipped', []))} skipped; "
              f"|rel. error| median {fmt(s.get('median'))}, p95 {fmt(s.get('p95'))}, max {fmt(s.get('max'))}; "
              f"oblique-flagged {h2['summary'].get('n_oblique_flagged')}.", "",
              "| id | metric | aggregate | target | measured | n | verdict | targets status |", "|---|---|---|---|---|---|---|---|"]
        t = h2.get("targets") or {}
        for c in t.get("checks", []):
            L.append(f"| {c['id']} | {c['metric']} | {c['aggregate']} | {c['op']} {c['target']} | {fmt(c['measured'])} | "
                     f"{c['n']} | {verdict(c['pass'])} | {t.get('freeze_status')} |")
        for m, c in (h2.get("implied_video_crop") or {}).items():
            L.append(f"\n- {m}: implied sensor-width fraction read out {fmt(c['implied_width_fraction_read_out_median'])} "
                     f"(n={c['n']}, configured {c['configured_width_crop']})")
    else:
        L.append(f"**NOT RUN** (`{b.cfg.get('h2')}`)")
    L.append("")

    # H3
    L += ["## H3 — tamper evidence of the audit log", ""]
    h3 = load_json(b.sources.get("h3"))
    if h3:
        src = b.rel(b.sources["h3"])
        L += [f"Source `{src}` — {h3['n_trials_per_type']} trials per tamper type, source: {h3['source']['kind']}.", "",
              "| tamper | " + " | ".join(h3["levels"]) + " |", "|---|" + "---|" * len(h3["levels"])]
        for kind, row in h3["results"].items():
            L.append(f"| {kind} | " + " | ".join(
                "n/a" if not row[lv]["n"] else f"{row[lv]['detected']}/{row[lv]['n']} ({fmt(row[lv]['rate'], 3)})"
                for lv in h3["levels"]) + " |")
        t = h3.get("targets") or {}
        L += ["", "| id | tamper | check | target | measured | n | verdict | targets status |", "|---|---|---|---|---|---|---|---|"]
        for c in t.get("checks", []):
            L.append(f"| {c['id']} | {c['tamper']} | {c['check']} | {c['op']} {c['target']} | {fmt(c['measured'])} | "
                     f"{c['n']} | {verdict(c['pass'])} | {t.get('freeze_status')} |")
    else:
        L.append(f"**NOT RUN** (`{b.cfg.get('h3')}`)")
    L.append("")

    L += ["## Missing inputs", ""] + ([f"- {m}" for m in b.missing] or ["- none"]) + [""]
    L += ["## Bundle integrity", "", "`MANIFEST.sha256` covers every file in this folder (verify: "
          "`sha256sum -c MANIFEST.sha256` from inside `results_bundle/`)."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "tools" / "results_bundle_config.json"))
    ap.add_argument("--bundle", default="results_bundle")
    ap.add_argument("--root", default=str(ROOT), help="project root the config paths are relative to")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    root, bundle = Path(a.root), Path(a.bundle)
    if not bundle.is_absolute():
        bundle = root / bundle
    bundle.mkdir(parents=True, exist_ok=True)
    b = Bundle(root, bundle, cfg)

    # freeze status
    tl = HE.load_targets(root / cfg["targets"])
    tdst = b.take("targets", cfg["targets"], sub=".")
    lock = root / cfg["targets"]
    lock = lock.with_suffix(".lock.json")
    if lock.exists():
        b.take("targets_lock", str(lock.relative_to(root)), sub=".")
    rec = b.path(cfg.get("freeze_record"))
    freeze = {"file": cfg["targets"], "sha256": tl["sha256"], "status": tl["status"], "lock": tl["lock"],
              "record": rec.is_file() if rec else False}
    if not freeze["record"]:
        b.missing.append(f"freeze record: {cfg.get('freeze_record')}")

    for e in cfg["evals"]:
        p = b.take(f"eval:{e['name']}", e["report"], sub="eval")
        if p:
            pc = b.path(e["report"]).with_name("per_class.csv")
            if pc.is_file():
                shutil.copy2(pc, p.with_name("per_class.csv"))
    for k, rel in cfg["decision_records"].items():
        b.take(f"decision:{k}", rel, sub="decisions")
    siblings = {"h2": ["h2_gsd.csv"], "h3": ["h3_tamper.csv", "h3_tamper_trials.jsonl"],
                "dataset_stats": ["dataset_stats_objaverse_attributions.csv"]}
    for k, sibs in siblings.items():
        p = b.take(k, cfg.get(k), sub=".")
        if p is None:
            continue
        src_dir = b.path(cfg[k]).parent
        for s in sibs:                                  # companions live next to the main file
            if (src_dir / s).is_file() and src_dir.resolve() != bundle.resolve():
                shutil.copy2(src_dir / s, bundle / s)
    sm = b.take("split_manifest", cfg.get("split_manifest"), sub="splits")
    if sm:
        lists = b.path(cfg["split_manifest"]).parent / "lists"
        if lists.is_dir():
            shutil.copytree(lists, sm.parent / "lists", dirs_exist_ok=True)

    figs = bundle / "figures"
    figs.mkdir(exist_ok=True)
    fig_synthetic(b, figs)
    fig_real(b, figs)
    fig_evidence(b, figs)
    h2fig = b.path(cfg.get("h2_figure"))
    if h2fig and h2fig.is_file():
        shutil.copy2(h2fig, figs / h2fig.name)
    else:
        b.missing.append(f"figure H2 scatter: {cfg.get('h2_figure')}")

    environment_txt(b)
    code_manifest(b)
    (bundle / "SUMMARY.md").write_text(summary_md(b, freeze), encoding="utf-8")

    man = bundle / "MANIFEST.sha256"
    files = sorted(p for p in bundle.rglob("*") if p.is_file() and p != man)
    man.write_text("".join(f"{sha256_file(p)}  {p.relative_to(bundle).as_posix()}\n" for p in files), encoding="utf-8")
    print(f"\n  bundle: {bundle}  ({len(files)} files + MANIFEST.sha256)")
    print(f"  targets: {freeze['status']}   missing inputs: {len(b.missing)}")
    for m in b.missing:
        print(f"    - {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
