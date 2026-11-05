#!/usr/bin/env python3
"""
Synthetic-only splits (ATHAR, added 2026-10-08).

From the renderer's  <synthetic_dir>/<environment>/scene_index.jsonl  this builds

  main/                     train on all three environments
    synthetic/  images/ labels/            TRAIN  (named 'synthetic' so 03b/03 pick it up as-is)
    val/        images/ labels/ scene_index.jsonl   synthetic VALIDATION: checkpoint choice, AdaIN arm
                                                    choice, 08 --calibrate, 10 --gate_data
    heldout/    images/ labels/ scene_index.jsonl   synthetic HELD-OUT test of record (never tiled,
                                                    never used for any choice) -> targets synthetic_heldout
    heldout_seq/ (optional, --heldout_seq_dir)     held-out sequences: 11 scored for evidence targets
    val_seq/    (optional, --val_seq_dir)           overlapping synthetic SEQUENCES for 11's
                                                    --stage_thresholds (rendered with 02 --sequence_frames)
  loeo_<env>/               leave-one-environment-out, one per environment
    synthetic/  = main TRAIN scenes of the other two environments
    val/        = main VAL scenes of the other two environments
    heldout/    = ALL scenes of <env>             -> targets sim_to_sim_loeo

Splitting is by SCENE (by sequence for sequence renders: every frame of a sequence goes to the same
split), stratified by environment x altitude profile, with a fixed seed.  Tiles follow their source
frame because 03b tiles each split folder separately.  Files are hard-linked (no extra disk) with a
copy fallback.  split_manifest.json records the parameters, the SHA-256 of every source index, per-split
counts, and per split a file list (lists/<split>.txt: "<sha256>  <relative path>") plus the SHA-256 of
that list.  A scene in two splits is a hard error.

Usage
  python tools/make_synth_splits.py --synthetic_dir data/raw/synthetic --out_dir data/splits
  python tools/make_synth_splits.py ... --val_seq_dir data/raw/synthetic_valseq     # + 11 sequences
Then e.g.
  python 03b_tile_dataset.py --input_dir data/splits/main --output_dir data/tiled/main --splits synthetic val
  python 08_tiled_eval.py --data_dir data/splits/main/heldout --meta data/splits/main/heldout/scene_index.jsonl ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ENVIRONMENTS = ("desert", "european_urban", "european_rural")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_index(env_dir: Path) -> list:
    """Records of one environment, with image/label paths re-anchored to env_dir (the index stores
    absolute paths of the render machine)."""
    recs = []
    for line in (env_dir / "scene_index.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        name = Path(r["image_path"].replace("\\", "/")).name
        r["_img"] = env_dir / "images" / name
        r["_lbl"] = env_dir / "labels" / (Path(name).stem + ".txt")
        r["_env_dir"] = env_dir.name
        if not r["_img"].exists():
            raise SystemExit(f"[ERROR] {r['_img']} listed in {env_dir}/scene_index.jsonl but missing")
        if not r["_lbl"].exists():
            raise SystemExit(f"[ERROR] label {r['_lbl']} missing (empty file = negative, absent = error)")
        recs.append(r)
    return recs


def group_key(r: dict) -> tuple:
    """Unit that must never be split: a sequence, else a scene."""
    sid = r.get("sequence_id")
    return (r["environment"], int(sid if sid is not None else r["scene_id"]))


def assign(groups: dict, val_frac: float, heldout_frac: float, seed: int) -> dict:
    """{group_key: 'train'|'val'|'heldout'}, stratified by environment x altitude profile."""
    strata = defaultdict(list)
    for k, recs in groups.items():
        strata[(k[0], recs[0].get("altitude_profile", "unknown"))].append(k)
    rng = random.Random(seed)
    out = {}
    for st in sorted(strata):
        keys = sorted(strata[st])
        rng.shuffle(keys)
        n = len(keys)
        n_ho = math.ceil(n * heldout_frac) if n >= 3 else 0
        n_va = math.ceil(n * val_frac) if n >= 3 else 0
        for i, k in enumerate(keys):
            out[k] = "heldout" if i < n_ho else ("val" if i < n_ho + n_va else "train")
    return out


def place(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def write_split(name: str, recs: list, root: Path, out: Path, mode: str, hash_files: bool,
                lists_dir: Path) -> dict:
    d = out / name
    stems = Counter(r["_img"].stem for r in recs)
    dup = [s for s, c in stems.items() if c > 1]
    if dup:
        raise SystemExit(f"[ERROR] split {name}: duplicate image names {dup[:5]} (tiles would overwrite)")
    lines, how = [], Counter()
    for r in sorted(recs, key=lambda r: r["_img"].name):
        for src, sub in ((r["_img"], "images"), (r["_lbl"], "labels")):
            dst = d / sub / src.name
            how[place(src, dst, mode)] += 1
            rel = dst.relative_to(root).as_posix()
            lines.append(f"{sha256_file(src) if hash_files else '-'}  {rel}")
    with open(d / "scene_index.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(recs, key=lambda r: r["_img"].name):
            rr = {k: v for k, v in r.items() if not k.startswith("_")}
            rr["image_path"] = f"images/{r['_img'].name}"
            rr["label_path"] = f"labels/{r['_lbl'].name}"
            f.write(json.dumps(rr) + "\n")
    lists_dir.mkdir(parents=True, exist_ok=True)
    lst = lists_dir / f"{name.replace('/', '__')}.txt"
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    groups = {group_key(r) for r in recs}
    return {"dir": d.relative_to(root).as_posix(), "n_frames": len(recs), "n_scenes": len(groups),
            "n_negative_frames": sum(bool(r.get("negative")) for r in recs),
            "by_environment": dict(Counter(r["environment"] for r in recs)),
            "by_altitude_profile": dict(Counter(r.get("altitude_profile", "unknown") for r in recs)),
            "by_capture_mode": dict(Counter(r.get("capture_mode", "unknown") for r in recs)),
            "file_list": lst.relative_to(root).as_posix(), "file_list_sha256": sha256_file(lst),
            "files_hashed": hash_files, "placement": dict(how)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic_dir", default="data/raw/synthetic",
                    help="renderer output: <dir>/<environment>/scene_index.jsonl + images/ + labels/")
    ap.add_argument("--val_seq_dir", default=None,
                    help="optional sequence render(s) for 11 (02 --sequence_frames N): <dir>/<env>/... or <dir>/...; "
                         "copied whole to main/val_seq")
    ap.add_argument("--heldout_seq_dir", default=None,
                    help="optional HELD-OUT sequence render (distinct --start_id) -> main/heldout_seq: 11 is scored "
                         "on it for the synthetic_heldout evidence targets (never used for any choice)")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--heldout_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    ap.add_argument("--no_hash_files", action="store_true", help="skip per-file SHA-256 (list hash still written)")
    ap.add_argument("--no_loeo", action="store_true")
    a = ap.parse_args(argv)

    syn, out = Path(a.synthetic_dir), Path(a.out_dir)
    env_dirs = sorted(p.parent for p in syn.glob("*/scene_index.jsonl"))
    if (syn / "scene_index.jsonl").exists():
        env_dirs.append(syn)
    if not env_dirs:
        raise SystemExit(f"[ERROR] no scene_index.jsonl under {syn}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"[ERROR] {out} is not empty -- splits are never rewritten in place; choose a new --out_dir")
    recs = [r for d in env_dirs for r in load_index(d)]
    groups = defaultdict(list)
    for r in recs:
        groups[group_key(r)].append(r)
    asg = assign(groups, a.val_frac, a.heldout_frac, a.seed)

    split_recs = {"main/synthetic": [], "main/val": [], "main/heldout": []}
    for k, rs in groups.items():
        split_recs[{"train": "main/synthetic", "val": "main/val", "heldout": "main/heldout"}[asg[k]]] += rs
    envs = sorted({r["environment"] for r in recs})
    if not a.no_loeo:
        for e in envs:
            split_recs[f"loeo_{e}/synthetic"] = [r for k, rs in groups.items() if k[0] != e and asg[k] == "train" for r in rs]
            split_recs[f"loeo_{e}/val"] = [r for k, rs in groups.items() if k[0] != e and asg[k] == "val" for r in rs]
            split_recs[f"loeo_{e}/heldout"] = [r for k, rs in groups.items() if k[0] == e for r in rs]
    # sequence renders (02 --sequence_frames N), each copied whole: val_seq = 11's stage thresholds
    # (selection); heldout_seq = 11 scored for the synthetic_heldout evidence targets (reporting only)
    seen_keys = set(groups)
    for split, src in (("main/val_seq", a.val_seq_dir), ("main/heldout_seq", a.heldout_seq_dir)):
        if not src:
            continue
        vs = Path(src)
        vdirs = sorted(p.parent for p in vs.glob("*/scene_index.jsonl")) + ([vs] if (vs / "scene_index.jsonl").exists() else [])
        vrecs = [r for d in vdirs for r in load_index(d)]
        clash = {group_key(r) for r in vrecs} & seen_keys
        if clash:
            raise SystemExit(f"[ERROR] {split} scene ids also in another render {sorted(clash)[:5]}: render the "
                             f"sequences with a distinct --start_id")
        seen_keys |= {group_key(r) for r in vrecs}
        split_recs[split] = vrecs

    # leakage check: within a family (main / each loeo fold) no group in two of its splits
    fams = defaultdict(dict)
    for name, rs in split_recs.items():
        fams[name.split("/")[0]][name] = {group_key(r) for r in rs}
    for fam, sp in fams.items():
        names = list(sp)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                both = sp[names[i]] & sp[names[j]]
                if both:
                    raise SystemExit(f"[ERROR] LEAKAGE: {len(both)} scenes in both {names[i]} and {names[j]}")

    out.mkdir(parents=True, exist_ok=True)
    lists = out / "lists"
    summary = {}
    for name, rs in split_recs.items():
        if not rs:
            print(f"  [WARN] split {name} is empty")
        summary[name] = write_split(name, rs, out, out, a.mode, not a.no_hash_files, lists)
        s = summary[name]
        print(f"  {name:<28} {s['n_frames']:>6} frames  {s['n_scenes']:>5} scenes  {s['by_environment']}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "tools/make_synth_splits.py", "tool_sha256": sha256_file(Path(__file__)),
        "params": {"val_frac": a.val_frac, "heldout_frac": a.heldout_frac, "seed": a.seed, "mode": a.mode,
                   "unit": "scene (sequence for sequence renders)",
                   "stratified_by": ["environment", "altitude_profile"]},
        "sources": {str(d / "scene_index.jsonl"): sha256_file(d / "scene_index.jsonl") for d in env_dirs},
        "n_frames_total": len(recs), "n_scenes_total": len(groups), "environments": envs,
        "splits": summary,
        "usage": {"model_selection": "main/val (and main/val_seq for 11) -- never main/heldout or any test set",
                  "report": "main/heldout -> targets synthetic_heldout; loeo_<env>/heldout -> sim_to_sim_loeo"},
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n  manifest: {out / 'split_manifest.json'}")
    return manifest


if __name__ == "__main__":
    main()
