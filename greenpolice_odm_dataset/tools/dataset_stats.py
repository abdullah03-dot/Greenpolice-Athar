#!/usr/bin/env python3
"""
Dataset statistics for the paper (ATHAR Task 7, added 2026-10-08) -> results_bundle/dataset_stats.json.

Everything is COUNTED from files on disk; nothing is estimated.
  renders      <synthetic_dir>/<env>/scene_index.jsonl + labels/  (and any --extra_render_dirs, e.g.
               the validation sequences): scenes and frames per environment / altitude profile / capture
               mode, labelled instances per class (total and per environment), objects per native
               pixel-size bucket, negative-scene share, renderer 'visible but unlabelled' counts
               (generation_stats_*.json)
  tiles        <tiled_dir>/tile_index.jsonl (03b): tiles per split, empty (negative) tiles
  splits       data/splits/split_manifest.json: frames / scenes per split
  assets       assets/models/<class>/: Objaverse models (reviewed = listed in a review round's
               index.json, else UNREVIEWED), procedural models (proc_* / generated piles and cables),
               rejected (assets/models/_rejected/<class>); distractors
  licences     ground textures (asset.json / manifest), HDRIs (manifest), procedural (generated here),
               Objaverse per-object licence from the LOCAL metadata cache only (no network; uids not in
               the cache are reported as unknown, never guessed)

Usage
  python tools/dataset_stats.py --synthetic_dir data/raw/synthetic --extra_render_dirs data/raw/synthetic_valseq \\
      --tiled_dirs data/tiled/main --splits_manifest data/splits/split_manifest.json --out results_bundle/dataset_stats.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ewads_metrics as M     # noqa: E402
import ewads_taxonomy as tx   # noqa: E402

UID_RE = re.compile(r"^[0-9a-f]{32}$")
MODEL_EXT = {".glb", ".obj", ".fbx"}


def size_bucket(px: float) -> str:
    for lab, (lo, hi) in M.SIZE_BUCKETS.items():
        if lo <= px < hi:
            return lab
    return f"<{min(lo for lo, _ in M.SIZE_BUCKETS.values()):g}"


def render_stats(render_dirs: list) -> dict:
    scenes, frames = {}, 0
    by = {k: Counter() for k in ("environment", "altitude_profile", "capture_mode")}
    frames_by = {k: Counter() for k in by}
    inst, inst_env, sizes, sizes_cls = Counter(), defaultdict(Counter), Counter(), defaultdict(Counter)
    neg_frames, empty_label_frames = 0, 0
    unl = Counter()
    sources = {}
    for root in render_dirs:
        root = Path(root)
        env_dirs = sorted(p.parent for p in root.glob("*/scene_index.jsonl")) + \
            ([root] if (root / "scene_index.jsonl").exists() else [])
        for d in env_dirs:
            idx = d / "scene_index.jsonl"
            sources[str(idx)] = hashlib.sha256(idx.read_bytes()).hexdigest()
            for f in d.glob("generation_stats_*.json"):
                for c, n in json.loads(f.read_text(encoding="utf-8")).get("visible_unlabelled_per_class", {}).items():
                    unl[c] += n
            for line in idx.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                frames += 1
                key = (str(root), r["environment"], r["sequence_id"] if r.get("sequence_id") is not None else r["scene_id"])
                if key not in scenes:
                    scenes[key] = r
                    for k in by:
                        by[k][str(r.get(k, "unknown"))] += 1
                for k in frames_by:
                    frames_by[k][str(r.get(k, "unknown"))] += 1
                neg_frames += bool(r.get("negative"))
                W, H = r.get("resolution", [None, None])
                lbl = d / "labels" / (Path(r["image_path"].replace("\\", "/")).stem + ".txt")
                lines = [l.split() for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()] if lbl.exists() else []
                empty_label_frames += not lines
                for p in lines:
                    c = tx.CLASSES[int(float(p[0]))]
                    inst[c] += 1
                    inst_env[r["environment"]][c] += 1
                    if W and H:
                        b = size_bucket(max(float(p[3]) * W, float(p[4]) * H))
                        sizes[b] += 1
                        sizes_cls[c][b] += 1
    n_sc = len(scenes)
    n_neg_sc = sum(bool(r.get("negative")) for r in scenes.values())
    return {"index_sha256": sources, "n_scenes": n_sc, "n_frames": frames,
            "scenes_by": {k: dict(v) for k, v in by.items()}, "frames_by": {k: dict(v) for k, v in frames_by.items()},
            "negative_scenes": n_neg_sc, "negative_scene_share": (n_neg_sc / n_sc) if n_sc else None,
            "frames_without_labels": empty_label_frames,
            "instances_per_class": {c: inst[c] for c in tx.CLASSES}, "instances_total": sum(inst.values()),
            "instances_per_class_by_environment": {e: {c: v[c] for c in tx.CLASSES} for e, v in sorted(inst_env.items())},
            "objects_by_size_bucket_px": dict(sizes),
            "objects_by_size_bucket_px_per_class": {c: dict(v) for c, v in sorted(sizes_cls.items())},
            "visible_but_unlabelled_per_class": {c: unl[c] for c in tx.CLASSES} if unl else None,
            "size_note": "bucket = native long side of the YOLO box in pixels (ewads_metrics.SIZE_BUCKETS)"}


def tile_stats(tiled_dirs: list) -> dict:
    out = {}
    for td in tiled_dirs:
        td = Path(td)
        ti = td / "tile_index.jsonl"
        if not ti.exists():
            out[str(td)] = {"error": "no tile_index.jsonl"}
            continue
        per = defaultdict(Counter)
        for line in ti.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            per[r.get("split", "unknown")]["tiles"] += 1
            per[r.get("split", "unknown")]["empty_tiles"] += int(r.get("n_labels", r.get("n_boxes", 1)) == 0)
        out[str(td)] = {s: dict(c) for s, c in sorted(per.items())}
    return out


def _review_stems(models_dir: Path) -> set:
    stems = set()
    for idx in models_dir.glob("_review*/index.json"):
        for tiles in json.loads(idx.read_text(encoding="utf-8")).values():
            stems.update(tiles.values())
    return stems


def asset_stats(models_dir: Path, distractors_dir: Path) -> dict:
    reviewed = _review_stems(models_dir)
    per, objaverse_uids = {}, set()
    for c in tx.CLASSES:
        d = models_dir / c
        files = [p for p in d.iterdir() if p.suffix.lower() in MODEL_EXT] if d.exists() else []
        proc = [p for p in files if p.stem.startswith(("proc_", "ewaste_pile_", "cable_bundle_"))]
        objv = [p for p in files if UID_RE.match(p.stem)]
        other = [p for p in files if p not in proc and p not in objv]
        rej_dir = models_dir / "_rejected" / c
        rej = [p for p in rej_dir.iterdir() if p.suffix.lower() in MODEL_EXT] if rej_dir.exists() else []
        objaverse_uids.update(p.stem for p in objv)
        per[c] = {"total": len(files), "objaverse_reviewed": sum(p.stem in reviewed for p in objv),
                  "objaverse_UNREVIEWED": sum(p.stem not in reviewed for p in objv),
                  "procedural": len(proc), "other": len(other), "rejected": len(rej)}
    dis = Counter()
    dis_uids = set()
    if distractors_dir.exists():
        for p in distractors_dir.rglob("*"):
            if p.suffix.lower() in MODEL_EXT:
                dis[p.parent.name] += 1
                if UID_RE.match(p.stem):
                    dis_uids.add(p.stem)
    return {"per_class": per, "totals": {k: sum(v[k] for v in per.values()) for k in next(iter(per.values()))},
            "distractors_per_category": dict(dis), "distractors_total": sum(dis.values()),
            "_objaverse_uids": sorted(objaverse_uids | dis_uids)}


def objaverse_licences(uids: list, metadata_dir: Path, snapshot: Optional[Path] = None) -> dict:
    """Per-object licence from the LOCAL Objaverse metadata cache (*.json.gz), then from a CSV snapshot
    kept in the repo for uids the cache lacks (the cache lives on the machine that fetched). No network."""
    want, found = set(uids), {}
    if metadata_dir.exists():
        for gz in sorted(metadata_dir.glob("*.json.gz")):
            if not want - set(found):
                break
            try:
                data = json.loads(gzip.open(gz, "rt", encoding="utf-8").read())
            except Exception:
                continue
            for u in want & set(data):
                a = data[u] or {}
                user = a.get("user") or {}
                found[u] = {"license": a.get("license") or "unspecified in metadata", "name": a.get("name"),
                            "creator": user.get("displayName") or user.get("username"),
                            "creator_url": user.get("profileUrl"), "url": a.get("viewerUrl") or a.get("uri")}
    n_cache = len(found)
    if snapshot is not None and snapshot.exists() and want - set(found):
        import csv
        for r in csv.DictReader(open(snapshot, encoding="utf-8")):    # travels with the repo (see --licence_snapshot)
            if r["uid"] in want and r["uid"] not in found:
                found[r["uid"]] = {k: r.get(k) or None for k in ("license", "name", "creator", "creator_url", "url")}
    counts = Counter(v["license"] for v in found.values())
    missing = sorted(want - set(found))
    if missing:
        counts["unknown (not in local metadata cache)"] = len(missing)
    nc = sorted(u for u, v in found.items() if "nc" in str(v["license"]).lower().split("-"))
    return {"source": str(metadata_dir), "n_from_local_cache": n_cache,
            "n_from_snapshot": len(found) - n_cache, "snapshot": str(snapshot) if snapshot else None,
            "n_models": len(want), "by_licence": dict(counts),
            "non_commercial_uids": nc, "missing_uids": missing,
            "note": "CC BY* requires attribution (see the *_objaverse_attributions.csv next to this file); "
                    "NC licences forbid commercial use -- check with counsel before deployment",
            "_attributions": found}


def licence_summary(assets_root: Path, uids: list, objaverse_meta: Path, snapshot: Optional[Path] = None) -> dict:
    out = {}
    tex = assets_root / "textures"
    tex_lic = Counter(json.loads(p.read_text(encoding="utf-8")).get("license", "unspecified")
                      for p in tex.rglob("asset.json")) if tex.exists() else Counter()
    out["ground_textures"] = {"n": sum(tex_lic.values()), "by_licence": dict(tex_lic),
                              "manifests": [str(p) for p in tex.rglob("manifest.json")] if tex.exists() else []}
    hm = assets_root / "haven" / "manifest.json"
    if hm.exists():
        h = json.loads(hm.read_text(encoding="utf-8"))
        n = sum(len(v) for v in (h.get("environments") or {}).values())
        out["hdris"] = {"n": n, "source": h.get("source"), "manifest": str(hm)}
    else:
        out["hdris"] = {"n": 0, "note": "no assets/haven/manifest.json"}
    out["procedural_models"] = "generated by tools/generate_procedural_models.py / generate_piles.py (no third-party licence)"
    out["objaverse_models_and_distractors"] = objaverse_licences(uids, objaverse_meta, snapshot)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic_dir", default="data/raw/synthetic")
    ap.add_argument("--extra_render_dirs", nargs="*", default=[])
    ap.add_argument("--tiled_dirs", nargs="*", default=None, help="default: every data/tiled/* with tile_index.jsonl")
    ap.add_argument("--splits_manifest", default="data/splits/split_manifest.json")
    ap.add_argument("--assets_dir", default="assets")
    ap.add_argument("--objaverse_metadata_dir", default=str(Path.home() / ".objaverse" / "hf-objaverse-v1" / "metadata"))
    ap.add_argument("--licence_snapshot", default="assets/models/_licences/objaverse_attributions.csv",
                    help="CSV (uid,license,name,creator,creator_url,url) used for uids not in the local cache")
    ap.add_argument("--out", default="results_bundle/dataset_stats.json")
    a = ap.parse_args(argv)

    assets = Path(a.assets_dir)
    tiled = [t for t in a.tiled_dirs if t] if a.tiled_dirs is not None else \
        sorted(str(p.parent) for p in Path("data/tiled").glob("*/tile_index.jsonl"))
    ast_ = asset_stats(assets / "models", assets / "distractors")
    uids = ast_.pop("_objaverse_uids")
    sm = Path(a.splits_manifest) if a.splits_manifest else None
    if sm is not None and not sm.is_file():
        sm = None
    stats = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "renders": {"main": render_stats([a.synthetic_dir]),
                    **({"extra": render_stats(a.extra_render_dirs)} if a.extra_render_dirs else {})},
        "tiles": tile_stats(tiled),
        "splits": ({k: {kk: v[kk] for kk in ("n_frames", "n_scenes", "by_environment", "n_negative_frames")}
                    for k, v in json.loads(sm.read_text(encoding="utf-8"))["splits"].items()} if sm else None),
        "splits_manifest_sha256": hashlib.sha256(sm.read_bytes()).hexdigest() if sm else None,
        "assets": ast_,
        "licences": licence_summary(assets, uids, Path(a.objaverse_metadata_dir),
                                    Path(a.licence_snapshot) if a.licence_snapshot else None),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    attrib = stats["licences"]["objaverse_models_and_distractors"].pop("_attributions")
    att_path = out.with_name(out.stem + "_objaverse_attributions.csv")
    import csv
    with open(att_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uid", "license", "name", "creator", "creator_url", "url"])
        for u in sorted(attrib):
            v = attrib[u]
            w.writerow([u, v["license"], v["name"], v["creator"], v["creator_url"], v["url"]])
    stats["licences"]["objaverse_models_and_distractors"]["attributions_csv"] = str(att_path)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    r = stats["renders"]["main"]
    print(f"\n  renders: {r['n_scenes']} scenes / {r['n_frames']} frames, {r['instances_total']} labelled instances, "
          f"negative-scene share {r['negative_scene_share'] if r['negative_scene_share'] is None else round(r['negative_scene_share'], 4)}")
    print(f"  scenes by environment: {r['scenes_by']['environment']}")
    t = stats["assets"]["totals"]
    print(f"  assets: {t}  distractors {stats['assets']['distractors_total']}")
    if t.get("objaverse_UNREVIEWED"):
        print(f"  ⚠  {t['objaverse_UNREVIEWED']} Objaverse models are in no review round's index.json")
    ob = stats["licences"]["objaverse_models_and_distractors"]
    print(f"  licences (Objaverse, local cache): {ob['by_licence']}")
    if ob["non_commercial_uids"]:
        print(f"  ⚠  {len(ob['non_commercial_uids'])} models are NonCommercial-licensed (listed in the JSON)")
    print(f"  attributions -> {ob['attributions_csv']}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
