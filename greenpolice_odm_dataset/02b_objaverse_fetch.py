"""
E-Waste Detection Pipeline — Phase 1c: Objaverse 3D Model Acquisition
======================================================================
Fetches free 3D models from Objaverse (800K+ objects) and Google Scanned
Objects for all 18 e-waste classes. This replaces Sketchfab as the
primary model source.

Why Objaverse over Sketchfab:
  • Fully scriptable — no manual browser downloads
  • 800K+ models, searchable by category and tags
  • Permissive CC licenses by default
  • Google Scanned Objects subset = photorealistic scanned real items

Output:
  assets/models/
    battery/         ← .glb files
    circuit_board/
    ...

Usage:
  conda activate ewads
  pip install objaverse trimesh

  # Fetch all classes (may take hours for full set):
  python 02b_objaverse_fetch.py --output_dir ./assets/models --n_per_class 15

  # Fetch one class:
  python 02b_objaverse_fetch.py --classes battery circuit_board --n_per_class 20

  # List what you already have:
  python 02b_objaverse_fetch.py --audit

Notes:
  - Objaverse downloads to ~/.objaverse/ cache automatically
  - Models are symlinked/copied into assets/models/<class>/
  - Script validates mesh integrity after download (rejects corrupt files)
  - Procedural pile generator creates ewaste_pile + cable_bundle assets
    that Objaverse cannot provide realistically
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
import random

# ─── Class-to-search-terms mapping ───────────────────────────────────────────
# Multiple search terms per class. Objaverse search is keyword-based.
# Broader terms first, narrow fallbacks at the end.
CLASS_SEARCH_TERMS = {
    "crt_monitor":     ["CRT monitor", "cathode ray tube television", "old TV monitor",
                        "retro television", "tube monitor"],
    "flat_screen":     ["LCD monitor", "flat screen monitor", "LED display",
                        "computer monitor", "flat panel display"],
    "circuit_board":   ["circuit board", "PCB electronics", "motherboard",
                        "printed circuit board", "electronic board"],
    "battery":         ["battery pack", "lithium battery", "car battery",
                        "battery cell", "rechargeable battery"],
    "cable_bundle":    ["cable bundle", "wire bundle", "electrical cables",
                        "computer cables", "tangled wires"],
    "single_cable":    ["electrical wire", "USB cable", "power cable",
                        "HDMI cable", "ethernet cable"],
    "mobile_phone":    ["smartphone", "mobile phone", "cell phone",
                        "old mobile phone", "feature phone"],
    "keyboard":        ["computer keyboard", "keyboard", "mechanical keyboard",
                        "laptop keyboard"],
    "printer":         ["desktop printer", "inkjet printer", "laser printer",
                        "office printer", "copier machine"],
    "small_appliance": ["toaster", "electric fan", "radio", "small appliance",
                        "electric kettle", "hair dryer"],
    "large_appliance": ["washing machine", "refrigerator", "old fridge",
                        "dishwasher", "dryer machine"],
    "cpu_chip":        ["CPU processor", "computer chip", "integrated circuit",
                        "microprocessor", "silicon chip"],
    "hard_drive":      ["hard disk drive", "HDD", "hard drive",
                        "SSD drive", "external hard drive"],
    "power_supply":    ["power supply unit", "PSU", "power adapter",
                        "transformer", "voltage converter"],
    "ewaste_pile":     [],  # Procedurally generated — see _generate_pile_assets()
    "plastic_casing":  ["plastic casing", "device enclosure", "plastic shell",
                        "computer case", "broken plastic"],
    "metal_component": ["metal heat sink", "metal chassis", "aluminum bracket",
                        "steel enclosure", "metal frame"],
    "lighting":        ["fluorescent tube", "CFL bulb", "LED strip",
                        "fluorescent lamp", "energy saving bulb"],
}

# ─── Class-to-LVIS-category mapping (--lvis mode) ────────────────────────────
# Objaverse-LVIS: ~46K objects with human-verified category labels. Far less
# noisy than free-text search (which matched e.g. a plate of cookies to
# "flat panel display"). Classes without an LVIS equivalent are absent here and
# still need keyword search. Monitors/TVs mix CRT and flat panels, so both
# classes still need a visual review pass.
CLASS_LVIS_CATEGORIES = {
    "crt_monitor":     ["television_set"],
    "flat_screen":     ["monitor_(computer_equipment) computer_monitor"],
    "battery":         ["battery", "car_battery"],
    "mobile_phone":    ["cellular_telephone"],
    "keyboard":        ["computer_keyboard"],
    "printer":         ["printer"],
    "small_appliance": ["toaster", "toaster_oven", "kettle", "teakettle", "fan",
                        "hair_dryer", "radio_receiver", "calculator", "alarm_clock",
                        "vacuum_cleaner", "blender", "mixer_(kitchen_tool)",
                        "heater", "stereo_(sound_system)", "router_(computer_equipment)",
                        "remote_control", "telephone", "camcorder", "projector",
                        "microwave_oven"],
    "large_appliance": ["automatic_washer", "refrigerator", "dishwasher",
                        "air_conditioner", "stove", "water_heater"],
    "lighting":        ["lightbulb"],
}


def check_dependencies():
    """Verify required packages are available."""
    missing = []
    try:
        import objaverse
    except ImportError:
        missing.append("objaverse")
    try:
        import trimesh
    except ImportError:
        missing.append("trimesh")

    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        sys.exit(1)


_ANNOTATIONS = None


def _load_annotations() -> dict:
    """Load all Objaverse annotations once per process (~800K entries)."""
    global _ANNOTATIONS
    if _ANNOTATIONS is None:
        import objaverse
        _ANNOTATIONS = objaverse.load_annotations()
    return _ANNOTATIONS


def search_objaverse(query: str, n: int = 50) -> list:
    """
    Search Objaverse by text query. Returns list of (uid, metadata) tuples.
    Uses objaverse's annotation metadata to filter by category.
    """
    try:
        # Load all annotations (cached on disk after first download, in memory after first call)
        annotations = _load_annotations()

        # Text search across name + description fields
        results = []
        query_lower = query.lower()

        for uid, ann in annotations.items():
            name = (ann.get("name") or "").lower()
            desc = (ann.get("description") or "").lower()
            # Objaverse tags are dicts like {"name": ..., "slug": ...}
            tags = " ".join(t.get("name", "") if isinstance(t, dict) else str(t)
                            for t in ann.get("tags") or []).lower()

            if query_lower in name or query_lower in desc or query_lower in tags:
                results.append((uid, ann))
                if len(results) >= n * 3:  # Oversample, filter later
                    break

        return results

    except Exception as e:
        print(f"  [WARN] Objaverse search failed for '{query}': {e}")
        return []


def download_objaverse_objects(uids: list, n: int) -> dict:
    """
    Download up to n objects from Objaverse by UID.
    Returns dict: {uid: local_path}
    """
    import objaverse

    # Limit to n
    selected_uids = uids[:n]

    if not selected_uids:
        return {}

    print(f"  Downloading {len(selected_uids)} objects from Objaverse...")
    # One UID at a time in-process: with a worker pool, a truncated download
    # (ContentTooShortError) can't be unpickled back to the parent and the
    # whole fetch hangs forever. Serial + try/except just skips the bad file.
    objects = {}
    for uid in selected_uids:
        try:
            objects.update(objaverse.load_objects(uids=[uid], download_processes=1))
        except Exception as e:
            print(f"    ✗ {uid[:20]}... download failed ({type(e).__name__}), skipped")
    return objects  # {uid: local_glb_path}


def validate_mesh(glb_path: str) -> bool:
    """
    Validate that a downloaded .glb file is a usable mesh.
    Rejects empty scenes, point clouds, or corrupt files.
    """
    try:
        import trimesh
        scene = trimesh.load(glb_path)

        if isinstance(scene, trimesh.Scene):
            # Multi-mesh scene — check at least one mesh with faces
            meshes = [g for g in scene.geometry.values()
                      if hasattr(g, "faces") and len(g.faces) > 10]
            return len(meshes) > 0

        elif isinstance(scene, trimesh.Trimesh):
            return len(scene.faces) > 10

        return False

    except Exception:
        return False


def copy_model_to_class_dir(glb_path: str, cls_name: str,
                             output_dir: Path, uid: str) -> Optional[Path]:
    """Copy validated model to class directory. Returns destination path."""
    cls_dir = output_dir / cls_name
    cls_dir.mkdir(parents=True, exist_ok=True)
    dst = cls_dir / f"{uid}.glb"
    if not dst.exists():
        shutil.copy2(glb_path, dst)
    return dst


def fetch_class(cls_name: str, search_terms: list, output_dir: Path,
                n_per_class: int) -> int:
    """
    Fetch n_per_class models for a single e-waste class.
    Returns count of successfully fetched models.
    """
    cls_dir = output_dir / cls_name
    cls_dir.mkdir(parents=True, exist_ok=True)

    # Check what we already have
    existing = list(cls_dir.glob("*.glb")) + list(cls_dir.glob("*.obj"))
    already = len(existing)
    need    = max(0, n_per_class - already)

    if need == 0:
        print(f"  {cls_name:<20} already has {already} models ✓")
        return already

    print(f"  {cls_name:<20} have {already}, need {need} more")

    fetched = 0
    for term in search_terms:
        if fetched >= need:
            break

        candidates = search_objaverse(term, n=100)
        if not candidates:
            continue

        # Shuffle to get variety across runs
        random.shuffle(candidates)
        uids = [uid for uid, _ in candidates]

        # Download
        n_to_dl = min(need - fetched + 5, 20)  # Overdownload slightly for rejects
        objects  = download_objaverse_objects(uids, n_to_dl)

        for uid, glb_path in objects.items():
            if fetched >= need:
                break
            if not glb_path or not Path(glb_path).exists():
                continue

            if validate_mesh(glb_path):
                dst = copy_model_to_class_dir(glb_path, cls_name, output_dir, uid)
                if dst:
                    fetched += 1
                    print(f"    ✓ {uid[:20]}... → {dst.name}")
            else:
                print(f"    ✗ {uid[:20]}... invalid mesh, skipped")

    total = already + fetched
    if fetched < need:
        print(f"  ⚠  {cls_name}: only got {total}/{n_per_class} "
              f"(Objaverse coverage limited for this class)")
    return total


def known_uids(output_dir: Path) -> set:
    """UIDs already placed in ANY class dir or in _rejected/ — never refetch these."""
    return {p.stem for p in output_dir.glob("**/*.glb")}


def fetch_class_lvis(cls_name: str, categories: list, output_dir: Path,
                     n_per_class: int, lvis: dict, seen: set) -> int:
    """
    Fill a class from Objaverse-LVIS categories, round-robin across categories
    so multi-category classes (e.g. small_appliance) stay diverse. `seen` is
    shared across classes so one model can never land in two classes.
    """
    cls_dir = output_dir / cls_name
    cls_dir.mkdir(parents=True, exist_ok=True)
    already = len(list(cls_dir.glob("*.glb")) + list(cls_dir.glob("*.obj")))
    need = max(0, n_per_class - already)
    if need == 0:
        print(f"  {cls_name:<20} already has {already} models ✓")
        return already
    print(f"  {cls_name:<20} have {already}, need {need} more (LVIS: {', '.join(categories)})")

    pools = []
    for cat in categories:
        uids = [u for u in lvis.get(cat, []) if u not in seen]
        random.shuffle(uids)
        pools.append(uids)

    fetched = 0
    while fetched < need and any(pools):
        for pool in pools:
            if fetched >= need or not pool:
                continue
            uid = pool.pop()
            seen.add(uid)
            for got_uid, glb_path in download_objaverse_objects([uid], 1).items():
                if glb_path and Path(glb_path).exists() and validate_mesh(glb_path):
                    dst = copy_model_to_class_dir(glb_path, cls_name, output_dir, got_uid)
                    fetched += 1
                    print(f"    ✓ {got_uid[:20]}... → {dst.name}")
                else:
                    print(f"    ✗ {got_uid[:20]}... invalid mesh, skipped")

    total = already + fetched
    if fetched < need:
        print(f"  ⚠  {cls_name}: only got {total}/{n_per_class} (LVIS categories exhausted)")
    return total


def generate_pile_assets(output_dir: Path, n: int = 30):
    """
    Generate ewaste_pile and cable_bundle assets procedurally using Blender's
    Python API (must be run inside Blender, not standard Python).

    This creates a Blender script that, when run with `blender --background --python`,
    generates randomized pile meshes using rigid body physics simulation.

    The generated script is saved for the user to run.
    """
    script_path = output_dir / "_generate_piles.py"

    blender_script = f'''"""
Procedural E-Waste Pile Generator
Run with: blender --background --python _generate_piles.py
Generates {n} randomized ewaste_pile meshes via rigid body physics.
"""
import bpy, math, random
from pathlib import Path

output_dir = Path("{(output_dir / 'ewaste_pile').resolve().as_posix()}")
output_dir.mkdir(parents=True, exist_ok=True)
cable_dir = Path("{(output_dir / 'cable_bundle').resolve().as_posix()}")
cable_dir.mkdir(parents=True, exist_ok=True)

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge(do_recursive=True)

def generate_pile(n_objects=15, seed=42):
    random.seed(seed)
    # Ground plane
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
    ground = bpy.context.active_object
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = "PASSIVE"

    # Drop random primitive shapes (stand-ins for e-waste objects)
    shapes = ["CUBE", "CYLINDER", "SPHERE", "CONE"]
    for i in range(n_objects):
        shape = random.choice(shapes)
        if shape == "CUBE":
            bpy.ops.mesh.primitive_cube_add(
                size=random.uniform(0.2, 0.8),
                location=(random.uniform(-1, 1), random.uniform(-1, 1), 
                          random.uniform(1, 4))
            )
        elif shape == "CYLINDER":
            bpy.ops.mesh.primitive_cylinder_add(
                radius=random.uniform(0.1, 0.4),
                depth=random.uniform(0.1, 0.6),
                location=(random.uniform(-1, 1), random.uniform(-1, 1),
                          random.uniform(1, 4))
            )
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=random.uniform(0.1, 0.4),
                location=(random.uniform(-1, 1), random.uniform(-1, 1),
                          random.uniform(1, 4))
            )
        obj = bpy.context.active_object
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = "ACTIVE"
        obj.rotation_euler = (
            random.uniform(0, math.pi),
            random.uniform(0, math.pi),
            random.uniform(0, math.pi)
        )

    # Run physics simulation
    bpy.context.scene.frame_end = 80
    bpy.ops.ptcache.bake_all(bake=True)
    bpy.context.scene.frame_set(80)
    bpy.ops.object.select_all(action="DESELECT")

    # Select all active rigid bodies (the pile objects)
    for obj in bpy.data.objects:
        if obj.rigid_body and obj.rigid_body.type == "ACTIVE":
            obj.select_set(True)

    # Join into single mesh
    bpy.context.view_layer.objects.active = bpy.data.objects[-1]
    bpy.ops.object.join()
    pile = bpy.context.active_object
    pile.name = f"ewaste_pile_{{seed:04d}}"
    return pile

for i in range({n}):
    clear_scene()
    pile = generate_pile(n_objects=random.randint(8, 25), seed=i*7+13)
    out_path = output_dir / f"pile_{{i:04d}}.obj"
    bpy.ops.object.select_all(action="DESELECT")
    pile.select_set(True)
    if hasattr(bpy.ops.wm, "obj_export"):   # Blender 4.x (legacy exporter removed)
        bpy.ops.wm.obj_export(
            filepath=str(out_path),
            export_selected_objects=True,
            export_materials=False
        )
    else:                                     # Blender <= 3.x
        bpy.ops.export_scene.obj(
            filepath=str(out_path),
            use_selection=True,
            use_materials=False
        )
    print(f"Pile {{i+1}}/{n}: {{out_path.name}}")

print("Done. {{n}} pile assets generated.")
'''

    with open(script_path, "w") as f:
        f.write(blender_script)

    print(f"\n[INFO] Procedural pile generator written to: {script_path}")
    print(f"  Run with:")
    print(f"  blender --background --python {script_path}")
    print(f"  This generates {n} ewaste_pile meshes via physics simulation.\n")


def audit_models(output_dir: Path):
    """Print a per-class inventory of available 3D models."""
    print(f"\n{'─'*50}")
    print(f"  Model Inventory: {output_dir}")
    print(f"{'─'*50}")
    total = 0
    for cls_name in sorted(CLASS_SEARCH_TERMS.keys()):
        cls_dir = output_dir / cls_name
        if cls_dir.exists():
            files = (list(cls_dir.glob("*.glb")) +
                     list(cls_dir.glob("*.obj")) +
                     list(cls_dir.glob("*.fbx")))
            n = len(files)
            status = "✓" if n >= 5 else ("⚠" if n > 0 else "✗")
            print(f"  {status} {cls_name:<20} {n:3d} models")
            total += n
        else:
            print(f"  ✗ {cls_name:<20}   0 models  (directory missing)")
    print(f"{'─'*50}")
    print(f"  Total: {total} models across {len(CLASS_SEARCH_TERMS)} classes")
    print(f"{'─'*50}\n")

    # Warn on classes with fewer than 5 models
    sparse = [cls for cls in CLASS_SEARCH_TERMS
              if len(list((output_dir / cls).glob("*") if (output_dir / cls).exists()
                          else [])) < 5]
    if sparse:
        print(f"  ⚠  Sparse classes (< 5 models): {', '.join(sparse)}")
        print(f"     Consider manual Sketchfab/BlenderKit downloads for these.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Objaverse 3D model fetcher for EWADS e-waste classes"
    )
    parser.add_argument("--output_dir",   type=str, default="./assets/models")
    parser.add_argument("--n_per_class",  type=int, default=15,
                        help="Target number of models per class")
    parser.add_argument("--classes",      nargs="+",
                        choices=list(CLASS_SEARCH_TERMS.keys()),
                        help="Specific classes to fetch (default: all)")
    parser.add_argument("--audit",        action="store_true",
                        help="Show model inventory and exit")
    parser.add_argument("--gen_piles",    action="store_true",
                        help="Write Blender pile generator script")
    parser.add_argument("--n_piles",      type=int, default=30)
    parser.add_argument("--lvis",         action="store_true",
                        help="Fetch classes in CLASS_LVIS_CATEGORIES from Objaverse-LVIS "
                             "(human-labelled) instead of free-text search")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.audit:
        audit_models(output_dir)
        return

    if args.gen_piles:
        generate_pile_assets(output_dir, n=args.n_piles)
        return

    check_dependencies()

    if args.lvis:
        import objaverse
        lvis = objaverse.load_lvis_annotations()
        seen = known_uids(output_dir)
        lvis_classes = [c for c in (args.classes or CLASS_LVIS_CATEGORIES)
                        if c in CLASS_LVIS_CATEGORIES]
        print(f"\n{'='*60}\n  Objaverse-LVIS Fetcher — EWADS\n"
              f"  Target: {args.n_per_class} per class | Classes: {len(lvis_classes)}\n"
              f"  Excluding {len(seen)} UIDs already placed or rejected\n{'='*60}\n")
        results = {c: fetch_class_lvis(c, CLASS_LVIS_CATEGORIES[c], output_dir,
                                       args.n_per_class, lvis, seen)
                   for c in lvis_classes}
        audit_models(output_dir)
        report_path = output_dir / "acquisition_report_lvis.json"
        report_path.write_text(json.dumps({"n_per_class_target": args.n_per_class,
                                           "results": results}, indent=2))
        print(f"Report written: {report_path}")
        return

    target_classes = args.classes or list(CLASS_SEARCH_TERMS.keys())
    # ewaste_pile handled procedurally — skip in Objaverse fetch
    procedural = {"ewaste_pile", "cable_bundle"}
    fetch_classes = [c for c in target_classes if c not in procedural]

    print(f"\n{'='*60}")
    print(f"  Objaverse Model Fetcher — EWADS")
    print(f"  Target: {args.n_per_class} models per class")
    print(f"  Classes: {len(fetch_classes)}")
    print(f"{'='*60}\n")

    results = {}
    for cls_name in fetch_classes:
        terms = CLASS_SEARCH_TERMS.get(cls_name, [])
        count = fetch_class(cls_name, terms, output_dir, args.n_per_class)
        results[cls_name] = count

    # Procedural pile/cable notice
    if "ewaste_pile" in target_classes or "cable_bundle" in target_classes:
        generate_pile_assets(output_dir, n=args.n_piles)

    # Final audit
    audit_models(output_dir)

    # Write acquisition report
    report = {
        "n_per_class_target": args.n_per_class,
        "results": results,
        "procedural_classes": list(procedural),
    }
    report_path = output_dir / "acquisition_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
