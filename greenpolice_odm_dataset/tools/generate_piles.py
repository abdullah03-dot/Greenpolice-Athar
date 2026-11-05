import blenderproc as bproc
# ^ BlenderProc requires this to be the very first line of the file
"""
Build procedural ewaste_pile and cable_bundle assets (replaces the old
primitive-based _generate_piles.py, whose physics bake scattered the shapes).

  ewaste_pile  : 8-20 reviewed e-waste models from assets/models/<class>/,
                 scaled to real-world size and dropped with rigid-body physics
                 into a heap, then joined into one mesh.
  cable_bundle : 8-30 tangled / coiled procedural cables (bevelled curves with
                 insulation colours), joined into one mesh.

Run AFTER the model review so only curated models end up in piles.

Usage:
  blenderproc run tools/generate_piles.py --kind ewaste_pile  --n 30
  blenderproc run tools/generate_piles.py --kind cable_bundle --n 30
"""
import argparse
import math
import random
from pathlib import Path

import bpy
import mathutils

# Keep in sync with CLASS_REAL_SIZE_M in 02_blenderproc_synthetic_v2.py
CLASS_REAL_SIZE_M = {
    "crt_monitor": 0.45, "flat_screen": 0.60, "circuit_board": 0.25,
    "battery": 0.25, "single_cable": 1.00, "mobile_phone": 0.15,
    "keyboard": 0.45, "printer": 0.50, "small_appliance": 0.35,
    "cpu_chip": 0.04, "hard_drive": 0.15, "power_supply": 0.16,
    "plastic_casing": 0.40, "metal_component": 0.30, "lighting": 0.30,
}
# Too big to sit inside a pile, or not a pile ingredient
PILE_EXCLUDE = {"large_appliance", "ewaste_pile", "cable_bundle"}

INSULATION_COLOURS = [
    (0.02, 0.02, 0.02), (0.55, 0.55, 0.55), (0.85, 0.85, 0.82), (0.6, 0.05, 0.03),
    (0.03, 0.12, 0.5), (0.75, 0.6, 0.05), (0.05, 0.35, 0.08), (0.45, 0.25, 0.1),
]


_LOAD_COUNTER = [0]


def load_model_unique(path: str):
    """
    bproc.loader.load_obj() with every newly created image given a unique name.
    Embedded glTF textures get generic names (Image_0, texture, ...); when a
    later model reuses a name, Cycles served the earlier model's cached pixels
    (e.g. a battery's label texture appeared on TVs and monitors).
    """
    before = set(bpy.data.images)
    loaded = bproc.loader.load_obj(path)
    _LOAD_COUNTER[0] += 1
    for img in set(bpy.data.images) - before:
        img.name = f"m{_LOAD_COUNTER[0]:06d}_{img.name}"
    return loaded


def merge_loaded(loaded):
    """Meshes only, parents baked, empties dropped, joined, origin at bounds centre."""
    meshes = [o.blender_obj for o in loaded if o.blender_obj.type == "MESH"]
    empties = [o.blender_obj for o in loaded if o.blender_obj.type != "MESH"]
    for m in meshes:
        world = m.matrix_world.copy()
        m.parent = None
        m.matrix_world = world
    for e in empties:
        bpy.data.objects.remove(e, do_unlink=True)
    if not meshes:
        return None
    select_only(meshes)
    if len(meshes) > 1:
        bpy.ops.object.join()
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    return bproc.types.MeshObject(meshes[0])


def select_only(objs):
    vl = bpy.context.view_layer
    # Removing objects leaves stale None entries until the view layer refreshes
    vl.update()
    for o in vl.objects:
        if o is not None:
            o.select_set(False)
    for o in objs:
        o.select_set(True)
    vl.objects.active = objs[0]


def export_joined(objs, out_path: Path):
    select_only(objs)
    if len(objs) > 1:
        bpy.ops.object.join()
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    select_only([objs[0]])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(out_path), export_format="GLB",
                              use_selection=True)


def build_pile(pool, rng):
    ground = bproc.object.create_primitive("PLANE", scale=[5, 5, 1])
    ground.enable_rigidbody(active=False, collision_shape="BOX", friction=1.0)

    items = []
    for j, (cls, path) in enumerate(rng.sample(pool, k=min(len(pool), rng.randint(8, 20)))):
        try:
            obj = merge_loaded(load_model_unique(str(path)))
        except Exception as e:
            print(f"    skip {path.name}: {e}")
            continue
        if obj is None:
            continue
        bb = obj.get_bound_box()
        extent = float((bb.max(axis=0) - bb.min(axis=0)).max())
        if extent < 1e-6:
            obj.delete()
            continue
        s = CLASS_REAL_SIZE_M.get(cls, 0.3) / extent * rng.uniform(0.7, 1.2)
        obj.set_scale([s, s, s])
        obj.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
        # Narrow column drop so parts land on each other and form a heap
        r, a = rng.uniform(0, 0.2), rng.uniform(0, 2 * math.pi)
        obj.set_location([r * math.cos(a), r * math.sin(a), 0.3 + j * 0.3])
        obj.set_rotation_euler([rng.uniform(0, 2 * math.pi) for _ in range(3)])
        obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL", friction=1.0,
                             linear_damping=0.3, angular_damping=0.5)
        items.append(obj)

    if len(items) < 4:
        return None
    # The simulation can replace the underlying bpy objects, invalidating our
    # wrappers — so look parts up again by name afterwards
    for k, o in enumerate(items):
        o.set_name(f"pile_part_{k:02d}")
    names = [o.get_name() for o in items]
    ground.set_name("pile_ground")
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=3, max_simulation_time=12, check_object_interval=1)

    # Drop anything that bounced away from the heap
    kept = []
    for n in names:
        o = bpy.data.objects.get(n)
        if o is None:
            continue
        # Keep parts that settled on the heap: near the centre, not fallen
        # through the ground, not launched upward
        if math.hypot(o.location.x, o.location.y) < 1.0 and -0.05 < o.location.z < 1.5:
            kept.append(o)
        else:
            bpy.data.objects.remove(o, do_unlink=True)
    g = bpy.data.objects.get("pile_ground")
    if g is not None:
        bpy.data.objects.remove(g, do_unlink=True)
    return kept if len(kept) >= 4 else None


def build_cable_bundle(rng):
    cables = []
    coiled = rng.random() < 0.4
    for c in range(rng.randint(8, 30)):
        curve = bpy.data.curves.new(f"cable_{c}", type="CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = rng.uniform(0.002, 0.007)
        curve.bevel_resolution = 2
        curve.resolution_u = 6
        spline = curve.splines.new("NURBS")
        n_pts = rng.randint(8, 16)
        spline.points.add(n_pts - 1)
        radius, turns = rng.uniform(0.08, 0.2), rng.uniform(2, 6)
        for k, p in enumerate(spline.points):
            if coiled:   # coil of cable around a common axis, with jitter
                t = k / (n_pts - 1) * turns * 2 * math.pi
                x, y = radius * math.cos(t), radius * math.sin(t)
                z = rng.uniform(0, 0.08)
            else:        # tangled heap
                x, y, z = rng.gauss(0, 0.18), rng.gauss(0, 0.18), abs(rng.gauss(0, 0.05))
            p.co = (x + rng.gauss(0, 0.02), y + rng.gauss(0, 0.02), z, 1.0)
        spline.use_endpoint_u = True
        obj = bpy.data.objects.new(f"cable_{c}", curve)
        bpy.context.scene.collection.objects.link(obj)

        mat = bpy.data.materials.new(f"insulation_{c}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*rng.choice(INSULATION_COLOURS), 1)
        bsdf.inputs["Roughness"].default_value = rng.uniform(0.4, 0.8)
        curve.materials.append(mat)
        cables.append(obj)

    select_only(cables)
    bpy.ops.object.convert(target="MESH")
    return [o for o in bpy.context.selected_objects]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["ewaste_pile", "cable_bundle"], required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--models_dir", default="./assets/models")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out_dir", default=None, help="default: <models_dir>/<kind>")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir) if args.out_dir else models_dir / args.kind
    rng = random.Random(args.seed)
    bproc.init()
    # Don't keep render data (incl. the image cache) between renders
    bpy.context.scene.render.use_persistent_data = False

    pool = [(d.name, p) for d in sorted(models_dir.iterdir())
            if d.is_dir() and not d.name.startswith("_") and d.name not in PILE_EXCLUDE
            for p in sorted(d.iterdir()) if p.suffix.lower() in {".glb", ".obj"}]
    if args.kind == "ewaste_pile":
        print(f"pile ingredient pool: {len(pool)} models")

    made = 0
    attempt = 0
    while made < args.n and attempt < args.n * 3:
        attempt += 1
        out = out_dir / f"{args.kind}_{made:04d}.glb"
        if out.exists():
            made += 1
            continue
        bproc.clean_up()
        objs = build_pile(pool, rng) if args.kind == "ewaste_pile" else build_cable_bundle(rng)
        if not objs:
            print(f"  attempt {attempt}: too few objects, retrying")
            continue
        export_joined(objs, out)
        made += 1
        print(f"[{made}/{args.n}] {out.name} ({len(objs)} parts)")


main()
