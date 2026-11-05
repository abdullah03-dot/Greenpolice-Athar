import blenderproc as bproc
# ^ BlenderProc requires this to be the very first line of the file
"""
Procedural 3D models for the six e-waste classes that Objaverse cannot supply:

    hard_drive  power_supply  single_cable  cpu_chip  metal_component  plastic_casing

Why procedural: none of these has an Objaverse-LVIS category, and keyword
search was ~50% wrong-class (see HANDOFF.md). All six are geometrically simple
manufactured objects with standard real-world dimensions, so a parametric
generator gives unlimited, correctly-sized variety.

Every model is built at real-world scale in metres, with generated label / PCB
/ vent textures embedded in the .glb. Next to each model a sidecar
`<name>.json` records its true size so the renderer places it at that size
instead of the class-wide default in CLASS_REAL_SIZE_M:

    {"real_size_m": 0.147, "scale_jitter": [0.97, 1.03],
     "min_feature_m": null, "variant": "hdd_3.5", "generator": "..."}

Output goes straight into assets/models/<class>/proc_<class>_NNNN.glb so the
normal review flow picks it up (HANDOFF.md section 7): render thumbnails with
--skip_reviewed, reject bad tiles in a round-3 decisions file, apply it.

Usage:
  blenderproc run tools/generate_procedural_models.py --n 40
  blenderproc run tools/generate_procedural_models.py --classes hard_drive cpu_chip --n 60
  blenderproc run tools/generate_procedural_models.py --n 40 --seed 7   # different batch

Existing files are never overwritten: numbering continues after the highest
proc_<class>_NNNN already present.
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

import bmesh
import bpy
import mathutils
import numpy as np

CLASSES = ["hard_drive", "power_supply", "single_cable",
           "cpu_chip", "metal_component", "plastic_casing"]
GENERATOR_VERSION = "ewads-procgen-1.0"

rng = random.Random(0)


# ─── Generated textures ───────────────────────────────────────────────────────
def _to_image(arr: np.ndarray, name: str):
    """numpy HxWx3 float [0,1] (row 0 = top) -> packed bpy image."""
    h, w, _ = arr.shape
    img = bpy.data.images.new(name, width=w, height=h, alpha=True)
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(arr, 0, 1)
    img.pixels.foreach_set(rgba[::-1].ravel())   # Blender stores bottom row first
    img.pack()
    return img


def tex_label(w=384, h=256, bg=None, fg=None):
    """Product sticker: text-like bars, a barcode, a logo block."""
    bg = np.array(bg if bg is not None else rng.choice(
        [(0.92, 0.92, 0.9), (0.85, 0.87, 0.9), (0.12, 0.12, 0.13), (0.1, 0.25, 0.55),
         (0.75, 0.78, 0.8), (0.9, 0.85, 0.6)]))
    fg = np.array(fg if fg is not None else ((0.08, 0.08, 0.08) if bg.mean() > 0.5 else (0.9, 0.9, 0.9)))
    a = np.ones((h, w, 3)) * bg
    y = int(h * 0.12)
    while y < h * 0.62:                                        # text lines
        x = int(w * rng.uniform(0.05, 0.1))
        lh = rng.randint(3, 8)
        while x < w * rng.uniform(0.55, 0.95):
            ww = rng.randint(4, 30)
            a[y:y + lh, x:x + ww] = fg
            x += ww + rng.randint(3, 7)
        y += lh + rng.randint(4, 10)
    bx, by = int(w * 0.08), int(h * 0.7)                       # barcode
    for i in range(rng.randint(30, 60)):
        bw = rng.randint(1, 3)
        if rng.random() < 0.6:
            a[by:by + int(h * 0.2), bx:bx + bw] = fg
        bx += bw + rng.randint(1, 2)
        if bx > w * 0.6:
            break
    lx, ly, ls = int(w * rng.uniform(0.65, 0.78)), int(h * 0.62), int(h * 0.28)   # logo
    a[ly:ly + ls, lx:lx + ls] = np.array(rng.choice([(0.7, 0.1, 0.1), (0.1, 0.3, 0.7), fg]))
    return _to_image(a, "tex_label")


def tex_pcb(w=384, h=384, base=None):
    """Circuit board: traces, pads, ICs, silkscreen."""
    base = np.array(base if base is not None else rng.choice(
        [(0.05, 0.28, 0.1), (0.03, 0.2, 0.08), (0.05, 0.12, 0.3), (0.06, 0.06, 0.06), (0.35, 0.2, 0.08)]))
    a = np.ones((h, w, 3)) * base
    trace = np.clip(base * 1.5 + 0.05, 0, 1)
    for _ in range(rng.randint(40, 90)):                       # traces (manhattan)
        x, y = rng.randrange(w), rng.randrange(h)
        for _ in range(rng.randint(2, 5)):
            L = rng.randint(10, 90)
            if rng.random() < 0.5:
                a[y:y + 2, x:min(w, x + L)] = trace; x = min(w - 1, x + L)
            else:
                a[y:min(h, y + L), x:x + 2] = trace; y = min(h - 1, y + L)
    gold = np.array((0.8, 0.62, 0.25))
    for _ in range(rng.randint(30, 80)):                       # pads
        x, y, s = rng.randrange(w - 6), rng.randrange(h - 6), rng.randint(2, 5)
        a[y:y + s, x:x + s] = gold
    for _ in range(rng.randint(3, 9)):                         # IC packages
        cw, ch = rng.randint(15, 70), rng.randint(15, 60)
        x, y = rng.randrange(max(1, w - cw)), rng.randrange(max(1, h - ch))
        a[y:y + ch, x:x + cw] = (0.04, 0.04, 0.045)
    for _ in range(rng.randint(10, 25)):                       # silkscreen
        x, y = rng.randrange(w - 20), rng.randrange(h - 5)
        a[y:y + 2, x:x + rng.randint(5, 18)] = (0.85, 0.85, 0.8)
    return _to_image(a, "tex_pcb")


def tex_vents(w=256, h=256, base=(0.1, 0.1, 0.1), hole=(0.01, 0.01, 0.01), kind=None):
    """Grille: round holes, slots or honeycomb-ish grid."""
    kind = kind or rng.choice(["holes", "slots", "grid"])
    a = np.ones((h, w, 3)) * np.array(base)
    step = rng.randint(10, 22)
    for yy in range(step // 2, h - 3, step):
        off = step // 2 if (kind == "holes" and (yy // step) % 2) else 0
        for xx in range(step // 2 + off, w - 3, step):
            if kind == "slots":
                a[yy:yy + max(2, step // 3), xx - step // 2 + 2:xx + step // 2 - 2] = hole
            else:
                r = max(2, step // 3)
                yv, xv = np.ogrid[-r:r + 1, -r:r + 1]
                m = (xv ** 2 + yv ** 2) <= r * r if kind == "holes" else np.ones_like(xv * yv, bool)
                ys, xs = slice(max(0, yy - r), yy + r + 1), slice(max(0, xx - r), xx + r + 1)
                sub = a[ys, xs]
                sub[m[:sub.shape[0], :sub.shape[1]]] = hole
    return _to_image(a, "tex_vents")


def tex_engraved(w=256, h=256, base=(0.72, 0.72, 0.72)):
    """CPU heat-spreader: brushed metal with engraved text."""
    a = np.ones((h, w, 3)) * np.array(base)
    a += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.015, (h, 1, 1))  # brushing
    dark = np.array(base) * 0.55
    y = int(h * 0.3)
    for _ in range(rng.randint(3, 6)):
        x = int(w * 0.15)
        while x < w * 0.8:
            ww = rng.randint(4, 16)
            a[y:y + 5, x:x + ww] = dark
            x += ww + 4
        y += 14
    return _to_image(a, "tex_engraved")


# ─── Materials ────────────────────────────────────────────────────────────────
def mat(name, color, metallic=0.0, roughness=0.5, image=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if image is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def jitter(c, amt=0.06):
    return tuple(min(1, max(0, v + rng.uniform(-amt, amt))) for v in c)


PLASTICS = [(0.78, 0.74, 0.62), (0.85, 0.83, 0.76), (0.05, 0.05, 0.05), (0.15, 0.15, 0.16),
            (0.45, 0.45, 0.47), (0.9, 0.9, 0.9), (0.62, 0.6, 0.55), (0.12, 0.2, 0.4)]
ALU = (0.72, 0.72, 0.74)
STEEL = (0.55, 0.56, 0.58)
BLACK_PAINT = (0.04, 0.04, 0.045)
COPPER = (0.8, 0.45, 0.25)


# ─── Geometry primitives (all sizes in metres) ────────────────────────────────
def _bm_with_uv():
    """bmesh.ops.create_*(calc_uvs=True) only fills a UV layer that already
    exists — without this every textured decal samples a single pixel."""
    bm = bmesh.new()
    bm.loops.layers.uv.new("UVMap")
    return bm


def _link(me, name, material):
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if material is not None:
        ob.data.materials.append(material)
    return ob


def box(dims, loc=(0, 0, 0), material=None, bevel=0.0, rot=(0, 0, 0)):
    """Axis-aligned box of size dims centred at loc."""
    me = bpy.data.meshes.new("box")
    bm = _bm_with_uv()
    bmesh.ops.create_cube(bm, size=1.0, calc_uvs=True)
    bmesh.ops.scale(bm, vec=mathutils.Vector(dims), verts=bm.verts)
    if bevel > 0:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=min(bevel, min(dims) * 0.45),
                        segments=2, affect="EDGES", profile=0.5)
    bm.to_mesh(me)
    bm.free()
    ob = _link(me, "box", material)
    ob.location = loc
    ob.rotation_euler = rot
    return ob


def decal(size_xy, loc, material, normal="+Z"):
    """
    Thin textured plane (full 0..1 UV) used for labels, PCBs, grilles.
    size_xy is in WORLD axes of the face it sits on:
      ±Z face -> (x, y)   ±X face -> (y, z)   ±Y face -> (x, z)
    """
    # The ±X rotation below maps local x -> world z, so swap for those faces
    sx, sy = (size_xy[1], size_xy[0]) if normal in ("+X", "-X") else size_xy
    me = bpy.data.meshes.new("decal")
    bm = _bm_with_uv()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5, calc_uvs=True)
    bmesh.ops.scale(bm, vec=(sx, sy, 1), verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    ob = _link(me, "decal", material)
    ob.location = loc
    ob.rotation_euler = {"+Z": (0, 0, 0), "-Z": (math.pi, 0, 0),
                         "+X": (0, math.pi / 2, 0), "-X": (0, -math.pi / 2, 0),
                         "+Y": (-math.pi / 2, 0, 0), "-Y": (math.pi / 2, 0, 0)}[normal]
    return ob


def cylinder(r, depth, loc=(0, 0, 0), material=None, rot=(0, 0, 0), verts=24):
    me = bpy.data.meshes.new("cyl")
    bm = _bm_with_uv()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=verts,
                          radius1=r, radius2=r, depth=depth, calc_uvs=True)
    bm.to_mesh(me)
    bm.free()
    ob = _link(me, "cyl", material)
    ob.location = loc
    ob.rotation_euler = rot
    return ob


def torus(R, r, loc=(0, 0, 0), material=None):
    me = bpy.data.meshes.new("torus")
    bm = bmesh.new()
    seg_major, seg_minor = 32, 8
    verts = []
    for i in range(seg_major):
        a = 2 * math.pi * i / seg_major
        ring = []
        for j in range(seg_minor):
            b = 2 * math.pi * j / seg_minor
            ring.append(bm.verts.new(((R + r * math.cos(b)) * math.cos(a),
                                      (R + r * math.cos(b)) * math.sin(a), r * math.sin(b))))
        verts.append(ring)
    for i in range(seg_major):
        for j in range(seg_minor):
            bm.faces.new((verts[i][j], verts[(i + 1) % seg_major][j],
                          verts[(i + 1) % seg_major][(j + 1) % seg_minor], verts[i][(j + 1) % seg_minor]))
    bm.to_mesh(me)
    bm.free()
    ob = _link(me, "torus", material)
    ob.location = loc
    return ob


def tube_along(points, radius, material):
    """Bevelled curve through 3D points, converted to a mesh."""
    cu = bpy.data.curves.new("wire", type="CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    cu.resolution_u = 8
    cu.use_fill_caps = True
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(points) - 1)
    for bp, p in zip(sp.bezier_points, points):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    ob = bpy.data.objects.new("wire", cu)
    bpy.context.scene.collection.objects.link(ob)
    cu.materials.append(material)
    return to_mesh(ob)


def to_mesh(ob):
    """Bake an object's modifiers / curve geometry into a plain mesh object."""
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True,
                                         depsgraph=dg)
    new = bpy.data.objects.new(ob.name + "_m", me)
    new.matrix_world = ob.matrix_world.copy()
    bpy.context.scene.collection.objects.link(new)
    bpy.data.objects.remove(ob, do_unlink=True)
    return new


def solidify_shell(ob, thickness):
    mod = ob.modifiers.new("solid", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = -1
    return to_mesh(ob)


def open_tray(dims, material, thickness, remove=("+Z",)):
    """Box with faces removed then given wall thickness — a stripped casing shell."""
    me = bpy.data.meshes.new("tray")
    bm = _bm_with_uv()
    bmesh.ops.create_cube(bm, size=1.0, calc_uvs=True)
    bmesh.ops.scale(bm, vec=mathutils.Vector(dims), verts=bm.verts)
    axes = {"+Z": (2, 1), "-Z": (2, -1), "+X": (0, 1), "-X": (0, -1), "+Y": (1, 1), "-Y": (1, -1)}
    bm.faces.ensure_lookup_table()
    kill = [f for f in bm.faces
            if any(abs(f.normal[axes[k][0]] - axes[k][1]) < 1e-3 for k in remove)]
    bmesh.ops.delete(bm, geom=kill, context="FACES_ONLY")
    bm.to_mesh(me)
    bm.free()
    return solidify_shell(_link(me, "tray", material), thickness)


def break_mesh(ob, n_cuts=1):
    """Cut the mesh with random planes and discard one side (broken fragment)."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    lo = mathutils.Vector([min(v.co[i] for v in bm.verts) for i in range(3)])
    hi = mathutils.Vector([max(v.co[i] for v in bm.verts) for i in range(3)])
    for _ in range(n_cuts):
        co = lo.lerp(hi, rng.uniform(0.35, 0.75))
        no = mathutils.Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.3, 0.3))).normalized()
        geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=co, plane_no=no, clear_outer=True)
    bm.to_mesh(me)
    bm.free()
    return ob


def join(objs, name):
    objs = [o for o in objs if o is not None]
    vl = bpy.context.view_layer
    vl.update()
    for o in vl.objects:
        if o is not None:
            o.select_set(False)
    for o in objs:
        o.select_set(True)
    vl.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    ob = vl.objects.active
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    ob.location = (0, 0, 0)
    ob.name = name
    return ob


# ─── Class generators — each returns (objects, variant, min_feature_m) ───────
def gen_hard_drive():
    kind = rng.choices(["hdd_3.5", "hdd_2.5", "ssd_2.5", "hdd_3.5_open"], [0.45, 0.25, 0.15, 0.15])[0]
    if kind.startswith("hdd_3.5"):
        L, W, H = 0.147, 0.1016, 0.0261
    else:
        L, W, H = 0.100, 0.0699, rng.choice([0.007, 0.0095])
    top_c = rng.choice([ALU, jitter(ALU, 0.05), (0.3, 0.3, 0.32), BLACK_PAINT])
    body_m = mat("hdd_body", jitter(top_c, 0.03), metallic=0.85 if top_c != BLACK_PAINT else 0.2,
                 roughness=rng.uniform(0.3, 0.55))
    parts = []
    if kind == "hdd_3.5_open":                                   # cover removed: tray + platter
        parts.append(open_tray((L, W, H), body_m, 0.003))
        platter = mat("platter", (0.85, 0.85, 0.88), metallic=1.0, roughness=0.05)
        parts.append(cylinder(0.0475, 0.0012, (-L * 0.12, 0, H * 0.25), platter, verts=48))
        parts.append(cylinder(0.012, 0.004, (-L * 0.12, 0, H * 0.3), body_m))
        arm = mat("arm", (0.6, 0.6, 0.62), metallic=0.9, roughness=0.3)
        parts.append(box((0.055, 0.008, 0.002), (L * 0.22, W * 0.12, H * 0.32), arm,
                         rot=(0, 0, rng.uniform(2.6, 3.0))))
        parts.append(box((0.02, 0.03, 0.006), (L * 0.36, W * 0.25, H * 0.2),
                         mat("magnet", (0.25, 0.25, 0.28), 0.9, 0.35)))
    else:
        parts.append(box((L, W, H), material=body_m, bevel=0.0015))
        lw, lh = L * rng.uniform(0.55, 0.85), W * rng.uniform(0.6, 0.85)
        parts.append(decal((lw, lh), (rng.uniform(-0.1, 0.1) * L, 0, H / 2 + 0.0002),
                           mat("label", (1, 1, 1), 0, 0.6, tex_label())))
        for sx in (-1, 1):                                      # screw heads
            for sy in (-1, 1):
                parts.append(cylinder(0.0025, 0.0006, (sx * (L / 2 - 0.006), sy * (W / 2 - 0.006),
                                                       H / 2 + 0.0002), mat("screw", STEEL, 0.9, 0.4)))
    if kind != "ssd_2.5":                                       # controller PCB underneath
        parts.append(decal((L * 0.75, W * 0.8), (0, 0, -H / 2 - 0.0002),
                           mat("pcb", (1, 1, 1), 0, 0.5, tex_pcb()), normal="-Z"))
    conn = mat("sata", (0.05, 0.05, 0.05), 0, 0.5)
    parts.append(box((0.004, W * 0.6, min(H * 0.35, 0.006)), (L / 2 + 0.001, 0, -H * 0.2), conn))
    return parts, kind, None


def gen_power_supply():
    kind = rng.choices(["atx_psu", "laptop_brick", "wall_adapter", "linear_transformer"],
                       [0.45, 0.3, 0.15, 0.1])[0]
    parts = []
    if kind == "atx_psu":
        L, W, H = 0.150, rng.choice([0.140, 0.160, 0.180]), 0.086
        shell = mat("psu", rng.choice([BLACK_PAINT, STEEL, (0.25, 0.25, 0.27)]),
                    metallic=rng.uniform(0.3, 0.8), roughness=rng.uniform(0.35, 0.6))
        parts.append(box((L, W, H), material=shell, bevel=0.001))
        fan_r = min(L, W) * rng.uniform(0.35, 0.42)
        grille = mat("fan_grille", (1, 1, 1), 0.6, 0.4,
                     tex_vents(base=(0.08, 0.08, 0.08), hole=(0.01, 0.01, 0.01), kind="holes"))
        parts.append(decal((fan_r * 2, fan_r * 2), (0, 0, H / 2 + 0.0003), grille))
        parts.append(torus(fan_r, 0.002, (0, 0, H / 2 + 0.001), mat("ring", STEEL, 0.9, 0.35)))
        parts.append(decal((W * 0.9, H * 0.8), (-L / 2 - 0.0003, 0, 0),
                           mat("rear_vent", (1, 1, 1), 0.5, 0.5,
                               tex_vents(base=(0.12, 0.12, 0.12), kind="grid")), normal="-X"))
        parts.append(decal((L * 0.7, H * 0.7), (0, W / 2 + 0.0003, 0),
                           mat("psu_label", (1, 1, 1), 0, 0.6, tex_label()), normal="+Y"))
        if rng.random() < 0.35:                                 # hard-wired cable loom stub
            pts = [(L / 2, 0, 0)] + [(L / 2 + 0.05 * k, rng.uniform(-0.05, 0.05), -H / 2 + 0.01)
                                     for k in range(1, 5)]
            parts.append(tube_along(pts, 0.006, mat("loom", (0.03, 0.03, 0.03), 0, 0.6)))
    elif kind == "laptop_brick":
        L, W, H = rng.uniform(0.10, 0.16), rng.uniform(0.045, 0.07), rng.uniform(0.024, 0.036)
        plastic = mat("brick", jitter(rng.choice([(0.04, 0.04, 0.04), (0.15, 0.15, 0.15),
                                                  (0.9, 0.9, 0.9)]), 0.02), 0, rng.uniform(0.3, 0.6))
        parts.append(box((L, W, H), material=plastic, bevel=min(W, H) * 0.3))
        parts.append(decal((L * 0.6, W * 0.7), (0, 0, H / 2 + 0.0002),
                           mat("brick_label", (1, 1, 1), 0, 0.5,
                               tex_label(bg=(0.1, 0.1, 0.1) if rng.random() < 0.7 else None))))
        cable = mat("dc_cable", (0.03, 0.03, 0.03), 0, 0.55)
        n = rng.randint(4, 8)
        x, y = L / 2, 0.0
        pts = [(x, y, 0)]
        for _ in range(n):
            x += rng.uniform(0.04, 0.12); y += rng.uniform(-0.08, 0.08)
            pts.append((x, y, -H / 2 + 0.003))
        parts.append(tube_along(pts, 0.002, cable))
        parts.append(cylinder(0.0045, 0.03, (pts[-1][0] + 0.015, pts[-1][1], pts[-1][2]), cable,
                              rot=(0, math.pi / 2, 0)))
    elif kind == "wall_adapter":
        L, W, H = rng.uniform(0.045, 0.08), rng.uniform(0.035, 0.06), rng.uniform(0.03, 0.05)
        plastic = mat("adapter", jitter(rng.choice(PLASTICS), 0.03), 0, rng.uniform(0.3, 0.6))
        parts.append(box((L, W, H), material=plastic, bevel=0.003))
        prong = mat("prong", (0.75, 0.7, 0.55), 0.9, 0.3)
        for s in (-1, 1):
            parts.append(box((0.018, 0.0025, 0.007), (-L / 2 - 0.009, s * 0.006, 0), prong))
        parts.append(decal((L * 0.7, W * 0.7), (0, 0, H / 2 + 0.0002),
                           mat("adapter_label", (1, 1, 1), 0, 0.5, tex_label())))
    else:                                                      # old iron-core transformer
        L, W, H = rng.uniform(0.06, 0.11), rng.uniform(0.05, 0.09), rng.uniform(0.05, 0.09)
        core = mat("core", (0.2, 0.2, 0.22), 0.8, 0.5)
        winding = mat("winding", rng.choice([COPPER, (0.05, 0.05, 0.05), (0.8, 0.15, 0.1)]), 0.3, 0.5)
        parts.append(box((L, W * 0.35, H), (0, -W * 0.33, 0), core))
        parts.append(box((L, W * 0.35, H), (0, W * 0.33, 0), core))
        parts.append(box((L * 0.7, W * 0.4, H * 0.8), (0, 0, 0), winding, bevel=0.004))
        parts.append(box((L * 1.3, W * 1.1, 0.002), (0, 0, -H / 2), mat("bracket", STEEL, 0.9, 0.4)))
    if kind == "laptop_brick":
        return parts, kind, None, L          # gate on the brick body, not the cable
    return parts, kind, None


def gen_single_cable():
    kind = rng.choices(["power_cord", "usb", "ethernet", "av", "bare_wire"], [0.35, 0.25, 0.15, 0.1, 0.15])[0]
    radius = {"power_cord": rng.uniform(0.0035, 0.0045), "usb": rng.uniform(0.0015, 0.0025),
              "ethernet": rng.uniform(0.0025, 0.003), "av": rng.uniform(0.002, 0.003),
              "bare_wire": rng.uniform(0.001, 0.002)}[kind]
    colour = {"power_cord": rng.choice([(0.03, 0.03, 0.03), (0.85, 0.85, 0.82), (0.5, 0.5, 0.5)]),
              "usb": rng.choice([(0.03, 0.03, 0.03), (0.9, 0.9, 0.9)]),
              "ethernet": rng.choice([(0.1, 0.25, 0.7), (0.75, 0.7, 0.2), (0.6, 0.6, 0.6), (0.7, 0.15, 0.1)]),
              "av": rng.choice([(0.75, 0.65, 0.05), (0.7, 0.1, 0.1), (0.9, 0.9, 0.9)]),
              "bare_wire": rng.choice([(0.7, 0.1, 0.05), (0.05, 0.05, 0.05), (0.8, 0.45, 0.25)])}[kind]
    ins = mat("insulation", jitter(colour, 0.04), 0.9 if kind == "bare_wire" and colour == (0.8, 0.45, 0.25)
              else 0.0, rng.uniform(0.35, 0.7))
    length = rng.uniform(0.5, 2.2)
    n = max(6, int(length / 0.12))
    step = length / n
    ang = rng.uniform(0, 2 * math.pi)
    x = y = 0.0
    pts = [(0, 0, radius)]
    for _ in range(n):                                          # lies on the ground, gentle bends
        ang += rng.gauss(0, 0.6)
        x += step * math.cos(ang); y += step * math.sin(ang)
        pts.append((x, y, radius + abs(rng.gauss(0, radius * 0.6))))
    parts = [tube_along(pts, radius, ins)]
    plug_m = mat("plug", jitter(colour if kind != "bare_wire" else (0.05, 0.05, 0.05), 0.05), 0, 0.5)
    metal = mat("contact", (0.75, 0.72, 0.65), 0.95, 0.3)
    for end, prev in ((pts[0], pts[1]), (pts[-1], pts[-2])):
        d = mathutils.Vector(end) - mathutils.Vector(prev)
        yaw = math.atan2(d.y, d.x)
        c = mathutils.Vector(end) + d.normalized() * 0.012
        if kind == "power_cord":
            parts.append(box((0.035, 0.03, 0.02), (c.x, c.y, 0.01), plug_m, bevel=0.003, rot=(0, 0, yaw)))
        elif kind in ("usb", "ethernet"):
            parts.append(box((0.03, 0.014, 0.009), (c.x, c.y, 0.005), plug_m, bevel=0.002, rot=(0, 0, yaw)))
            e = c + d.normalized() * 0.02
            parts.append(box((0.014, 0.012, 0.0045), (e.x, e.y, 0.004), metal, rot=(0, 0, yaw)))
        elif kind == "av":
            parts.append(cylinder(0.005, 0.03, (c.x, c.y, 0.005), plug_m, rot=(0, math.pi / 2, yaw)))
        else:                                                   # stripped copper tip
            e = mathutils.Vector(end) + d.normalized() * 0.01
            parts.append(cylinder(radius * 0.6, 0.015, (e.x, e.y, radius), mat("cu", COPPER, 1, 0.3),
                                  rot=(0, math.pi / 2, yaw), verts=8))
    return parts, kind, 2 * radius


def gen_cpu_chip():
    kind = rng.choices(["cpu_lga", "cpu_pga", "ic_qfp", "ic_bga"], [0.4, 0.2, 0.25, 0.15])[0]
    parts = []
    if kind in ("cpu_lga", "cpu_pga"):
        s = 0.0375 if kind == "cpu_lga" else 0.040
        if rng.random() < 0.25:
            s = rng.choice([0.045, 0.0525, 0.0375])             # other socket sizes
        sub = mat("substrate", (1, 1, 1), 0, 0.5,
                  tex_pcb(base=rng.choice([(0.05, 0.3, 0.12), (0.05, 0.18, 0.08), (0.3, 0.2, 0.08)])))
        parts.append(box((s, s, 0.0012), (0, 0, 0), mat("sub_edge", (0.05, 0.25, 0.1), 0, 0.5)))
        parts.append(decal((s, s), (0, 0, 0.0006 + 0.0001), sub))
        ihs_s = s * rng.uniform(0.78, 0.85)
        ihs = mat("ihs", (1, 1, 1), 0.95, rng.uniform(0.2, 0.4), tex_engraved())
        parts.append(box((ihs_s, ihs_s, 0.0025), (0, 0, 0.0018), mat("ihs_side", ALU, 0.95, 0.3), bevel=0.001))
        parts.append(decal((ihs_s * 0.95, ihs_s * 0.95), (0, 0, 0.0031 + 0.0001), ihs))
        if kind == "cpu_pga":
            parts.append(decal((s * 0.95, s * 0.95), (0, 0, -0.0006 - 0.0001),
                               mat("pins", (1, 1, 1), 0.8, 0.35,
                                   tex_vents(base=(0.1, 0.25, 0.1), hole=(0.85, 0.65, 0.3), kind="holes")),
                               normal="-Z"))
    elif kind == "ic_qfp":
        s = rng.uniform(0.010, 0.028)
        sy = s * rng.uniform(0.8, 1.0)                          # body depth (may be rectangular)
        body = mat("epoxy", (0.03, 0.03, 0.035), 0, 0.4)
        parts.append(box((s, sy, 0.0025), (0, 0, 0.0013), body, bevel=0.0003))
        legs = mat("legs", (0.8, 0.8, 0.8), 0.95, 0.3)
        n = rng.randint(6, 18)
        for side in range(4):
            for k in range(n):
                t = (k + 0.5) / n - 0.5
                if side < 2:        # legs along the x edges, attached to the ±y faces
                    x, y = t * s * 0.9, (1 if side == 0 else -1) * (sy / 2 + 0.0008)
                    parts.append(box((s / n * 0.4, 0.0016, 0.0004), (x, y, 0.0006), legs))
                else:               # legs along the y edges, attached to the ±x faces
                    x, y = (1 if side == 2 else -1) * (s / 2 + 0.0008), t * sy * 0.9
                    parts.append(box((0.0016, sy / n * 0.4, 0.0004), (x, y, 0.0006), legs))
        parts.append(decal((s * 0.6, s * 0.35), (0, 0, 0.0026 + 0.0001),
                           mat("ic_print", (1, 1, 1), 0, 0.5, tex_label(bg=(0.03, 0.03, 0.035),
                                                                        fg=(0.6, 0.6, 0.6)))))
    else:
        s = rng.uniform(0.012, 0.03)
        parts.append(box((s, s, 0.0012), (0, 0, 0.0006), mat("bga_sub", (0.05, 0.2, 0.08), 0, 0.5)))
        lid = mat("lid", rng.choice([(0.03, 0.03, 0.035), ALU]), 0.5, 0.35)
        parts.append(box((s * 0.75, s * 0.75, 0.0012), (0, 0, 0.0018), lid, bevel=0.0003))
    return parts, kind, None


def gen_metal_component():
    kind = rng.choices(["heatsink_extruded", "heatsink_tower", "chassis_panel", "bracket", "drive_cage"],
                       [0.3, 0.15, 0.25, 0.15, 0.15])[0]
    finish = rng.choice([ALU, jitter(ALU, 0.04), (0.08, 0.08, 0.09), STEEL])
    metal = mat("metal", finish, 0.9 if finish != (0.08, 0.08, 0.09) else 0.4, rng.uniform(0.25, 0.55))
    parts = []
    if kind == "heatsink_extruded":
        L, W = rng.uniform(0.05, 0.15), rng.uniform(0.04, 0.12)
        base_t, fin_h = rng.uniform(0.003, 0.008), rng.uniform(0.015, 0.045)
        parts.append(box((L, W, base_t), (0, 0, base_t / 2), metal))
        n = rng.randint(8, 28)
        fin_t = max(0.0008, W / n * rng.uniform(0.25, 0.45))
        for k in range(n):
            y = -W / 2 + (k + 0.5) * W / n
            parts.append(box((L, fin_t, fin_h), (0, y, base_t + fin_h / 2), metal))
    elif kind == "heatsink_tower":
        W, D, H = rng.uniform(0.10, 0.14), rng.uniform(0.04, 0.06), rng.uniform(0.12, 0.16)
        n = rng.randint(30, 50)
        for k in range(n):
            parts.append(box((W, D, 0.0004), (0, 0, 0.02 + k * (H - 0.02) / n), metal))
        cu = mat("heatpipe", COPPER, 1.0, 0.3)
        for k in range(4):
            parts.append(cylinder(0.003, H, ((k - 1.5) * W / 5, 0, H / 2 + 0.005), cu, verts=12))
        parts.append(box((0.04, 0.04, 0.012), (0, 0, 0.006), cu))
    elif kind == "chassis_panel":
        L, W, t = rng.uniform(0.25, 0.45), rng.uniform(0.18, 0.4), rng.uniform(0.0008, 0.0012)
        parts.append(box((L, W, t), (0, 0, 0), metal))
        fl = rng.uniform(0.012, 0.03)
        for s in rng.sample([-1, 1], k=rng.randint(1, 2)):      # bent flanges
            parts.append(box((L, t, fl), (0, s * W / 2, fl / 2), metal))
        vw, vh = L * rng.uniform(0.2, 0.5), W * rng.uniform(0.2, 0.5)
        parts.append(decal((vw, vh), (rng.uniform(-0.2, 0.2) * L, rng.uniform(-0.2, 0.2) * W, t / 2 + 0.0002),
                           mat("vents", (1, 1, 1), 0.8, 0.4,
                               tex_vents(base=finish, hole=(0.02, 0.02, 0.02)))))
    elif kind == "bracket":
        L, W, t = rng.uniform(0.08, 0.2), rng.uniform(0.015, 0.05), rng.uniform(0.001, 0.003)
        h = rng.uniform(0.015, 0.05)
        parts.append(box((L, W, t), (0, 0, 0), metal))
        parts.append(box((t, W, h), (L / 2, 0, h / 2), metal))
        if rng.random() < 0.5:
            parts.append(box((t, W, h), (-L / 2, 0, h / 2), metal))
    else:                                                      # 3.5" drive cage
        L, W, H = 0.16, 0.105, rng.choice([0.045, 0.07, 0.1])
        t = 0.0008
        for s in (-1, 1):
            parts.append(box((L, t, H), (0, s * W / 2, H / 2), metal))
        parts.append(box((L, W, t), (0, 0, 0), metal))
        for k in range(rng.randint(1, 3)):
            parts.append(box((L * 0.9, 0.006, t), (0, -W / 2 + 0.003, (k + 1) * H / 3), metal))
    return parts, kind, None


def gen_plastic_casing():
    kind = rng.choices(["open_shell", "monitor_back", "side_panel", "broken_fragment"],
                       [0.35, 0.2, 0.15, 0.3])[0]
    colour = jitter(rng.choice(PLASTICS), 0.04)
    plastic = mat("plastic", colour, 0, rng.uniform(0.3, 0.7))
    parts = []
    if kind in ("open_shell", "broken_fragment"):
        dims = (rng.uniform(0.15, 0.45), rng.uniform(0.12, 0.35), rng.uniform(0.03, 0.12))
        remove = ["+Z"] + (["-X"] if rng.random() < 0.3 else [])
        shell = open_tray(dims, plastic, rng.uniform(0.002, 0.004), remove=tuple(remove))
        if kind == "broken_fragment":
            shell = break_mesh(shell, rng.randint(1, 2))
        parts.append(shell)
        if rng.random() < 0.6:
            parts.append(decal((dims[0] * 0.4, dims[1] * 0.3),
                               (0, 0, -dims[2] / 2 - 0.0003),
                               mat("vents", (1, 1, 1), 0, 0.6,
                                   tex_vents(base=colour, hole=(0.02, 0.02, 0.02), kind="slots")),
                               normal="-Z"))
    elif kind == "monitor_back":
        w, h, d = rng.uniform(0.35, 0.55), rng.uniform(0.25, 0.38), rng.uniform(0.02, 0.06)
        parts.append(open_tray((w, h, d), plastic, 0.0025))
        parts.append(box((0.1, 0.1, 0.004), (0, 0, -d / 2 - 0.002), plastic))   # VESA / stand boss
        parts.append(decal((w * 0.5, h * 0.15), (0, h * 0.3, -d / 2 - 0.0003),
                           mat("vents", (1, 1, 1), 0, 0.6,
                               tex_vents(base=colour, hole=(0.02, 0.02, 0.02), kind="slots")), normal="-Z"))
    else:
        L, W = rng.uniform(0.3, 0.45), rng.uniform(0.15, 0.2)
        parts.append(box((L, W, 0.003), (0, 0, 0), plastic, bevel=0.001))
        parts.append(box((L, 0.003, 0.02), (0, W / 2, 0.01), plastic))
        parts.append(decal((L * 0.3, W * 0.6), (L * 0.25, 0, 0.0018),
                           mat("vents", (1, 1, 1), 0, 0.6,
                               tex_vents(base=colour, hole=(0.02, 0.02, 0.02)))))
    return parts, kind, None


GENERATORS = {
    "hard_drive": gen_hard_drive, "power_supply": gen_power_supply,
    "single_cable": gen_single_cable, "cpu_chip": gen_cpu_chip,
    "metal_component": gen_metal_component, "plastic_casing": gen_plastic_casing,
}
# Real, manufactured sizes vary little between units of the same variant, so
# the renderer should not rescale these by the generic 0.7-1.4x jitter.
SCALE_JITTER = {"hard_drive": [0.98, 1.02], "power_supply": [0.95, 1.05],
                "single_cable": [0.85, 1.15], "cpu_chip": [0.97, 1.03],
                "metal_component": [0.85, 1.15], "plastic_casing": [0.85, 1.15]}


def export_glb(ob, path: Path):
    vl = bpy.context.view_layer
    for o in vl.objects:
        o.select_set(False)
    ob.select_set(True)
    vl.objects.active = ob
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=True,
                              export_apply=True, export_image_format="AUTO")


def next_index(cls_dir: Path, cls: str) -> int:
    nums = [int(m.group(1)) for p in cls_dir.glob(f"proc_{cls}_*.glb")
            if (m := re.match(rf"proc_{cls}_(\d+)", p.stem))]
    return max(nums) + 1 if nums else 0


def main():
    ap = argparse.ArgumentParser(description="Procedural models for thin e-waste classes")
    ap.add_argument("--models_dir", default="./assets/models")
    ap.add_argument("--classes", nargs="+", choices=CLASSES, default=CLASSES)
    ap.add_argument("--n", type=int, default=40, help="models to add per class")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    global rng
    rng = random.Random(args.seed)
    bproc.init()
    models_dir = Path(args.models_dir)

    summary = {}
    for cls in args.classes:
        cls_dir = models_dir / cls
        start = next_index(cls_dir, cls)
        variants = {}
        for k in range(args.n):
            bproc.clean_up()
            out = GENERATORS[cls]()
            parts, variant, min_feature = out[:3]
            body_m = out[3] if len(out) > 3 else None
            ob = join(parts, f"proc_{cls}")
            dims = ob.dimensions
            name = f"proc_{cls}_{start + k:04d}"
            export_glb(ob, cls_dir / f"{name}.glb")
            (cls_dir / f"{name}.json").write_text(json.dumps({
                "real_size_m": round(max(dims), 4),
                "dims_m": [round(d, 4) for d in dims],
                "scale_jitter": SCALE_JITTER[cls],
                "min_feature_m": None if min_feature is None else round(min_feature, 4),
                # solid body / full extent: lets the renderer gate on what is
                # actually visible from altitude (e.g. brick body, not its cable)
                "gate_frac": 1.0 if body_m is None else round(min(1.0, body_m / max(dims)), 4),
                "variant": variant, "generator": GENERATOR_VERSION, "seed": args.seed,
            }, indent=2))
            variants[variant] = variants.get(variant, 0) + 1
            print(f"  {name}.glb  {variant:<20} {dims[0]*1000:6.1f} x {dims[1]*1000:6.1f} x "
                  f"{dims[2]*1000:6.1f} mm")
        summary[cls] = variants
    print("\nGenerated:", json.dumps(summary, indent=2))
    print("Next: render thumbnails for review (HANDOFF.md section 7) before rendering scenes.")


main()
