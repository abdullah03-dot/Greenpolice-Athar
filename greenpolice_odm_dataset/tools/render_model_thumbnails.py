import blenderproc as bproc
# ^ BlenderProc requires this to be the very first line of the file
"""
Render one thumbnail per 3D model in assets/models/<class>/ for manual review.

Objaverse keyword search is noisy (e.g. a plate of cookies matched "flat panel
display"), so every fetched model should be eyeballed before it is used to
generate labelled training data.

Usage:
  blenderproc run tools/render_model_thumbnails.py \\
    --models_dir ./assets/models --out_dir ./assets/models/_review
Then:
  python tools/make_review_sheets.py --thumbs_dir ./assets/models/_review
"""
import argparse
from pathlib import Path

import bpy
import numpy as np
from PIL import Image


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
    """Same idea as EWasteSceneBuilder._merge_loaded_model: meshes only, joined."""
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
    vl = bpy.context.view_layer
    # Removing objects leaves stale None entries until the view layer refreshes
    vl.update()
    for o in vl.objects:
        if o is not None:
            o.select_set(False)
    for m in meshes:
        m.select_set(True)
    vl.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    return bproc.types.MeshObject(meshes[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="./assets/models")
    ap.add_argument("--out_dir", default="./assets/models/_review")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--debug", action="store_true", help="stop at first failure with traceback")
    ap.add_argument("--skip_reviewed", default=None,
                    help="index.json from an earlier review; models listed there are skipped")
    args = ap.parse_args()

    models_dir, out_dir = Path(args.models_dir), Path(args.out_dir)
    bproc.init()
    # Don't keep render data (incl. the image cache) between renders
    bpy.context.scene.render.use_persistent_data = False
    bproc.renderer.set_max_amount_of_samples(32)
    bproc.renderer.set_noise_threshold(0.1)
    bproc.camera.set_resolution(args.size, args.size)

    files = sorted(p for p in models_dir.glob("*/*")
                   if p.suffix.lower() in {".glb", ".obj", ".fbx"}
                   and not p.parent.name.startswith("_"))
    if args.skip_reviewed:
        import json
        reviewed = {stem for tiles in json.loads(Path(args.skip_reviewed).read_text()).values()
                    for stem in tiles.values()}
        files = [f for f in files if f.stem not in reviewed]
    print(f"{len(files)} models to render")

    for i, f in enumerate(files):
        dst = out_dir / f.parent.name / f"{f.stem}.png"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        bproc.clean_up()
        try:
            obj = merge_loaded(load_model_unique(str(f)))
            if obj is None:
                raise RuntimeError("no mesh")
            obj.set_location([0, 0, 0])
            bb = obj.get_bound_box()
            extent = float((bb.max(axis=0) - bb.min(axis=0)).max()) or 1.0
            obj.set_scale([1.0 / extent] * 3)   # fit to a unit cube

            world = bpy.context.scene.world
            world.use_nodes = True
            world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
            world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
            sun = bproc.types.Light()
            sun.set_type("SUN")
            sun.set_energy(3)
            sun.set_rotation_euler([0.8, 0.2, 0.6])

            cam_pos = np.array([1.2, -1.2, 0.9])
            rot = bproc.camera.rotation_from_forward_vec(-cam_pos)
            bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam_pos, rot))
            bproc.camera.set_intrinsics_from_blender_params(lens=0.75, lens_unit="FOV")

            img = bproc.renderer.render()["colors"][0]
            Image.fromarray(img).save(dst)
            print(f"[{i+1}/{len(files)}] ok   {f.parent.name}/{f.name}")
        except Exception as e:
            Image.new("RGB", (args.size, args.size), (90, 0, 0)).save(dst)
            print(f"[{i+1}/{len(files)}] FAIL {f.parent.name}/{f.name}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
                break


main()
