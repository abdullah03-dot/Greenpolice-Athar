import blenderproc as bproc
# ^ BlenderProc requires this to be the very first line of the file
"""
E-Waste Detection Pipeline — Phase 1b: BlenderProc Synthetic Data Generation v3
================================================================================
v3 (2026-09-28) fixes the three blockers in HANDOFF.md section 5:

  [1] GROUND. The terrain was one flat colour: cc_textures were never
      downloaded, the tag matching could not match ambientCG file names, and
      the untextured Displace modifier only lifted the plane uniformly.
      Now: a subdivided terrain mesh with real low-frequency relief, a PBR
      material that blends two ambientCG ground scans at their physical scale
      with macro colour variation (tools/fetch_ground_textures.py), and a
      procedural multi-tone fallback that is never flat. Plus NON-E-WASTE
      CLUTTER (rocks, shrubs, sticks, tyres, boxes, bags, bottles, rope/hose,
      blocks) that is rendered but never labelled, and optional true-negative
      scenes with no e-waste at all.

  [2] SMALL OBJECTS. Phones/chips were rendered at altitudes where they are
      2-5 px (a 4 cm chip at 60 m is under 1 px), then dropped by
      MIN_PIXEL_AREA — so they appeared in images WITHOUT labels, teaching the
      detector that e-waste is background. Now:
        • camera = real DJI Phantom 4 Pro optics (1" sensor 13.2 x 8.8 mm,
          fixed 8.8 mm lens, ~73.7° HFOV) instead of 17.3 mm + 35-90° FOV;
        • ground sampling distance is computed BEFORE objects are placed and
          a model is only spawned if it will be >= SPAWN_MIN_PX long (and
          thin parts such as cables >= MIN_FEATURE_PX wide) at this altitude
          and resolution — tiny classes appear only where they are resolvable;
        • every visible e-waste object with >= LABEL_MIN_VISIBLE_PX pixels is
          labelled; the rest (heavily occluded slivers) are counted in
          metadata as visible_unlabelled so you can audit them;
        • objects are scattered inside the camera footprint, so low-altitude
          scenes are no longer mostly empty;
        • bbox width/height off-by-one fixed (a 1-px-wide object had area 0).

  [3] THIN CLASSES. Models may now carry a sidecar <model>.json with their
      true size (written by tools/generate_procedural_models.py), used
      instead of the class-wide CLASS_REAL_SIZE_M.

  Removed no-ops (they changed metadata but not pixels): world mist only
  writes a Mist render pass in Cycles, not visible haze; use_motion_blur has
  no effect on a single static frame. Both effects are applied in
  03_augmentation_pipeline.py (RandomFog, MotionBlur) — don't claim them for
  the render stage in the paper.

Kept from v2 (laptop bug-fix round, see HANDOFF.md section 4): glTF empty-root
merge, real-world rescaling, unique image names (texture leak), per-scene
pass_index, keyframe reset, Nishita sky, single bproc.init(), skip logging.

Prerequisites:
  python tools/fetch_ground_textures.py            # ~40 curated ambientCG grounds
  blenderproc run tools/generate_procedural_models.py --n 40
  (optional) python tools/fetch_distractors.py     # Objaverse-LVIS clutter models

Usage:
  blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 5000 --environment all

  # Quick visual check (small images, few samples):
  blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 6 --environment desert \\
      --output_dir /tmp/ewads_test --res_scale 0.25 --samples 32
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import bmesh
import bpy
import mathutils
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ewads_survey as es   # sequence geometry shared with 11's CoverageScheduler (numpy only)


# ─── Taxonomy ─────────────────────────────────────────────────────────────────
EWASTE_CLASSES = [
    "crt_monitor",    # 00
    "flat_screen",    # 01
    "circuit_board",  # 02
    "battery",        # 03
    "cable_bundle",   # 04
    "single_cable",   # 05
    "mobile_phone",   # 06
    "keyboard",       # 07
    "printer",        # 08
    "small_appliance",# 09
    "large_appliance",# 10
    "cpu_chip",       # 11
    "hard_drive",     # 12
    "power_supply",   # 13
    "ewaste_pile",    # 14
    "plastic_casing", # 15
    "metal_component",# 16
    "lighting",       # 17
]

# ─── Class sampling weights (forensic tiers) ─────────────────────────────────
CLASS_SAMPLING_WEIGHTS = {
    "crt_monitor":    1.4,   # Tier 1 — toxic
    "flat_screen":    1.0,
    "circuit_board":  1.4,   # Tier 1 — toxic
    "battery":        1.6,   # Tier 1 — highest toxicity
    "cable_bundle":   1.2,
    "single_cable":   0.8,
    "mobile_phone":   1.2,
    "keyboard":       0.9,
    "printer":        0.9,
    "small_appliance":0.9,
    "large_appliance":1.1,
    "cpu_chip":       1.0,
    "hard_drive":     1.0,
    "power_supply":   1.0,
    "ewaste_pile":    1.5,   # Common in real dumps
    "plastic_casing": 0.8,
    "metal_component":0.8,
    "lighting":       0.9,
}
# Small classes are only eligible in low-altitude / high-resolution scenes, so
# they get a boost in the scenes where they CAN appear. Tune from the per-class
# label counts printed at the end of each run (generation_stats.json).
SMALL_CLASS_BOOST = {
    "cpu_chip": 4.0, "mobile_phone": 1.8, "single_cable": 2.0, "hard_drive": 1.6,
    "power_supply": 1.4, "battery": 1.3, "circuit_board": 1.2,
}

# ─── Adaptive class balance (HANDOFF item 4) ──────────────────────────────────
# Fixed weights can't balance the dataset: imbalance comes from ELIGIBILITY (a
# pile is resolvable in every scene, a 4 cm chip in a few percent of them).
# The sampler therefore tracks labels actually written per class (seeded from
# labels already on disk, so resumed runs keep balancing) and multiplies each
# class's weight by (target share / observed share) ** BALANCE_ALPHA.
# Target share = CLASS_SAMPLING_WEIGHTS normalised (the forensic tiers).
BALANCE_ALPHA = 1.0          # 0 disables (= fixed weights; use --no_balance for the ablation)
BALANCE_CLAMP = (0.1, 10.0)   # limits on the correction factor
BALANCE_PRIOR = 30           # pseudo-labels per class: damps corrections in the first scenes
_LABEL_COUNTS = Counter()    # shared by all environments in one process

# ─── Resolvability rules (pixels are NATIVE render pixels) ────────────────────
# These assume training uses native-resolution tiles (see note in summary).
SPAWN_MIN_PX         = 16   # model's longest side must reach this to be placed at all
MIN_FEATURE_PX       = 1.0  # thin parts (cable diameter) must reach this width
SPAWN_MAX_FRAC       = 0.6  # skip models longer than this fraction of the short frame side
LABEL_MIN_VISIBLE_PX = 12   # visible (unoccluded) pixels needed to write a label
LABEL_MIN_SIDE_PX    = 4    # visible bbox's longer side must reach this
# Default thin-feature size per class when a model has no sidecar
CLASS_MIN_FEATURE_M = {"single_cable": 0.005}

# ─── Altitude profiles ────────────────────────────────────────────────────────
# (min_m, max_m, objects_per_scene_range)
ALTITUDE_PROFILES = {
    "high_surveillance": (40, 80, (5, 20)),
    "mid_confirmation":  (15, 40, (3, 10)),
    "low_evidence":      (3,  15, (1, 5)),
}
ALTITUDE_WEIGHTS = {
    "high_surveillance": 0.40,
    "mid_confirmation":  0.35,
    "low_evidence":      0.25,
}

# ─── Camera: DJI Phantom 4 Pro (fixed-focal 1" sensor) ────────────────────────
# 1" CMOS is 13.2 x 8.8 mm; the lens is a fixed 8.8 mm (24 mm equiv, 84° diag).
# HFOV = 2*atan(6.6/8.8) ≈ 73.7°. The earlier 17.3 mm width is Micro Four
# Thirds, and the 35-90° FOV range was a zoom the P4P doesn't have.
CAMERA_PROFILES = {
    "dji_phantom4_pro": {
        "sensor_width_mm": 13.2,
        "focal_mm": 8.8,
        # (resolution, weight): 4K video, 20 MP 3:2 stills, 1080p video
        "resolutions": [((3840, 2160), 0.55), ((5472, 3648), 0.30), ((1920, 1080), 0.15)],
        "tilt_sigma_deg": 8.0,
    },
    # ABLATION ONLY (2026-10-08): same sensor and resolutions, 18.33 mm lens = 50 mm 35mm-equivalent
    # (P4P crop factor 43.27/15.86 = 2.727; the real lens is 24 mm-equivalent).  Objects render ~2.1x
    # larger per altitude than the real camera sees -- tests whether correct geometry matters.
    "p4p_mismatch_50mm_equiv": {
        "sensor_width_mm": 13.2,
        "focal_mm": 18.333,
        "resolutions": [((3840, 2160), 0.55), ((5472, 3648), 0.30), ((1920, 1080), 0.15)],
        "tilt_sigma_deg": 8.0,
    },
}
DEFAULT_CAMERA = "dji_phantom4_pro"

# Capture mode of each native resolution (recorded per scene; dataset statistics / per-mode results)
CAPTURE_MODE_BY_RES = {(3840, 2160): "video_4k", (5472, 3648): "stills_3x2", (1920, 1080): "video_1080p"}

# ─── Environments ─────────────────────────────────────────────────────────────
ENVIRONMENT_CONFIGS = {
    "desert": {
        "ground_color_range": {"r": (0.6, 0.9), "g": (0.5, 0.8), "b": (0.2, 0.5)},
        "sun_elevation_range": (15, 85),
        "sun_strength_range":  (3.0, 8.0),
        "ambient_strength_range": (0.1, 0.4),
        "relief_m": (0.2, 1.2), "relief_wavelength_m": (15, 45),
        "scatter_pattern": "clustered",
        "scene_fraction": 0.50,
        # distractor type -> instances per 100 m² of ground
        "distractors": {"rock": 2.5, "shrub": 0.6, "stick": 0.5, "tyre": 0.06,
                        "box": 0.10, "bag": 0.25, "bottle": 0.30, "rope": 0.06, "block": 0.05},
        "shrub_palette": [(0.35, 0.3, 0.15), (0.3, 0.32, 0.12), (0.45, 0.38, 0.22)],
    },
    "european_urban": {
        "ground_color_range": {"r": (0.3, 0.6), "g": (0.3, 0.6), "b": (0.3, 0.6)},
        "sun_elevation_range": (10, 55),
        "sun_strength_range":  (0.5, 3.0),
        "ambient_strength_range": (0.5, 1.2),
        "relief_m": (0.0, 0.08), "relief_wavelength_m": (20, 60),
        "scatter_pattern": "linear",
        "scene_fraction": 0.25,
        "distractors": {"rock": 0.4, "shrub": 0.3, "stick": 0.2, "tyre": 0.08,
                        "box": 0.35, "bag": 0.5, "bottle": 0.6, "rope": 0.06, "block": 0.35},
        "shrub_palette": [(0.1, 0.25, 0.06), (0.15, 0.3, 0.08), (0.25, 0.3, 0.1)],
    },
    "european_rural": {
        "ground_color_range": {"r": (0.1, 0.4), "g": (0.2, 0.5), "b": (0.1, 0.3)},
        "sun_elevation_range": (5, 50),
        "sun_strength_range":  (0.3, 2.5),
        "ambient_strength_range": (0.8, 1.5),
        "relief_m": (0.1, 0.6), "relief_wavelength_m": (10, 35),
        "scatter_pattern": "scattered",
        "scene_fraction": 0.25,
        "distractors": {"rock": 0.8, "shrub": 1.8, "stick": 0.8, "log": 0.15, "tyre": 0.05,
                        "box": 0.08, "bag": 0.15, "bottle": 0.15, "rope": 0.05, "block": 0.03},
        "shrub_palette": [(0.06, 0.2, 0.04), (0.1, 0.28, 0.05), (0.2, 0.26, 0.08)],
    },
}
# Share of scenes lit by a real HDRI (when tools/fetch_hdris.py has run);
# the rest use the procedural Nishita sky + sun lamp, for lighting diversity
HDRI_PROBABILITY = 0.6
HDRI_STRENGTH_RANGE = (0.6, 1.6)

MAX_DISTRACTORS_PER_TYPE = 160
MAX_DISTRACTORS_TOTAL = 500
DISTRACTOR_MIN_PX = 3            # don't bother placing clutter smaller than this

# Approximate real-world longest dimension per class, in metres. Objaverse
# models use arbitrary units, so every loaded model is rescaled to this
# (unless it has a sidecar <model>.json with its own real_size_m).
CLASS_REAL_SIZE_M = {
    "crt_monitor": 0.45, "flat_screen": 0.60, "circuit_board": 0.25,
    "battery": 0.25, "cable_bundle": 0.50, "single_cable": 1.00,
    "mobile_phone": 0.15, "keyboard": 0.45, "printer": 0.50,
    "small_appliance": 0.35, "large_appliance": 1.70, "cpu_chip": 0.04,
    "hard_drive": 0.15, "power_supply": 0.16, "ewaste_pile": 2.50,
    "plastic_casing": 0.40, "metal_component": 0.30, "lighting": 1.20,
}
DEFAULT_SCALE_JITTER = (0.7, 1.4)

# Classes whose models span very different real sizes. Without this every
# `lighting` model (mostly Objaverse-LVIS "lightbulb") was scaled to 1.2 m
# fluorescent-tube length, and AA cells in `battery` to 25 cm. Rules use the
# model's own proportions: a = longest/middle side, b = middle/shortest side.
# First matching rule wins -> (real longest side m, scale jitter).
CLASS_SIZE_BY_SHAPE = {
    "lighting": [
        (lambda a, b: a >= 4.0, 1.20, (0.5, 1.25)),   # fluorescent tube / LED batten
        (lambda a, b: True,     0.12, (0.8, 1.6)),    # bulb, CFL, LED lamp
    ],
    "battery": [
        (lambda a, b: a >= 2.2 and b < 1.5, 0.055, (0.8, 1.3)),  # AA / 18650 cylindrical cell
        (lambda a, b: True,                 0.25,  (0.6, 1.3)),  # pack, 9V, car / UPS battery
    ],
}

MATERIAL_WEATHERING = {
    "rust_probability": 0.60,
    "fade_probability": 0.70,
    "burn_probability": 0.20,
    "roughness_range": (0.4, 1.0),
}


# ─── Small helpers ────────────────────────────────────────────────────────────
def hfov_rad(cam: dict) -> float:
    return 2 * math.atan(cam["sensor_width_mm"] / 2 / cam["focal_mm"])


def gsd_m_per_px(altitude_m: float, width_px: int, cam: dict) -> float:
    """Nadir ground sampling distance (metres per pixel)."""
    return 2 * altitude_m * math.tan(hfov_rad(cam) / 2) / width_px


def _mix(nt, kind, fac, a, b):
    """Blender 4.x Mix node. kind 'RGBA' or 'FLOAT'. Sockets by verified index."""
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = kind
    ia, ib, out = (6, 7, 2) if kind == "RGBA" else (2, 3, 0)
    for src, idx in ((fac, 0), (a, ia), (b, ib)):
        if isinstance(src, (int, float)):
            n.inputs[idx].default_value = src
        elif isinstance(src, tuple):
            n.inputs[idx].default_value = src
        else:
            nt.links.new(src, n.inputs[idx])
    return n.outputs[out]


def _world_bbox(ob):
    corners = [ob.matrix_world @ mathutils.Vector(c) for c in ob.bound_box]
    return (min(c.x for c in corners), max(c.x for c in corners),
            min(c.y for c in corners), max(c.y for c in corners),
            min(c.z for c in corners), max(c.z for c in corners))


def _read_sidecar(path: Path) -> dict:
    side = path.with_suffix(".json")
    if side.exists():
        try:
            return json.loads(side.read_text())
        except Exception:
            pass
    return {}


# ─── Scene Builder ─────────────────────────────────────────────────────────────
class EWasteSceneBuilder:

    def __init__(self, models_dir: Path, textures_dir: Path, hdri_dir: Path,
                 output_dir: Path, environment: str = "desert",
                 dry_run: bool = False, res_scale: float = 1.0,
                 distractors_dir: Optional[Path] = None,
                 negative_fraction: float = 0.0, camera="dji_phantom4_pro",
                 sequence_frames: int = 1, sequence_views: int = 3, sequence_max_object_m: float = 1.0):
        self.models_dir = models_dir
        # >1: render an overlapping along-track SEQUENCE per scene (for 11's synthetic validation)
        self.sequence_frames = max(1, int(sequence_frames))
        self.sequence_views = sequence_views
        self.sequence_max_object_m = sequence_max_object_m
        self.textures_dir = textures_dir
        self.hdri_dir = hdri_dir
        self.output_dir = output_dir
        self.environment = environment
        self.env_config = ENVIRONMENT_CONFIGS[environment]
        self.dry_run = dry_run
        self.res_scale = res_scale
        self.negative_fraction = negative_fraction
        # camera: a CAMERA_PROFILES name or a full profile dict (--override_* custom optics)
        self.cam = CAMERA_PROFILES[camera] if isinstance(camera, str) else camera
        self.camera_name = camera if isinstance(camera, str) else camera.get("name", "custom")

        self.img_dir  = output_dir / "images"
        self.lbl_dir  = output_dir / "labels"
        self.meta_dir = output_dir / "metadata"
        self.index_path = output_dir / "scene_index.jsonl"
        for d in [self.img_dir, self.lbl_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._model_index = self._build_model_index()
        self._ground_assets = self._index_ground_textures()
        self._asset_distractors = self._index_asset_distractors(distractors_dir)
        self.stats = defaultdict(Counter)
        self.balance_alpha = BALANCE_ALPHA
        self._seed_label_counts()

    # ── Asset indexing ────────────────────────────────────────────────────────
    def _build_model_index(self) -> dict:
        """{class: [{path, size_m, jitter, min_feature_m}]} — sidecars override class defaults."""
        index = {cls: [] for cls in EWASTE_CLASSES}
        for cls in EWASTE_CLASSES:
            cls_dir = self.models_dir / cls
            if not cls_dir.exists():
                continue
            for f in sorted(list(cls_dir.glob("**/*.obj")) + list(cls_dir.glob("**/*.fbx"))
                            + list(cls_dir.glob("**/*.glb"))):
                meta = _read_sidecar(f)
                index[cls].append({
                    "path": f,
                    "size_m": float(meta.get("real_size_m") or CLASS_REAL_SIZE_M[cls]),
                    "jitter": tuple(meta.get("scale_jitter") or DEFAULT_SCALE_JITTER),
                    "min_feature_m": meta.get("min_feature_m", CLASS_MIN_FEATURE_M.get(cls)),
                    "has_sidecar": bool(meta),
                    # fraction of the extent that is solid body (e.g. a laptop brick
                    # whose extent includes a thin DC cable); gating uses body size
                    "gate_frac": float(meta.get("gate_frac", 1.0)),
                })
        return index

    def _has_models(self) -> bool:
        return any(self._model_index.values())

    def _index_ground_textures(self) -> list:
        """
        ambientCG sets from tools/fetch_ground_textures.py:
          <textures_dir>/ground/<environment>/<AssetID>/<AssetID>_2K-JPG_Color.jpg (+ maps, asset.json)
        """
        root = self.textures_dir / "ground" / self.environment
        assets = []
        for color in sorted(root.glob("*/*_Color.jpg")) if root.exists() else []:
            maps = {k: color.with_name(color.name.replace("_Color", f"_{k}"))
                    for k in ("Roughness", "NormalGL", "AmbientOcclusion")}
            meta = _read_sidecar(color.parent / "asset.json")
            assets.append({"id": color.parent.name, "color": color,
                           "maps": {k: v for k, v in maps.items() if v.exists()},
                           "size_m": meta.get("real_size_m")})
        if not assets:
            print(f"  ⚠  No ground textures in {root} — using the procedural fallback ground.\n"
                  f"     Run: python tools/fetch_ground_textures.py")
        return assets

    @staticmethod
    def _index_asset_distractors(root: Optional[Path]) -> list:
        """Optional Objaverse clutter from tools/fetch_distractors.py: <root>/<category>/*.glb"""
        out = []
        if root and root.exists():
            for f in sorted(root.glob("*/*.glb")):
                if f.parent.name.startswith("_"):
                    continue
                meta = _read_sidecar(f)
                out.append({"path": f, "category": f.parent.name,
                            "size_m": float(meta.get("real_size_m", 0.5))})
        return out

    # ── Scene geometry: altitude, resolution, footprint ──────────────────────
    def _plan_view(self, altitude_profile: str) -> dict:
        alt_min, alt_max, _ = ALTITUDE_PROFILES[altitude_profile]
        altitude = random.uniform(alt_min, alt_max)
        res_choices, res_w = zip(*self.cam["resolutions"])
        W0, H0 = random.choices(res_choices, weights=res_w)[0]
        W, H = max(64, int(W0 * self.res_scale)), max(36, int(H0 * self.res_scale))
        gsd = gsd_m_per_px(altitude, W, self.cam)
        return {"altitude_m": altitude, "resolution": (W, H), "gsd": gsd,
                "capture_mode": CAPTURE_MODE_BY_RES.get((W0, H0), f"{W0}x{H0}"),
                "footprint_m": (gsd * W, gsd * H),
                "yaw_deg": random.uniform(0, 360),
                "tilt_deg": (random.gauss(0, self.cam["tilt_sigma_deg"]),
                             random.gauss(0, 3.0))}

    def _eligible_models(self, cls: str, view: dict) -> list:
        gsd, short_side = view["gsd"], min(view["footprint_m"])
        ok = []
        for m in self._model_index[cls]:
            lo, hi = m["jitter"]
            if m["size_m"] * m["gate_frac"] * lo / gsd < SPAWN_MIN_PX:
                continue
            if m["size_m"] * hi > SPAWN_MAX_FRAC * short_side:
                continue
            if m["min_feature_m"] and m["min_feature_m"] / gsd < MIN_FEATURE_PX:
                continue
            ok.append(m)
        return ok

    def _sample_classes_for_scene(self, n_objects: int, view: dict) -> list:
        eligible = {c: self._eligible_models(c, view) for c in EWASTE_CLASSES}
        eligible = {c: ms for c, ms in eligible.items() if ms}
        if not eligible:
            return []
        classes = list(eligible)
        corr = self._balance_factors()
        w = np.array([CLASS_SAMPLING_WEIGHTS.get(c, 1.0) * SMALL_CLASS_BOOST.get(c, 1.0) * corr[c]
                      for c in classes])
        picks = [str(c) for c in np.random.choice(classes, size=n_objects, replace=True, p=w / w.sum())]
        # Reweighting can't stop over-represented classes from filling scenes
        # where they are the ONLY resolvable ones (high altitude -> piles,
        # appliances). So thin them: keep each pick with probability = its
        # correction factor (<1 only for over-represented classes). The scene
        # gets fewer objects; clutter still fills it.
        if self.balance_alpha > 0:
            kept = [c for c in picks if random.random() < min(1.0, corr[c])]
            picks = kept or [max(picks, key=lambda c: corr[c])]
        self._eligible_cache = eligible
        return [(c, random.choice(eligible[c])) for c in picks]

    # ── [ITEM 4] Adaptive class balance ──────────────────────────────────────
    def _target_shares(self) -> dict:
        have = [c for c in EWASTE_CLASSES if self._model_index[c]]
        tot = sum(CLASS_SAMPLING_WEIGHTS[c] for c in have)
        return {c: (CLASS_SAMPLING_WEIGHTS[c] / tot if c in have else 0.0) for c in EWASTE_CLASSES}

    def _balance_factors(self) -> dict:
        if self.balance_alpha <= 0:
            return {c: 1.0 for c in EWASTE_CLASSES}
        t = self._target_shares()
        k = len([c for c in t if t[c] > 0])
        smoothed = {c: _LABEL_COUNTS[c] + BALANCE_PRIOR * k * t[c] for c in EWASTE_CLASSES}
        total = sum(smoothed.values()) or 1.0
        out = {}
        for c in EWASTE_CLASSES:
            if t[c] <= 0:
                out[c] = 1.0
                continue
            f = (t[c] / (smoothed[c] / total)) ** self.balance_alpha
            out[c] = min(BALANCE_CLAMP[1], max(BALANCE_CLAMP[0], f))
        return out

    def _seed_label_counts(self):
        """Count labels already written in this output dir (resumed / earlier runs)."""
        n = 0
        for f in self.lbl_dir.glob("*.txt"):
            for line in f.read_text().splitlines():
                parts = line.split()
                if parts and parts[0].isdigit() and int(parts[0]) < len(EWASTE_CLASSES):
                    _LABEL_COUNTS[EWASTE_CLASSES[int(parts[0])]] += 1
                    n += 1
        if n:
            print(f"  Class balance seeded from {n} existing labels in {self.lbl_dir}")

    # ── Lighting ─────────────────────────────────────────────────────────────
    @staticmethod
    def _world():
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        world.use_nodes = True        # clean_up() creates a world WITHOUT a node tree
        return world

    def _set_nishita_sky(self, elevation_deg: float, azimuth_deg: float):
        world = self._world()
        nodes, links = world.node_tree.nodes, world.node_tree.links
        nodes.clear()
        sky = nodes.new("ShaderNodeTexSky")
        sky.sky_type = "NISHITA"
        sky.sun_elevation = math.radians(elevation_deg)
        # Same sun position as the SUN lamp, so sky colour and shadows agree
        sky.sun_rotation = math.radians(azimuth_deg)
        sky.sun_disc = False
        bg = nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = random.uniform(0.1, 0.3)
        out = nodes.new("ShaderNodeOutputWorld")
        links.new(sky.outputs["Color"], bg.inputs["Color"])
        links.new(bg.outputs["Background"], out.inputs["Surface"])

    def _hdri_pool(self) -> list:
        """HDRIs from tools/fetch_hdris.py: <hdri_dir>/<environment>/*.hdr, else any under hdri_dir."""
        if not hasattr(self, "_hdris"):
            env_dir = self.hdri_dir / self.environment
            root = env_dir if env_dir.exists() else self.hdri_dir
            self._hdris = (sorted(root.glob("**/*.hdr")) + sorted(root.glob("**/*.exr"))
                           if root.exists() else [])
        return self._hdris

    def _randomize_lighting(self, scene_meta: dict):
        env = self.env_config
        hdris = self._hdri_pool()
        if hdris and random.random() < HDRI_PROBABILITY:
            # Outdoor HDRIs already contain the sun and sky fill. Adding the SUN
            # lamp + AREA fill on top double-lights the scene (the same blow-out
            # the v2 round fixed for the Nishita sky), so HDRI scenes use the
            # HDRI alone. Rotation moves the sun direction; strength = exposure.
            self._world()
            hdri = random.choice(hdris)
            rot = random.uniform(0, 2 * math.pi)
            strength = random.uniform(*HDRI_STRENGTH_RANGE)
            bproc.world.set_world_background_hdr_img(str(hdri), strength=strength,
                                                     rotation_euler=[0.0, 0.0, rot])
            scene_meta["lighting"] = {"type": "hdri", "hdri": hdri.name,
                                      "rotation_deg": round(math.degrees(rot), 1),
                                      "strength": round(strength, 2)}
            return

        sun_elevation = random.uniform(*env["sun_elevation_range"])
        sun_azimuth   = random.uniform(0, 360)
        sun_strength  = random.uniform(*env["sun_strength_range"])
        self._set_nishita_sky(sun_elevation, sun_azimuth)
        v = mathutils.Vector((math.cos(math.radians(sun_azimuth)) * math.cos(math.radians(sun_elevation)),
                              math.sin(math.radians(sun_azimuth)) * math.cos(math.radians(sun_elevation)),
                              math.sin(math.radians(sun_elevation))))
        sun = bproc.types.Light()
        sun.set_type("SUN")
        sun.set_rotation_euler(v.to_track_quat("Z", "Y").to_euler())   # SUN shines along local -Z
        sun.set_energy(sun_strength)
        sun.set_color(self._sun_color_from_elevation(sun_elevation))
        ambient = bproc.types.Light()
        ambient.set_type("AREA")
        ambient.set_energy(random.uniform(*env["ambient_strength_range"]) * 100)
        ambient.set_location([0, 0, 50])
        ambient.set_color([0.8, 0.85, 1.0])
        scene_meta["lighting"] = {"type": "nishita_sun", "sun_elevation_deg": round(sun_elevation, 1),
                                  "sun_azimuth_deg": round(sun_azimuth, 1),
                                  "sun_strength": round(sun_strength, 2)}

    @staticmethod
    def _sun_color_from_elevation(elev: float) -> list:
        if elev > 45:  return [1.0, 0.95, 0.85]
        elif elev > 20: return [1.0, 0.80, 0.60]
        else:           return [1.0, 0.60, 0.35]

    # ── [FIX 1] Terrain: real relief + PBR ground ────────────────────────────
    def _build_terrain(self, scene_meta: dict, view: dict):
        env = self.env_config
        fw, fh = view["footprint_m"]
        side = 2.0 * math.hypot(fw, fh) + 20.0        # covers the frame incl. tilt
        segs = 160

        me = bpy.data.meshes.new("terrain")
        bm = bmesh.new()
        bm.loops.layers.uv.new("UVMap")
        bmesh.ops.create_grid(bm, x_segments=segs, y_segments=segs, size=side / 2, calc_uvs=True)
        bm.to_mesh(me)
        bm.free()
        ground = bpy.data.objects.new("terrain_ground", me)
        bpy.context.scene.collection.objects.link(ground)

        # Low-frequency relief (dunes / undulation), baked into the mesh so the
        # placement ray casts and the render see the same surface
        amp = random.uniform(*env["relief_m"])
        wavelength = random.uniform(*env["relief_wavelength_m"])
        if amp > 0.005:
            tex = bpy.data.textures.new("relief", type="CLOUDS")
            tex.noise_scale = wavelength
            tex.noise_depth = 2
            mod = ground.modifiers.new("relief", "DISPLACE")
            mod.texture = tex
            mod.texture_coords = "LOCAL"
            mod.strength = amp
            mod.mid_level = 0.5
            dg = bpy.context.evaluated_depsgraph_get()
            baked = bpy.data.meshes.new_from_object(ground.evaluated_get(dg), depsgraph=dg)
            ground.modifiers.clear()
            ground.data = baked
        for p in ground.data.polygons:
            p.use_smooth = True

        mat, mat_meta = (self._ground_material_pbr() if self._ground_assets
                         else self._ground_material_procedural())
        ground.data.materials.append(mat)
        scene_meta["terrain"] = {"side_m": round(side, 1), "relief_amp_m": round(amp, 3),
                                 "relief_wavelength_m": round(wavelength, 1), **mat_meta}
        return ground

    def _pbr_layer(self, nt, coord_out, asset: dict, tile_m: float, rot: float):
        """Image-texture set for one ground asset -> (color, roughness, normal-colour) sockets."""
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (1 / tile_m, 1 / tile_m, 1 / tile_m)
        mapping.inputs["Rotation"].default_value = (0, 0, rot)
        nt.links.new(coord_out, mapping.inputs["Vector"])

        def img(path, non_color):
            node = nt.nodes.new("ShaderNodeTexImage")
            node.image = bpy.data.images.load(str(path), check_existing=True)
            if non_color:
                node.image.colorspace_settings.name = "Non-Color"
            nt.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
            return node.outputs["Color"]

        color = img(asset["color"], False)
        if "AmbientOcclusion" in asset["maps"]:
            ao = img(asset["maps"]["AmbientOcclusion"], True)
            m = nt.nodes.new("ShaderNodeMix"); m.data_type = "RGBA"; m.blend_type = "MULTIPLY"
            m.inputs[0].default_value = 0.8
            nt.links.new(color, m.inputs[6]); nt.links.new(ao, m.inputs[7])
            color = m.outputs[2]
        rough = img(asset["maps"]["Roughness"], True) if "Roughness" in asset["maps"] else None
        normal = img(asset["maps"]["NormalGL"], True) if "NormalGL" in asset["maps"] else None
        return color, rough, normal

    def _ground_material_pbr(self):
        """Blend two ambientCG ground scans with a large-scale noise mask so the
        tiling repeat is broken up, then add macro brightness/hue variation."""
        mat = bpy.data.materials.new("terrain_mat")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        coord = nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]   # metres in object space

        a, b = random.sample(self._ground_assets, 2) if len(self._ground_assets) > 1 \
            else (self._ground_assets[0], self._ground_assets[0])
        tiles = [x["size_m"] or random.uniform(1.2, 3.0) for x in (a, b)]
        la = self._pbr_layer(nt, coord, a, tiles[0], random.uniform(0, 2 * math.pi))
        lb = self._pbr_layer(nt, coord, b, tiles[1], random.uniform(0, 2 * math.pi))

        mask_n = nt.nodes.new("ShaderNodeTexNoise")
        mask_n.inputs["Scale"].default_value = 1 / random.uniform(8, 35)
        mask_n.inputs["Detail"].default_value = 4
        nt.links.new(coord, mask_n.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        c = random.uniform(0.4, 0.6)
        ramp.color_ramp.elements[0].position = c - random.uniform(0.04, 0.12)
        ramp.color_ramp.elements[1].position = c + random.uniform(0.04, 0.12)
        nt.links.new(mask_n.outputs["Fac"], ramp.inputs["Fac"])
        fac = ramp.outputs["Color"]

        color = _mix(nt, "RGBA", fac, la[0], lb[0])
        # Macro variation: slow brightness / saturation drift over tens of metres
        macro = nt.nodes.new("ShaderNodeTexNoise")
        macro.inputs["Scale"].default_value = 1 / random.uniform(40, 120)
        nt.links.new(coord, macro.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = random.uniform(0.78, 0.92)
        mr.inputs["To Max"].default_value = random.uniform(1.05, 1.2)
        nt.links.new(macro.outputs["Fac"], mr.inputs["Value"])
        hsv = nt.nodes.new("ShaderNodeHueSaturation")
        hsv.inputs["Hue"].default_value = 0.5 + random.uniform(-0.02, 0.02)
        hsv.inputs["Saturation"].default_value = random.uniform(0.85, 1.15)
        nt.links.new(mr.outputs["Result"], hsv.inputs["Value"])
        nt.links.new(color, hsv.inputs["Color"])
        nt.links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])

        if la[1] is not None and lb[1] is not None:
            nt.links.new(_mix(nt, "RGBA", fac, la[1], lb[1]), bsdf.inputs["Roughness"])
        else:
            bsdf.inputs["Roughness"].default_value = random.uniform(0.75, 0.95)
        if la[2] is not None and lb[2] is not None:
            nmap = nt.nodes.new("ShaderNodeNormalMap")
            nmap.inputs["Strength"].default_value = random.uniform(0.6, 1.2)
            nt.links.new(_mix(nt, "RGBA", fac, la[2], lb[2]), nmap.inputs["Color"])
            nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
        spec = "Specular IOR Level" if "Specular IOR Level" in bsdf.inputs else "Specular"
        bsdf.inputs[spec].default_value = random.uniform(0.1, 0.35)
        return mat, {"ground": "pbr", "ground_assets": [a["id"], b["id"]],
                     "tile_m": [round(t, 2) for t in tiles]}

    def _ground_material_procedural(self):
        """Fallback when no textures are downloaded: multi-tone noise + pebbles + bump.
        Worse than scanned PBR, but never a single flat colour."""
        cr = self.env_config["ground_color_range"]
        c1 = [random.uniform(*cr[k]) for k in "rgb"]
        c2 = [max(0.0, v * random.uniform(0.55, 0.8)) for v in c1]
        c3 = [min(1.0, v * random.uniform(1.05, 1.25)) for v in c1]
        mat = bpy.data.materials.new("terrain_proc")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        coord = nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]

        fine = nt.nodes.new("ShaderNodeTexNoise")
        fine.inputs["Scale"].default_value = random.uniform(1.5, 6)
        fine.inputs["Detail"].default_value = 8
        nt.links.new(coord, fine.inputs["Vector"])
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.inputs["Scale"].default_value = random.uniform(4, 15)
        nt.links.new(coord, vor.inputs["Vector"])
        broad = nt.nodes.new("ShaderNodeTexNoise")
        broad.inputs["Scale"].default_value = 1 / random.uniform(6, 25)
        nt.links.new(coord, broad.inputs["Vector"])

        ramp = nt.nodes.new("ShaderNodeValToRGB")
        # Mixed noise values cluster around 0.5; stretch the ramp so the three
        # tones actually show up as patches instead of averaging to one colour
        ramp.color_ramp.elements[0].position = random.uniform(0.3, 0.4)
        ramp.color_ramp.elements[1].position = random.uniform(0.6, 0.7)
        ramp.color_ramp.elements[0].color = (*c2, 1)
        ramp.color_ramp.elements[1].color = (*c3, 1)
        mid = ramp.color_ramp.elements.new(0.5)
        mid.color = (*c1, 1)
        comb = _mix(nt, "FLOAT", 0.35, fine.outputs["Fac"], vor.outputs["Distance"])
        comb = _mix(nt, "FLOAT", 0.5, comb, broad.outputs["Fac"])
        nt.links.new(comb, ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = random.uniform(0.2, 0.6)
        bump.inputs["Distance"].default_value = 0.02
        nt.links.new(fine.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        bsdf.inputs["Roughness"].default_value = random.uniform(0.75, 0.95)
        return mat, {"ground": "procedural_fallback"}

    # ── Placement helpers ────────────────────────────────────────────────────
    @staticmethod
    def _raycast_terrain_z(terrain_obj, x: float, y: float) -> float:
        inv = terrain_obj.matrix_world.inverted()
        origin = inv @ mathutils.Vector((x, y, 500.0))
        direction = (inv.to_3x3() @ mathutils.Vector((0, 0, -1))).normalized()
        hit, location, _, _ = terrain_obj.ray_cast(origin, direction)
        return (terrain_obj.matrix_world @ location).z if hit else 0.0

    def _ground(self, ob, terrain, embed_frac=(0.0, 0.08)):
        """Drop an already rotated/scaled object so its lowest point touches the
        terrain under its centre, slightly embedded (real dumped items settle)."""
        bpy.context.view_layer.update()
        x0, x1, y0, y1, z0, z1 = _world_bbox(ob)
        tz = self._raycast_terrain_z(terrain, (x0 + x1) / 2, (y0 + y1) / 2)
        ob.location.z += (tz - z0) - (z1 - z0) * random.uniform(*embed_frac)

    def _scene_layout(self, view: dict) -> dict:
        """Cluster centres / line for this scene, sized to the camera footprint."""
        fw, fh = view["footprint_m"]
        R = 0.5 * min(fw, fh) * 0.92          # inscribed radius: in frame for any yaw
        pattern = self.env_config["scatter_pattern"]
        lay = {"R": R, "pattern": pattern}
        if pattern == "clustered":
            n = random.randint(1, 3)
            lay["centres"] = [self._in_disk(0.55 * R) for _ in range(n)]
            lay["spread"] = max(0.3, min(4.0, 0.18 * R))
        elif pattern == "linear":
            ang = random.uniform(0, math.pi)
            lay["origin"] = self._in_disk(0.3 * R)
            lay["dir"] = (math.cos(ang), math.sin(ang))
            lay["width"] = max(0.25, 0.08 * R)
        return lay

    @staticmethod
    def _in_disk(r):
        a, d = random.uniform(0, 2 * math.pi), r * math.sqrt(random.random())
        return d * math.cos(a), d * math.sin(a)

    def _scatter_position(self, lay: dict) -> tuple:
        R = lay["R"]
        if lay["pattern"] == "clustered":
            cx, cy = random.choice(lay["centres"])
            x, y = cx + random.gauss(0, lay["spread"]), cy + random.gauss(0, lay["spread"])
        elif lay["pattern"] == "linear":
            t, s = random.uniform(-R, R), random.gauss(0, lay["width"])
            (ox, oy), (dx, dy) = lay["origin"], lay["dir"]
            x, y = ox + t * dx - s * dy, oy + t * dy + s * dx
        else:
            x, y = self._in_disk(0.95 * R)
        d = math.hypot(x, y)
        if d > R:                               # keep the centre inside the frame
            x, y = x * R / d, y * R / d
        return x, y

    # ── E-waste objects ──────────────────────────────────────────────────────
    def _load_ewaste_objects(self, scene_meta: dict, picks: list, terrain, view: dict,
                             lay: dict) -> list:
        loaded_objects, placed = [], []          # placed: (x, y, radius)
        for i, (cls_name, model) in enumerate(picks):
            try:
                loaded = load_model_unique(str(model["path"]))
            except Exception as e:
                scene_meta.setdefault("load_errors", []).append(f"{cls_name}: {str(e)[:80]}")
                continue
            if not loaded:
                continue
            obj = self._merge_loaded_model(loaded)
            if obj is None:
                scene_meta.setdefault("load_errors", []).append(
                    f"{cls_name}: no mesh in {model['path'].name}")
                continue
            obj.set_name(f"ewaste_{cls_name}_{i}")
            obj.set_cp("category_id", EWASTE_CLASSES.index(cls_name))

            bb = obj.get_bound_box()
            dims = sorted((bb.max(axis=0) - bb.min(axis=0)).tolist(), reverse=True)
            extent = float(dims[0])
            if (cls_name in CLASS_SIZE_BY_SHAPE and not model["has_sidecar"]
                    and not model.get("sized_by_shape")):
                a = dims[0] / max(dims[1], 1e-9)
                b = dims[1] / max(dims[2], 1e-9)
                for rule, size_m, jitter in CLASS_SIZE_BY_SHAPE[cls_name]:
                    if rule(a, b):
                        # Correct the index entry so later eligibility checks use it
                        model.update(size_m=size_m, jitter=jitter, sized_by_shape=True)
                        break
            target = model["size_m"] * random.uniform(*model["jitter"])
            # Re-check resolvability with the true size (only known after loading):
            # a model too small to label here must not appear unlabelled
            if (target * model["gate_frac"] / view["gsd"] < SPAWN_MIN_PX
                    or target > SPAWN_MAX_FRAC * min(view["footprint_m"])):
                obj.delete()
                self.stats["gate_rejected_after_load"][cls_name] += 1
                continue
            s = target / extent if extent > 1e-6 else 1.0
            radius = target / 2

            # Rejection-sample a spot that doesn't interpenetrate earlier objects
            for _ in range(15):
                x, y = self._scatter_position(lay)
                if all(math.hypot(x - px, y - py) > 0.8 * (radius + pr) for px, py, pr in placed):
                    break
            else:
                # Frame is full (common at low altitude): drop the object rather
                # than let two meshes interpenetrate
                obj.delete()
                self.stats["no_free_spot"][cls_name] += 1
                continue
            placed.append((x, y, radius))

            obj.set_rotation_euler([random.gauss(0, 0.15), random.gauss(0, 0.15),
                                    random.uniform(0, 2 * math.pi)])
            obj.set_scale([s] * 3)
            obj.set_location([x, y, 0.0])
            self._ground(obj.blender_obj, terrain)
            self._apply_weathering(obj.blender_obj, cls_name)

            loaded_objects.append((obj, cls_name))
            scene_meta.setdefault("objects", []).append({
                "class": cls_name, "model": model["path"].name,
                "size_m": round(target, 4), "expected_px": round(target / view["gsd"], 1),
                "position": [round(x, 2), round(y, 2)],
            })
        return loaded_objects, placed

    @staticmethod
    def _merge_loaded_model(loaded: list):
        """glTF imports arrive as EMPTY hierarchies: bake transforms, drop empties, join."""
        meshes = [o.blender_obj for o in loaded if o.blender_obj.type == "MESH"]
        empties = [o.blender_obj for o in loaded if o.blender_obj.type != "MESH"]
        if not meshes:
            for e in empties:
                bpy.data.objects.remove(e, do_unlink=True)
            return None
        for m in meshes:
            world = m.matrix_world.copy()
            m.parent = None
            m.matrix_world = world
        for e in empties:
            bpy.data.objects.remove(e, do_unlink=True)
        view_layer = bpy.context.view_layer
        view_layer.update()
        for o in view_layer.objects:
            if o is not None:
                o.select_set(False)
        for m in meshes:
            m.select_set(True)
        view_layer.objects.active = meshes[0]
        if len(meshes) > 1:
            bpy.ops.object.join()
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        meshes[0].select_set(False)
        return bproc.types.MeshObject(meshes[0])

    def _apply_weathering(self, blender_obj, cls_name: str):
        w = MATERIAL_WEATHERING
        is_metal = cls_name in ["circuit_board", "metal_component", "hard_drive",
                                "power_supply", "cpu_chip"]
        for slot in blender_obj.material_slots:
            if not slot.material:
                continue
            mat = slot.material
            if not mat.use_nodes:
                mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if not bsdf:
                continue
            bsdf.inputs["Roughness"].default_value = random.uniform(*w["roughness_range"])
            base = list(bsdf.inputs["Base Color"].default_value)
            if is_metal and random.random() < w["rust_probability"]:
                f = random.uniform(0.1, 0.6)
                rust = [0.6, 0.2, 0.05]
                bsdf.inputs["Base Color"].default_value = [base[k] * (1 - f) + rust[k] * f
                                                           for k in range(3)] + [1.0]
                bsdf.inputs["Metallic"].default_value = random.uniform(0.3, 0.7)
            elif not is_metal and random.random() < w["fade_probability"]:
                f = random.uniform(0.05, 0.35)
                bsdf.inputs["Base Color"].default_value = [min(1.0, c + f) for c in base[:3]] + [1.0]
            if random.random() < w["burn_probability"]:
                f = random.uniform(0.2, 0.8)
                base = list(bsdf.inputs["Base Color"].default_value)
                bsdf.inputs["Base Color"].default_value = [c * (1 - f) * 0.3 for c in base[:3]] + [1.0]

    # ── [FIX 1] Non-e-waste clutter (rendered, never labelled) ───────────────
    @staticmethod
    def _simple_mat(name, color, roughness=0.8, metallic=0.0, hue_var=0.04, noise_scale=None,
                    bump=0.0):
        """Principled material; per-instance hue/value variation via Object Info Random."""
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        info = nt.nodes.new("ShaderNodeObjectInfo")
        hsv = nt.nodes.new("ShaderNodeHueSaturation")
        hsv.inputs["Color"].default_value = (*color, 1)
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = 0.5 - hue_var
        mr.inputs["To Max"].default_value = 0.5 + hue_var
        nt.links.new(info.outputs["Random"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], hsv.inputs["Hue"])
        mv = nt.nodes.new("ShaderNodeMapRange")
        mv.inputs["To Min"].default_value = 0.75
        mv.inputs["To Max"].default_value = 1.2
        nt.links.new(info.outputs["Random"], mv.inputs["Value"])
        nt.links.new(mv.outputs["Result"], hsv.inputs["Value"])
        out = hsv.outputs["Color"]
        if noise_scale:                                       # surface mottling (rock, bark)
            nz = nt.nodes.new("ShaderNodeTexNoise")
            nz.inputs["Scale"].default_value = noise_scale
            dark = _mix(nt, "RGBA", nz.outputs["Fac"],
                        tuple(c * 0.6 for c in color) + (1,), tuple(min(1, c * 1.2) for c in color) + (1,))
            out = _mix(nt, "RGBA", 0.5, out, dark)
        nt.links.new(out, bsdf.inputs["Base Color"])
        if bump > 0:                                          # fine surface relief
            bn = nt.nodes.new("ShaderNodeTexNoise")
            bn.inputs["Scale"].default_value = 30
            bn.inputs["Detail"].default_value = 6
            b = nt.nodes.new("ShaderNodeBump")
            b.inputs["Strength"].default_value = bump
            nt.links.new(bn.outputs["Fac"], b.inputs["Height"])
            nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])
        return mat

    @staticmethod
    def _mesh_obj(name, bm, material):
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        ob.data.materials.append(material)
        return ob

    def _displaced(self, ob, strength, scale):
        tex = bpy.data.textures.new("dn", type="CLOUDS")
        tex.noise_scale = scale
        mod = ob.modifiers.new("d", "DISPLACE")
        mod.texture = tex
        mod.strength = strength
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), depsgraph=dg)
        ob.modifiers.clear()
        ob.data = me
        return ob

    def _proto(self, kind: str):
        """One prototype mesh for a clutter type, unit-ish size (longest side ~1)."""
        env = self.env_config
        if kind == "rock":
            bm = bmesh.new()
            bmesh.ops.create_icosphere(bm, subdivisions=4, radius=0.5)
            col = random.choice([(0.34, 0.31, 0.27), (0.42, 0.35, 0.27), (0.27, 0.26, 0.25),
                                 (0.47, 0.41, 0.32), (0.32, 0.27, 0.21), (0.52, 0.45, 0.36)])
            ob = self._mesh_obj("rock", bm, self._simple_mat("rock_m", col, 0.92, noise_scale=8,
                                                             bump=0.35))
            self._displaced(ob, random.uniform(0.15, 0.35), random.uniform(0.2, 0.45))
            for poly in ob.data.polygons:
                poly.use_smooth = True
            ob.scale = (1, random.uniform(0.6, 1.0), random.uniform(0.35, 0.7))
            return ob, (1, 1, 1)
        if kind == "shrub":
            bm = bmesh.new()
            for _ in range(random.randint(5, 12)):
                r = random.uniform(0.15, 0.3)
                g = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=r)
                d = random.uniform(0, 0.3); a = random.uniform(0, 2 * math.pi)
                bmesh.ops.translate(bm, verts=g["verts"],
                                    vec=(d * math.cos(a), d * math.sin(a), r * 0.6))
            col = random.choice(env["shrub_palette"])
            ob = self._mesh_obj("shrub", bm, self._simple_mat("shrub_m", col, 0.95, hue_var=0.03,
                                                              noise_scale=25))
            self._displaced(ob, 0.08, 0.1)
            for poly in ob.data.polygons:
                poly.use_smooth = True
            return ob, (1, 1, 1)
        if kind in ("stick", "log"):
            bm = bmesh.new()
            r = random.uniform(0.012, 0.03) if kind == "stick" else random.uniform(0.06, 0.12)
            bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=r, radius2=r * 0.7, depth=1.0)
            bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0),
                             matrix=mathutils.Matrix.Rotation(math.pi / 2, 3, "Y"))
            col = random.choice([(0.35, 0.25, 0.15), (0.45, 0.35, 0.22), (0.55, 0.5, 0.42)])
            return self._mesh_obj(kind, bm, self._simple_mat(kind + "_m", col, 0.9, noise_scale=30)), (1, 1, 1)
        if kind == "tyre":
            bm = bmesh.new()
            R, r = 0.4, 0.11
            rings = []
            for i in range(28):
                a = 2 * math.pi * i / 28
                ring = [bm.verts.new(((R + r * math.cos(b)) * math.cos(a), (R + r * math.cos(b)) * math.sin(a),
                                      r * math.sin(b)))
                        for b in (2 * math.pi * j / 10 for j in range(10))]
                rings.append(ring)
            for i in range(28):
                for j in range(10):
                    bm.faces.new((rings[i][j], rings[(i + 1) % 28][j],
                                  rings[(i + 1) % 28][(j + 1) % 10], rings[i][(j + 1) % 10]))
            ob = self._mesh_obj("tyre", bm, self._simple_mat("tyre_m", (0.03, 0.03, 0.03), 0.85, hue_var=0))
            ob.scale = (1, 1, 1.3)
            return ob, (1, 1, 1)
        if kind in ("box", "block"):
            bm = bmesh.new()
            bmesh.ops.create_cube(bm, size=1.0)
            if kind == "box":
                dims = (1.0, random.uniform(0.6, 1.0), random.choice([random.uniform(0.4, 0.8), 0.04]))
                col = random.choice([(0.55, 0.4, 0.25), (0.62, 0.5, 0.35), (0.5, 0.45, 0.4)])
            else:
                dims = (1.0, 0.5, 0.5)
                col = random.choice([(0.55, 0.55, 0.52), (0.6, 0.25, 0.15), (0.7, 0.68, 0.62)])
            bmesh.ops.scale(bm, vec=dims, verts=bm.verts)
            return self._mesh_obj(kind, bm, self._simple_mat(kind + "_m", col, 0.9, noise_scale=12)), (1, 1, 1)
        if kind == "bag":
            bm = bmesh.new()
            bmesh.ops.create_grid(bm, x_segments=16, y_segments=16, size=0.5)
            col = random.choice([(0.9, 0.9, 0.9), (0.05, 0.05, 0.05), (0.15, 0.3, 0.6),
                                 (0.8, 0.75, 0.2), (0.3, 0.55, 0.3)])
            ob = self._mesh_obj("bag", bm, self._simple_mat("bag_m", col, random.uniform(0.3, 0.6)))
            self._displaced(ob, random.uniform(0.1, 0.25), random.uniform(0.12, 0.3))
            mod = ob.modifiers.new("s", "SOLIDIFY"); mod.thickness = 0.01
            return ob, (1, 1, 1)
        if kind == "bottle":
            bm = bmesh.new()
            can = random.random() < 0.35
            r, h = (0.033, 0.12) if can else (0.035, 0.26)
            bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=r, radius2=r, depth=h)
            bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0),
                             matrix=mathutils.Matrix.Rotation(math.pi / 2, 3, "Y"))
            if can:
                mat = self._simple_mat("can_m", random.choice([(0.7, 0.1, 0.1), (0.1, 0.3, 0.7),
                                                                (0.8, 0.8, 0.8)]), 0.3, metallic=0.9)
            else:
                mat = self._simple_mat("bottle_m", random.choice([(0.6, 0.75, 0.8), (0.2, 0.5, 0.25),
                                                                   (0.85, 0.85, 0.9)]), 0.1)
            ob = self._mesh_obj("bottle", bm, mat)
            return ob, (1, 1, 1)
        if kind == "rope":           # hose / rope: the hard negative for cable classes
            cu = bpy.data.curves.new("rope", type="CURVE")
            cu.dimensions = "3D"
            cu.bevel_depth = random.uniform(0.008, 0.012)
            cu.bevel_resolution = 2
            sp = cu.splines.new("BEZIER")
            n = random.randint(6, 12)
            sp.bezier_points.add(n - 1)
            x = y = 0.0; ang = random.uniform(0, 6.3)
            for bp in sp.bezier_points:
                ang += random.gauss(0, 0.8); x += 0.1 * math.cos(ang); y += 0.1 * math.sin(ang)
                bp.co = (x, y, 0.01)
                bp.handle_left_type = bp.handle_right_type = "AUTO"
            ob = bpy.data.objects.new("rope", cu)
            bpy.context.scene.collection.objects.link(ob)
            cu.materials.append(self._simple_mat("rope_m", random.choice(
                [(0.1, 0.45, 0.1), (0.85, 0.55, 0.1), (0.75, 0.7, 0.55), (0.15, 0.2, 0.6)]), 0.7))
            dg = bpy.context.evaluated_depsgraph_get()
            me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), depsgraph=dg)
            new = bpy.data.objects.new("rope", me)
            bpy.context.scene.collection.objects.link(new)
            bpy.data.objects.remove(ob, do_unlink=True)
            return new, (1, 1, 1)
        raise ValueError(kind)

    DISTRACTOR_SIZE_M = {   # longest side range (metres) of each clutter type
        "rock": (0.08, 0.9), "shrub": (0.4, 1.6), "stick": (0.3, 1.4), "log": (1.0, 3.0),
        "tyre": (0.55, 0.8), "box": (0.25, 0.7), "block": (0.39, 0.45), "bag": (0.3, 0.8),
        "bottle": (0.12, 0.33), "rope": (0.8, 3.0),
    }

    def _add_distractors(self, scene_meta: dict, terrain, view: dict, ewaste: list,
                         negative: bool):
        fw, fh = view["footprint_m"]
        Rd = 0.55 * math.hypot(fw, fh)                 # frame + tilt margin
        area = math.pi * Rd * Rd
        counts = Counter()
        total = 0
        density_boost = random.uniform(0.4, 1.6) * (1.5 if negative else 1.0)
        for kind, dens in self.env_config["distractors"].items():
            lo, hi = self.DISTRACTOR_SIZE_M[kind]
            if hi / view["gsd"] < DISTRACTOR_MIN_PX:
                continue
            n = int(dens * area / 100 * density_boost)
            n = min(n + (1 if random.random() < dens else 0), MAX_DISTRACTORS_PER_TYPE,
                    MAX_DISTRACTORS_TOTAL - total)
            if n <= 0:
                continue
            # A few unique prototypes per type; every instance is a linked duplicate
            # (shares mesh + material, so hundreds of instances stay cheap). The
            # prototypes' base scale and size are recorded once, before any reuse.
            protos = []
            for _ in range(min(3, n)):
                pob, _ = self._proto(kind)
                bpy.context.view_layer.update()
                protos.append((pob, tuple(pob.scale), max(pob.dimensions) or 1.0))
            for k in range(n):
                proto, base_scale, base_longest = random.choice(protos)
                ob = proto.copy()
                bpy.context.scene.collection.objects.link(ob)
                size = random.uniform(max(lo, DISTRACTOR_MIN_PX * view["gsd"]), hi)
                s = size / base_longest
                ob.scale = tuple(v * s for v in base_scale)
                x, y = self._in_disk(Rd)
                # Mostly keep clear of e-waste, but allow occasional occlusion
                if any(math.hypot(x - ex, y - ey) < er + size / 2 for ex, ey, er in ewaste) \
                        and random.random() < 0.85:
                    x, y = self._in_disk(Rd)
                flat = kind in ("stick", "log", "rope", "bag", "bottle", "box", "block")
                tilt = 0.08 if flat else 0.25
                if kind == "tyre" and random.random() < 0.15:
                    tilt = 1.3                           # tyre leaning / on its side
                ob.rotation_euler = (random.gauss(0, tilt), random.gauss(0, tilt),
                                     random.uniform(0, 2 * math.pi))
                ob.location = (x, y, 0)
                ob.name = f"distractor_{kind}_{k}"
                self._ground(ob, terrain, embed_frac=(0.0, 0.25 if kind == "rock" else 0.08))
                counts[kind] += 1
                total += 1
            for proto, _, _ in protos:              # copies keep the shared mesh data
                bpy.data.objects.remove(proto, do_unlink=True)

        # Optional Objaverse clutter (tools/fetch_distractors.py)
        if self._asset_distractors:
            n_assets = random.randint(0, max(1, int(area / 400)))
            for j in range(min(n_assets, 30)):
                d = random.choice(self._asset_distractors)
                if d["size_m"] / view["gsd"] < DISTRACTOR_MIN_PX:
                    continue
                try:
                    obj = self._merge_loaded_model(load_model_unique(str(d["path"])))
                except Exception:
                    continue
                if obj is None:
                    continue
                bb = obj.get_bound_box()
                ext = float((bb.max(axis=0) - bb.min(axis=0)).max()) or 1.0
                obj.set_scale([d["size_m"] * random.uniform(0.8, 1.2) / ext] * 3)
                obj.set_rotation_euler([random.gauss(0, 0.2), random.gauss(0, 0.2), random.uniform(0, 6.3)])
                x, y = self._in_disk(Rd)
                obj.set_location([x, y, 0])
                obj.set_name(f"distractor_asset_{d['category']}_{j}")
                self._ground(obj.blender_obj, terrain)
                counts["asset:" + d["category"]] += 1
        scene_meta["distractors"] = dict(counts)
        return counts

    # ── Camera ────────────────────────────────────────────────────────────────
    def _setup_camera(self, view: dict, scene_meta: dict) -> Optional[list]:
        """Adds one camera pose per frame. -> list of (dx, dy) offsets, or None if a sequence would
        see past the terrain (caller skips the scene)."""
        W, H = view["resolution"]
        bproc.camera.set_resolution(W, H)
        offsets = [(0.0, 0.0)]
        if self.sequence_frames > 1:
            fw, fh = view["footprint_m"]
            seq = es.sequence_offsets_m(self.sequence_frames, fh, fw, view["yaw_deg"],
                                        views=self.sequence_views, max_object_m=self.sequence_max_object_m,
                                        terrain_half_side_m=scene_meta["terrain"]["side_m"] / 2 - 1.0)
            if not seq["fits"]:
                return None
            offsets = seq["offsets"]
            scene_meta["sequence"] = {"n_frames": len(offsets), "spacing_m": round(seq["spacing_m"], 3),
                                      "overlap_frac": round(seq["overlap_frac"], 3),
                                      "views": self.sequence_views,
                                      "max_object_m": self.sequence_max_object_m,
                                      "offsets_m": [[round(x, 3), round(y, 3)] for x, y in offsets]}
        for dx, dy in offsets:
            bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(
                [dx, dy, view["altitude_m"]],
                [math.radians(view["tilt_deg"][0]), math.radians(view["tilt_deg"][1]),
                 math.radians(view["yaw_deg"])]))                # (0,0,yaw) = straight down
        cam = bpy.context.scene.camera.data
        cam.sensor_fit = "HORIZONTAL"
        cam.sensor_width = self.cam["sensor_width_mm"]
        bproc.camera.set_intrinsics_from_blender_params(
            lens=self.cam["focal_mm"], lens_unit="MILLIMETERS",
            clip_start=0.1, clip_end=view["altitude_m"] * 4 + 50)
        scene_meta.update({
            "altitude_m": round(view["altitude_m"], 2),
            "resolution": [W, H],
            "capture_mode": view["capture_mode"],
            "gsd_cm_per_px": round(view["gsd"] * 100, 3),
            "footprint_m": [round(v, 2) for v in view["footprint_m"]],
            "camera": {"profile": self.camera_name,
                       "sensor_width_mm": self.cam["sensor_width_mm"], "focal_mm": self.cam["focal_mm"],
                       "hfov_deg": round(math.degrees(hfov_rad(self.cam)), 2),
                       "tilt_deg": [round(v, 2) for v in view["tilt_deg"]],
                       "yaw_deg": round(view["yaw_deg"], 1)},
            "post_effects": "haze/motion-blur/ISO noise applied in 03_augmentation_pipeline.py",
        })
        return offsets

    # ── [FIX 2] Labels from segmentation ─────────────────────────────────────
    def _extract_annotations_from_segmask(self, seg_map, loaded_objects, resolution,
                                          scene_meta, name_to_instance) -> list:
        W, H = resolution
        anns = []
        present = set(np.unique(seg_map).tolist())
        for obj, cls_name in loaded_objects:
            iid = name_to_instance.get(obj.get_name())
            rec = {"class": cls_name, "visible": iid is not None and iid in present}
            scene_meta.setdefault("visibility", []).append(rec)
            if not rec["visible"]:
                continue
            ys, xs = np.nonzero(seg_map == iid)
            vis_px = int(len(xs))
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            bw, bh = x1 - x0 + 1, y1 - y0 + 1                  # inclusive pixel extent
            rec.update(visible_px=vis_px, bbox_px=[bw, bh])
            if vis_px < LABEL_MIN_VISIBLE_PX or max(bw, bh) < LABEL_MIN_SIDE_PX:
                rec["labelled"] = False
                self.stats["visible_unlabelled"][cls_name] += 1
                continue
            rec["labelled"] = True
            anns.append((EWASTE_CLASSES.index(cls_name),
                         (x0 + bw / 2) / W, (y0 + bh / 2) / H, bw / W, bh / H))
            self.stats["labels"][cls_name] += 1
            _LABEL_COUNTS[cls_name] += 1
        return anns

    # ── Main scene generator ─────────────────────────────────────────────────
    def generate_scene(self, scene_id: int, altitude_profile: str = "all") -> Optional[dict]:
        bproc.clean_up()
        # clean_up() doesn't reset frame_start/frame_end (v2 fix: scene N rendered N frames)
        bproc.utility.reset_keyframes()
        scene_meta = {"scene_id": scene_id, "environment": self.environment}
        try:
            if altitude_profile == "all":
                chosen = random.choices(list(ALTITUDE_WEIGHTS), weights=list(ALTITUDE_WEIGHTS.values()))[0]
            else:
                chosen = altitude_profile
            scene_meta["altitude_profile"] = chosen

            # 1. View first: altitude, resolution and GSD decide which models can appear
            view = self._plan_view(chosen)
            negative = random.random() < self.negative_fraction
            picks = []
            if not negative:
                n_objects = random.randint(*ALTITUDE_PROFILES[chosen][2])
                picks = self._sample_classes_for_scene(n_objects, view)
                if not picks:
                    scene_meta["skip_reason"] = (f"no class resolvable at {view['altitude_m']:.0f} m, "
                                                 f"{view['gsd']*100:.2f} cm/px")
                    return scene_meta
                scene_meta["eligible_classes"] = sorted(self._eligible_cache)
            scene_meta["negative"] = negative

            # 2. World
            self._randomize_lighting(scene_meta)
            terrain = self._build_terrain(scene_meta, view)
            lay = self._scene_layout(view)
            loaded_objects, placed = self._load_ewaste_objects(scene_meta, picks, terrain, view, lay)
            if picks and not loaded_objects:
                scene_meta["skip_reason"] = "no_objects_loaded"
                return scene_meta
            self._add_distractors(scene_meta, terrain, view, placed, negative)
            offsets = self._setup_camera(view, scene_meta)
            if offsets is None:
                scene_meta["skip_reason"] = (f"sequence of {self.sequence_frames} frames would leave the "
                                             f"terrain at {view['altitude_m']:.0f} m")
                return scene_meta

            if self.dry_run:
                scene_meta["dry_run"] = True
                scene_meta["yolo_annotations"] = len(loaded_objects)
                print(f"    [DRY RUN] {scene_id}: {len(loaded_objects)} objs, {chosen}, "
                      f"{view['resolution']}, {view['gsd']*100:.2f} cm/px")
                return scene_meta

            global _SEGMENTATION_ENABLED
            if not _SEGMENTATION_ENABLED:
                bproc.renderer.enable_segmentation_output(map_by=["instance", "name"])
                _SEGMENTATION_ENABLED = True
            # enable_segmentation_output() only indexes meshes existing at call time (v2 fix)
            for index, mesh_obj in enumerate(o for o in bpy.context.scene.objects if o.type == "MESH"):
                mesh_obj.pass_index = index + 1

            data = bproc.renderer.render()
            import PIL.Image
            n_frames = len(offsets)
            frames, any_labels = [], False
            for k in range(n_frames):
                seg_map = data["instance_segmaps"][k]
                img_name = (f"synthetic_{self.environment}_{scene_id:06d}" if n_frames == 1 else
                            f"synthetic_{self.environment}_{scene_id:06d}_f{k:02d}")
                img_path = self.img_dir / f"{img_name}.jpg"
                PIL.Image.fromarray(data["colors"][k]).save(str(img_path), quality=92)
                name_to_instance = {m["name"]: m["idx"] for m in data["instance_attribute_maps"][k]}
                anns = self._extract_annotations_from_segmask(
                    seg_map, loaded_objects, view["resolution"], scene_meta, name_to_instance)
                any_labels |= bool(anns)
                frames.append((k, img_name, img_path, anns))

            # single frame: as before, a positive scene without labels is dropped.  Sequence: kept if ANY
            # frame has labels (frames at the ends may legitimately show none -- that is what 11 sees).
            if not any_labels and not negative:
                vis = scene_meta.get("visibility", [])
                scene_meta["skip_reason"] = (
                    f"no_labelled_objects ({sum(v['visible'] for v in vis)}/{len(vis)} visible, "
                    f"{view['altitude_m']:.0f}m)")
                for _, _, img_path, _ in frames:
                    img_path.unlink(missing_ok=True)
                return scene_meta

            for k, img_name, img_path, anns in frames:
                lbl_path = self.lbl_dir / f"{img_name}.txt"
                meta_path = self.meta_dir / f"{img_name}.json"
                with open(lbl_path, "w") as f:                       # empty file for negatives
                    for a in anns:
                        f.write(f"{a[0]} {a[1]:.6f} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f}\n")
                fmeta = {**scene_meta, "yolo_annotations": len(anns), "image_path": str(img_path),
                         "label_path": str(lbl_path), "frame_index": k,
                         "camera_offset_m": [round(offsets[k][0], 3), round(offsets[k][1], 3)]}
                with open(meta_path, "w") as f:
                    json.dump(fmeta, f, indent=2, default=str)
                with open(self.index_path, "a") as f:
                    f.write(json.dumps({
                        "scene_id": scene_id, "environment": self.environment,
                        "altitude_profile": chosen, "altitude_m": scene_meta["altitude_m"],
                        "gsd_cm_per_px": scene_meta["gsd_cm_per_px"], "resolution": view["resolution"],
                        "capture_mode": view["capture_mode"],
                        "camera_profile": self.camera_name, "focal_mm": self.cam["focal_mm"],
                        "sensor_width_mm": self.cam["sensor_width_mm"],
                        "n_annotations": len(anns), "negative": negative,
                        "classes_present": sorted({a[0] for a in anns}),
                        "n_distractors": sum(scene_meta.get("distractors", {}).values()),
                        "sequence_id": scene_id if n_frames > 1 else None,
                        "frame_index": k, "n_frames": n_frames,
                        "camera_offset_m": fmeta["camera_offset_m"],
                        "image_path": str(img_path), "label_path": str(lbl_path),
                    }, default=str) + "\n")
            scene_meta.update(yolo_annotations=sum(len(a) for *_, a in frames), n_frames=n_frames)
            self.stats["scenes"]["negative" if negative else "positive"] += 1
            return scene_meta

        except Exception as e:
            import traceback
            scene_meta["error"] = str(e)
            scene_meta["traceback"] = traceback.format_exc()
            return scene_meta


_LOAD_COUNTER = [0]


def load_model_unique(path: str):
    """
    bproc.loader.load_obj() with every newly created image given a unique name
    (v2 fix: generic embedded glTF texture names made Cycles reuse cached pixels).
    """
    before = set(bpy.data.images)
    loaded = bproc.loader.load_obj(path)
    _LOAD_COUNTER[0] += 1
    for img in set(bpy.data.images) - before:
        img.name = f"m{_LOAD_COUNTER[0]:06d}_{img.name}"
    return loaded


# ─── Batch orchestrator ────────────────────────────────────────────────────────
_BPROC_INITIALIZED = False
_SEGMENTATION_ENABLED = False


def run_batch_generation(models_dir: Path, textures_dir: Path, hdri_dir: Path,
                         output_dir: Path, n_scenes: int, environment: str,
                         altitude_mode: str, start_id: int = 0, dry_run: bool = False,
                         res_scale: float = 1.0, samples: int = 128,
                         distractors_dir: Optional[Path] = None, negative_fraction: float = 0.06,
                         balance: bool = True, sequence_frames: int = 1, sequence_views: int = 3,
                         sequence_max_object_m: float = 1.0, camera=DEFAULT_CAMERA):
    global _BPROC_INITIALIZED
    if not _BPROC_INITIALIZED:
        bproc.init()
        bpy.context.scene.render.use_persistent_data = False
        bproc.renderer.set_max_amount_of_samples(samples)
        bproc.renderer.set_noise_threshold(0.05)
        _BPROC_INITIALIZED = True

    builder = EWasteSceneBuilder(models_dir=models_dir, textures_dir=textures_dir,
                                 hdri_dir=hdri_dir, output_dir=output_dir,
                                 environment=environment, dry_run=dry_run, res_scale=res_scale,
                                 distractors_dir=distractors_dir,
                                 negative_fraction=negative_fraction,
                                 sequence_frames=sequence_frames, sequence_views=sequence_views,
                                 sequence_max_object_m=sequence_max_object_m, camera=camera)
    if not balance:
        builder.balance_alpha = 0.0
    ok = fail = skip = 0
    print(f"\n{'='*64}\n  E-Waste Synthetic Generation v3\n  Environment : {environment}"
          f"\n  Altitude    : {altitude_mode}\n  Scenes      : {n_scenes}\n  Output      : {output_dir}"
          f"\n  Ground      : {len(builder._ground_assets)} ambientCG sets"
          f"{'' if builder._ground_assets else ' (procedural fallback!)'}"
          f"\n  Res scale   : {res_scale}   Negatives: {negative_fraction:.0%}"
          f"\n  Camera      : {builder.camera_name} (focal {builder.cam['focal_mm']} mm, sensor "
          f"{builder.cam['sensor_width_mm']} mm)"
          + ("   <-- ABLATION: not the real camera" if builder.camera_name != DEFAULT_CAMERA else "")
          + f"\n{'='*64}\n")
    if not builder._has_models():
        raise SystemExit(f"No 3D models in {models_dir}")

    for i in range(n_scenes):
        scene_id = start_id + i
        stem = f"synthetic_{environment}_{scene_id:06d}" + ("_f00" if sequence_frames > 1 else "")
        if (output_dir / "images" / f"{stem}.jpg").exists():
            skip += 1
            continue
        meta = builder.generate_scene(scene_id, altitude_profile=altitude_mode)
        if meta and "error" not in meta and "skip_reason" not in meta:
            ok += 1
            print(f"  [{scene_id:06d}] ✓ {meta.get('yolo_annotations', 0):2d} labels "
                  f"{meta.get('altitude_m', 0):5.1f}m {meta.get('gsd_cm_per_px', 0):5.2f}cm/px "
                  f"clutter={sum(meta.get('distractors', {}).values()):3d}"
                  f"{'  NEGATIVE' if meta.get('negative') else ''}")
        elif meta and "skip_reason" in meta:
            skip += 1
            print(f"  [{scene_id:06d}] ○ skipped: {meta['skip_reason']}")
            with open(output_dir / "skipped_scenes.jsonl", "a") as f:
                f.write(json.dumps(meta, default=str) + "\n")
        else:
            fail += 1
            print(f"  [{scene_id:06d}] ✗ {str((meta or {}).get('error', 'unknown'))[:90]}")
            with open(output_dir / "failed_scenes.jsonl", "a") as f:
                f.write(json.dumps(meta, default=str) + "\n")

    labels, unl = builder.stats["labels"], builder.stats["visible_unlabelled"]
    tgt = builder._target_shares()
    tot_all = sum(_LABEL_COUNTS.values()) or 1
    print(f"\n{'='*64}\n  Done: {ok} rendered | {skip} skipped | {fail} failed"
          f"   balance={'on' if builder.balance_alpha > 0 else 'OFF'}")
    print(f"  {'class':<18}{'this run':>9}{'unlabelled':>11}{'all-time share':>16}{'target':>8}")
    for c in EWASTE_CLASSES:
        print(f"  {c:<18}{labels[c]:>9}{unl[c]:>11}{_LABEL_COUNTS[c] / tot_all:>15.1%}{tgt[c]:>9.1%}")
    stats = {"rendered": ok, "skipped": skip, "failed": fail,
             "scenes": dict(builder.stats["scenes"]),
             "labels_per_class": {c: labels[c] for c in EWASTE_CLASSES},
             "visible_unlabelled_per_class": {c: unl[c] for c in EWASTE_CLASSES},
             "all_time_label_share": {c: round(_LABEL_COUNTS[c] / tot_all, 4) for c in EWASTE_CLASSES},
             "target_share": {c: round(tgt[c], 4) for c in EWASTE_CLASSES},
             "balance": builder.balance_alpha > 0,
             "gate_rejected_after_load": dict(builder.stats["gate_rejected_after_load"]),
             "no_free_spot": dict(builder.stats["no_free_spot"]),
             "rules": {"SPAWN_MIN_PX": SPAWN_MIN_PX, "MIN_FEATURE_PX": MIN_FEATURE_PX,
                       "LABEL_MIN_VISIBLE_PX": LABEL_MIN_VISIBLE_PX,
                       "LABEL_MIN_SIDE_PX": LABEL_MIN_SIDE_PX, "res_scale": res_scale}}
    (output_dir / f"generation_stats_{start_id:06d}.json").write_text(json.dumps(stats, indent=2))
    print(f"  Stats: {output_dir / f'generation_stats_{start_id:06d}.json'}\n{'='*64}\n")
    return ok


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BlenderProc E-Waste Synthetic Generator v3")
    p.add_argument("--models_dir",    default="./assets/models")
    p.add_argument("--textures_dir",  default="./assets/textures")
    p.add_argument("--hdri_dir",      default="./assets/haven")
    p.add_argument("--distractors_dir", default="./assets/distractors",
                   help="optional Objaverse clutter from tools/fetch_distractors.py")
    p.add_argument("--output_dir",    default="./data/raw/synthetic")
    p.add_argument("--n_scenes",      type=int, default=5000)
    p.add_argument("--environment",   default="all", choices=list(ENVIRONMENT_CONFIGS) + ["all"])
    p.add_argument("--altitude_mode", default="all", choices=list(ALTITUDE_PROFILES) + ["all"])
    p.add_argument("--start_id",      type=int, default=0)
    p.add_argument("--negative_fraction", type=float, default=0.06,
                   help="fraction of scenes with clutter but NO e-waste (empty label file)")
    p.add_argument("--res_scale",     type=float, default=1.0,
                   help="scale render resolution (quick tests only; resolvability rules follow it)")
    p.add_argument("--samples",       type=int, default=128)
    p.add_argument("--seed",          type=int, default=None)
    p.add_argument("--no_balance",    action="store_true",
                   help="fixed class weights only (ablation baseline for the adaptive balance)")
    p.add_argument("--dry_run",       action="store_true")
    p.add_argument("--sequence_frames", type=int, default=1,
                   help=">1: render N overlapping frames per scene along-track (synthetic validation "
                        "SEQUENCES for 11_evidence_pipeline.py); files <stem>_fNN.jpg")
    p.add_argument("--sequence_views", type=int, default=3,
                   help="spacing = (along-track footprint - max object) / views, as 11's scheduler")
    p.add_argument("--sequence_max_object_m", type=float, default=1.0)
    p.add_argument("--camera_override", default=None, choices=[k for k in CAMERA_PROFILES if k != DEFAULT_CAMERA],
                   help="ABLATION ONLY: render with a different camera profile (e.g. p4p_mismatch_50mm_equiv)")
    p.add_argument("--override_focal_mm", type=float, default=None,
                   help="ABLATION ONLY: custom focal length (mm) on the real sensor/resolutions")
    p.add_argument("--override_sensor_width_mm", type=float, default=None,
                   help="ABLATION ONLY: custom sensor width (mm)")
    a = p.parse_args()
    camera = a.camera_override or DEFAULT_CAMERA
    if a.override_focal_mm or a.override_sensor_width_mm:
        base = dict(CAMERA_PROFILES[camera])
        if a.override_focal_mm:
            base["focal_mm"] = a.override_focal_mm
        if a.override_sensor_width_mm:
            base["sensor_width_mm"] = a.override_sensor_width_mm
        base["name"] = f"custom(focal={base['focal_mm']},sensor_w={base['sensor_width_mm']})"
        camera = base
    if a.seed is not None:
        random.seed(a.seed)
        np.random.seed(a.seed)

    common = dict(models_dir=Path(a.models_dir), textures_dir=Path(a.textures_dir),
                  hdri_dir=Path(a.hdri_dir), altitude_mode=a.altitude_mode, dry_run=a.dry_run,
                  res_scale=a.res_scale, samples=a.samples,
                  distractors_dir=Path(a.distractors_dir), negative_fraction=a.negative_fraction,
                  balance=not a.no_balance, sequence_frames=a.sequence_frames,
                  sequence_views=a.sequence_views, sequence_max_object_m=a.sequence_max_object_m,
                  camera=camera)
    if a.environment == "all":
        envs  = list(ENVIRONMENT_CONFIGS)
        fracs = [ENVIRONMENT_CONFIGS[e]["scene_fraction"] for e in envs]
        n_each = [int(a.n_scenes * f / sum(fracs)) for f in fracs]
        n_each[0] += a.n_scenes - sum(n_each)
        offset = a.start_id
        for env, n in zip(envs, n_each):
            run_batch_generation(output_dir=Path(a.output_dir) / env, n_scenes=n,
                                 environment=env, start_id=offset, **common)
            offset += n
    else:
        run_batch_generation(output_dir=Path(a.output_dir), n_scenes=a.n_scenes,
                             environment=a.environment, start_id=a.start_id, **common)
