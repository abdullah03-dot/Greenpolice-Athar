"""
E-Waste Detection Pipeline — Phase 3: Albumentations Augmentation Pipeline
===========================================================================
Applies forensic-grade augmentations to the hybrid dataset (synthetic + real).
Designed specifically for aerial e-waste imagery.

Design decisions documented here:
  • Two augmentation policies: SYNTHETIC (aggressive) and REAL (conservative)
    Synthetic images need heavier augmentation to close the sim-to-real gap.
    Real images need lighter touch — preserve forensic detail.

  • Augmentation is applied AFTER tiling and AdaIN style transfer.
    Correct order: BlenderProc → 03b tiling → AdaIN (on tiles) → Albumentations → Training
    (updated 2026-10-02: this script used to LongestMaxSize every frame to 640 px, which
    turns a 24 px phone into 3 px.  It now expects native-resolution TILES from
    03b_tile_dataset.py and refuses to downscale unless you pass --allow_downscale.)

  • CLAHE applied early (before noise) to maximize contrast benefit on
    low-visibility materials (plastic casings, dark circuit boards)

  • MotionBlur parameters tuned to DJI Phantom 4 Pro vibration profile
    (rotor frequency ~50Hz, typical shutter 1/500s → kernel 3-7px at 1080p)

  • Hard-negative augmentation: 10% of synthetic images become label-free
    backgrounds (all augmentations applied, YOLO label cleared) to
    train the model to NOT detect in absence of e-waste.

Usage:
  conda activate ewads
  pip install albumentations opencv-python-headless pillow pyyaml tqdm

  # 1) cut native-resolution tiles (see 03b_tile_dataset.py), 2) optional AdaIN on tiles, then:
  python 03_augmentation_pipeline.py \\
    --input_dir ./data/tiled \\
    --output_dir ./data/augmented \\
    --n_aug_per_image 3 \\
    --rfs_t 0.1                 # repeat-factor sampling for rare classes (0 = off)

  # Single image test (inspect output before full run):
  python 03_augmentation_pipeline.py --test_image ./data/raw/real/images/001.jpg

  # Validate YOLO labels after augmentation:
  python 03_augmentation_pipeline.py --validate_labels --output_dir ./data/augmented
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ewads_tiling as et

try:
    import albumentations as A
except ImportError:
    print("[ERROR] albumentations not found. Install with: pip install 'albumentations>=2.0,<3'")
    sys.exit(1)
# The policies below use the Albumentations 2.x API (std_range, fill, *_range, compression_type).
# The old 1.x argument names (var_limit, value, quality_lower, ImageCompression.ImageType...) were
# removed: under 2.x they are silently IGNORED with a warning (augmentation strengths change
# without any error) and `ImageType` crashes outright.  environment.yml says ">=1.4.0", which
# resolves to 2.x on a fresh install, so this is the version you actually have.
if int(A.__version__.split(".")[0]) < 2:
    print(f"[ERROR] albumentations {A.__version__} found; this script needs >=2.0. "
          f"pip install -U 'albumentations>=2.0,<3'")
    sys.exit(1)

# ─── Class taxonomy (must match BlenderProc script) ───────────────────────────
EWASTE_CLASSES = [
    "crt_monitor", "flat_screen", "circuit_board", "battery",
    "cable_bundle", "single_cable", "mobile_phone", "keyboard",
    "printer", "small_appliance", "large_appliance", "cpu_chip",
    "hard_drive", "power_supply", "ewaste_pile", "plastic_casing",
    "metal_component", "lighting",
]

# ─── Augmentation Policies ────────────────────────────────────────────────────
# Free rotation (A.Rotate) re-fits an axis-aligned box around the rotated object, which is
# label noise for mAP@0.5:0.95 (IoU thresholds up to 0.95).  Measured (2026-10-02, 3000 random
# shapes per row; IoU of the re-fitted box vs the true tight box of the rotated object):
#                      +-10 deg: largest_box / ellipse     +-45 deg: largest_box / ellipse
#   rect (boxes, PCs)       0.86 / 0.92                         0.65 / 0.78
#   thin (cables)           0.81 / 0.80  (p5 0.36 / 0.49)       0.58 / 0.60
#   blob (piles, bundles)   0.85 / 0.95                         0.64 / 0.86
# So rotation stays OFF: the renderer already gives every object a uniform 0-360 deg yaw and
# flips + 90-degree turns (on below) are exact.  If you enable it (--rotate_limit, e.g. for a
# small REAL set), the tighter 'ellipse' box method is used and the limit should stay <= 10.
ROTATE_LIMIT_DEFAULT = 0


def _rotate(limit: int, p: float):
    if limit <= 0:
        return []
    try:
        return [A.Rotate(limit=limit, p=p, border_mode=cv2.BORDER_CONSTANT, fill=114, rotate_method="ellipse")]
    except TypeError:                                    # older Albumentations: no rotate_method
        return [A.Rotate(limit=limit, p=p, border_mode=cv2.BORDER_CONSTANT, fill=114)]


def build_synthetic_policy(image_size: int = 768, rotate_limit: int = ROTATE_LIMIT_DEFAULT) -> A.Compose:
    """
    Heavy augmentation for synthetic images.
    Goal: close sim-to-real gap as aggressively as possible.

    Applied AFTER AdaIN style transfer — weather/noise on top of style-adapted images.
    Bbox-safe: all spatial transforms preserve YOLO bounding box coordinates.
    """
    return A.Compose([
        # ── Resize to training resolution ──────────────────────────────────
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=114),

        # ── Spatial transforms ─────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),   # Aerial imagery can be any orientation
        A.RandomRotate90(p=0.5),

        # Drone yaw variation: only if --rotate_limit > 0 (see note above)
        *_rotate(rotate_limit, 0.6),

        # Scale variation (altitude simulation beyond what BlenderProc provides)
        A.RandomScale(scale_limit=0.3, p=0.4),

        # Perspective warp (oblique angle simulation)
        A.Perspective(scale=(0.02, 0.08), p=0.3),

        # ── Photometric — applied early, before noise ──────────────────────
        # CLAHE: critical for low-contrast materials (plastic casings, dark PCBs)
        A.CLAHE(clip_limit=(1.5, 4.0), tile_grid_size=(8, 8), p=0.7),

        # Color/brightness variation
        A.RandomBrightnessContrast(brightness_limit=0.3,
                                   contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=15,
                             sat_shift_limit=30,
                             val_shift_limit=20, p=0.7),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                   b_shift_limit=15, p=0.4),

        # ── Weather simulation ─────────────────────────────────────────────
        # RandomFog: simulates desert haze, morning mist, humidity
        A.RandomFog(fog_coef_range=(0.05, 0.35), alpha_coef=0.08, p=0.25),

        # RandomRain: simulates rain streaks through camera
        A.RandomRain(slant_range=(-10, 10),
                     drop_length=15, drop_width=1,
                     drop_color=(180, 180, 180),
                     blur_value=3, brightness_coefficient=0.85,
                     rain_type="drizzle", p=0.15),

        # RandomSunFlare: lens flare from direct sun hits (common in desert)
        A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 0.5),
                         angle_range=(0, 1),
                         num_flare_circles_range=(3, 6),
                         src_radius=200, src_color=(255, 200, 150),
                         p=0.10),

        # RandomShadow: tree/building shadows cast on dump site
        A.RandomShadow(shadow_roi=(0, 0.4, 1, 1),
                       num_shadows_limit=(1, 3),
                       shadow_dimension=5, p=0.35),

        # ── Sensor noise simulation ────────────────────────────────────────
        # ISONoise: simulates high-ISO camera noise
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),

        # GaussNoise as fallback noise
        # old var_limit (10, 60) -> std 3.2..7.7 grey levels -> /255
        A.GaussNoise(std_range=(0.0124, 0.0304), p=0.3),

        # MotionBlur: simulates DJI Phantom 4 Pro rotor vibration
        # Kernel size 3-7px at 1080p ≈ real vibration blur profile
        A.MotionBlur(blur_limit=(3, 7), p=0.35),

        # ── Image quality degradation ──────────────────────────────────────
        # JPEG compression artifacts (telemetry-compressed footage)
        A.ImageCompression(quality_range=(60, 95), compression_type="jpeg", p=0.4),

        # Downscale + upscale (simulates lower-res drone footage upsampled)
        A.Downscale(scale_range=(0.7, 0.95), p=0.2),

        # (Cutout/occlusion is applied AFTER this policy by safe_cutout(): Albumentations'
        #  CoarseDropout is not label-aware and could erase a 12 px object whose label stays.)

    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.25,    # Drop bbox if < 25% visible after transform
        min_area=0.0001,
    ))


def build_real_policy(image_size: int = 768, rotate_limit: int = ROTATE_LIMIT_DEFAULT) -> A.Compose:
    """
    Conservative augmentation for real drone captures.
    Goal: increase sample diversity without distorting forensic detail.
    No weather simulation (weather already present in real captures).
    CLAHE applied — real drone images often have insufficient contrast.
    """
    return A.Compose([
        # Resize
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=114),

        # Spatial
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        *_rotate(min(rotate_limit, 15), 0.4),

        # Photometric — gentle
        A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                   contrast_limit=0.15, p=0.6),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15,
                             val_shift_limit=10, p=0.5),

        # Light noise
        A.GaussNoise(std_range=(0.0088, 0.0196), p=0.3),     # old var_limit (5, 25)
        A.MotionBlur(blur_limit=(3, 5), p=0.2),

        # Shadow (vegetation occlusion is common in real captures)
        A.RandomShadow(shadow_roi=(0, 0.3, 1, 1),
                       num_shadows_limit=(1, 2),
                       shadow_dimension=4, p=0.3),

        # JPEG compression
        A.ImageCompression(quality_range=(75, 98), compression_type="jpeg", p=0.3),

    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.30,    # Higher threshold for real images
        min_area=0.0001,
    ))


def build_validation_policy(image_size: int = 768) -> A.Compose:
    """
    No augmentation — resize only. Applied to validation and test sets.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=114),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
    ))


# ─── YOLO Label I/O ───────────────────────────────────────────────────────────
def read_yolo_label(label_path: Path) -> tuple:
    """
    Read YOLO format label file.
    Returns (class_ids: list[int], bboxes: list[[cx,cy,w,h]])
    """
    if not label_path.exists():
        return [], []

    class_ids, bboxes = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(float(parts[0]))
                cx, cy, w, h = (float(x) for x in parts[1:5])
                # 6-decimal rounding can leave a box edge at -5e-7 / 1.0000005, which
                # Albumentations 2.x rejects.  Re-fit the box inside [0, 1].
                x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
                x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                class_ids.append(cls_id)
                bboxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])

    return class_ids, bboxes


def write_yolo_label(label_path: Path, class_ids: list, bboxes: list):
    """Write YOLO format label file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for cls_id, bbox in zip(class_ids, bboxes):
            cx, cy, w, h = bbox
            # int(): Albumentations 2.x hands class labels back as floats ("7.0"), which breaks
            # strict YOLO readers.
            f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def validate_bbox(bbox: list) -> bool:
    """Sanity check on a YOLO bbox."""
    cx, cy, w, h = bbox
    return (0 < cx < 1 and 0 < cy < 1 and
            0 < w <= 1 and 0 < h <= 1 and
            w * h > 1e-6)


# ─── Parallel workers (CPU-bound: CLAHE/fog/noise on 768 px tiles ~0.3 s each) ───
_AUG = None


def _init_worker(kwargs: dict, seed: int):
    global _AUG
    import os
    random.seed(seed + os.getpid())
    np.random.seed((seed + os.getpid()) % (2**32 - 1))
    _AUG = EWasteAugmentor(**kwargs)


def _run_task(t: tuple):
    img_path, lbl_path, out_img, out_lbl, source_type, aug_i, is_neg = t
    return _AUG.augment_image(Path(img_path), Path(lbl_path), Path(out_img), Path(out_lbl),
                              source_type=source_type, aug_index=aug_i, is_hard_negative=is_neg), is_neg


# ─── Label-aware occlusion ────────────────────────────────────────────────────
def safe_cutout(img: np.ndarray, bboxes, p: float = 0.25, max_holes: int = 8,
                size: tuple = (8, 32), margin: int = 3) -> np.ndarray:
    """Grey-out random squares (debris/vegetation occlusion) that do NOT touch any labelled box."""
    if random.random() >= p:
        return img
    h, w = img.shape[:2]
    boxes = [((cx - bw / 2) * w - margin, (cy - bh / 2) * h - margin,
              (cx + bw / 2) * w + margin, (cy + bh / 2) * h + margin) for cx, cy, bw, bh in bboxes]
    out = img.copy()
    for _ in range(random.randint(1, max_holes)):
        s_ = random.randint(*size)
        x1, y1 = random.randint(0, max(0, w - s_)), random.randint(0, max(0, h - s_))
        x2, y2 = x1 + s_, y1 + s_
        if any(not (x2 <= b[0] or x1 >= b[2] or y2 <= b[1] or y1 >= b[3]) for b in boxes):
            continue
        out[y1:y2, x1:x2] = 114
    return out


# ─── Core Augmentor ───────────────────────────────────────────────────────────
class EWasteAugmentor:

    def __init__(self, image_size: int = 768, hard_negative_rate: float = 0.0,
                 rotate_limit: int = ROTATE_LIMIT_DEFAULT, allow_downscale: bool = False,
                 workers: int = 1, seed: int = 2026):
        self.workers, self.seed, self.rotate_limit = workers, seed, rotate_limit
        self.image_size = image_size
        self.allow_downscale = allow_downscale
        # WARNING (fixed 2026-09-28): a "hard negative" here clears the labels of
        # an image that still CONTAINS e-waste, which teaches the detector that
        # e-waste is background — the same failure as unlabelled small objects.
        # True negatives now come from 02_blenderproc_synthetic_v2.py
        # (--negative_fraction: clutter-only scenes with empty label files) and
        # from real footage with no dumping. Only raise this for images you KNOW
        # contain no e-waste.
        if hard_negative_rate > 0:
            print(f"  ⚠  hard_negative_rate={hard_negative_rate}: labels will be CLEARED on images "
                  f"that may contain e-waste. Use 0 unless the inputs are e-waste-free.")
        self.hard_negative_rate = hard_negative_rate

        self.synthetic_policy  = build_synthetic_policy(image_size, rotate_limit)
        self.real_policy       = build_real_policy(image_size, rotate_limit)
        self.validation_policy = build_validation_policy(image_size)

    def augment_image(self, img_path: Path, lbl_path: Path,
                      out_img_dir: Path, out_lbl_dir: Path,
                      source_type: str,
                      aug_index: int = 0,
                      is_hard_negative: bool = False) -> Optional[str]:
        """
        Apply augmentation to one image+label pair.
        Returns output image filename on success, None on failure.

        Args:
            source_type: "synthetic" | "real" | "val"
            aug_index: 0 for original (resize only for val), 1+ for augmented copies
            is_hard_negative: clear all labels (used to create background negatives)
        """
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read labels
        class_ids, bboxes = read_yolo_label(lbl_path)

        # Hard negative: clear labels
        if is_hard_negative:
            class_ids, bboxes = [], []

        # Select policy
        if source_type == "val":
            policy = self.validation_policy
        elif source_type == "synthetic":
            policy = self.synthetic_policy
        else:
            policy = self.real_policy

        # Apply augmentation
        try:
            result = policy(
                image=img,
                bboxes=bboxes if bboxes else [],
                class_labels=class_ids if class_ids else [],
            )
        except Exception as e:
            if not getattr(self, "_warned", False):      # do not swallow errors silently
                print(f"  [WARN] augmentation failed on {img_path.name}: {type(e).__name__}: {e}")
                self._warned = True
            return None

        aug_img    = result["image"]
        aug_bboxes = result["bboxes"]
        aug_cls    = result["class_labels"]
        if source_type == "synthetic":
            aug_img = safe_cutout(aug_img, aug_bboxes)

        # Validate output bboxes
        valid = [(cls, bbox) for cls, bbox in zip(aug_cls, aug_bboxes)
                 if validate_bbox(list(bbox))]
        if valid:
            aug_cls_f, aug_bboxes_f = zip(*valid)
        else:
            aug_cls_f, aug_bboxes_f = [], []

        # Build output filename
        stem    = img_path.stem
        out_name = f"{stem}_aug{aug_index:02d}"
        if is_hard_negative:
            out_name += "_neg"

        out_img_path = out_img_dir / f"{out_name}.jpg"
        out_lbl_path = out_lbl_dir / f"{out_name}.txt"

        # Save image
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        save_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_img_path), save_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Save label (even if empty — hard negatives need empty label files)
        write_yolo_label(out_lbl_path, list(aug_cls_f), list(aug_bboxes_f))

        return out_name

    def process_split(self, input_dir: Path, output_dir: Path,
                      source_type: str, n_aug: int = 3) -> dict:
        """
        Process all images in a split directory.
        Returns statistics dict.
        """
        img_dir = input_dir / "images"
        lbl_dir = input_dir / "labels"
        out_img = output_dir / "images"
        out_lbl = output_dir / "labels"

        if not img_dir.exists():
            print(f"  [WARN] No images directory at {img_dir}")
            return {}

        image_files = (list(img_dir.glob("*.jpg")) +
                       list(img_dir.glob("*.jpeg")) +
                       list(img_dir.glob("*.png")))

        # Refuse to silently destroy resolution: images larger than image_size would be shrunk
        # by LongestMaxSize.  Tiles from 03b are exactly image_size, so this passes for them.
        biggest = 0
        from PIL import Image as _PIL
        for p_ in image_files[:200]:
            try:
                with _PIL.open(p_) as im_:
                    biggest = max(biggest, max(im_.size))      # header only, no decode
            except Exception:
                pass
        if biggest > self.image_size and not self.allow_downscale:
            print(f"\n  [ERROR] {input_dir} contains images up to {biggest}px but image_size is "
                  f"{self.image_size}.\n  They would be DOWNSCALED, erasing small objects (a 24 px phone "
                  f"-> {24 * self.image_size // biggest} px).\n  Run 03b_tile_dataset.py first and point "
                  f"--input_dir at its output, or pass --allow_downscale to override.")
            sys.exit(2)

        print(f"\n  Processing {len(image_files)} images from {input_dir.name}")
        print(f"  Source type   : {source_type}")
        print(f"  Augmentations : {n_aug} per image")
        print(f"  Hard-neg rate : {self.hard_negative_rate:.0%}")

        stats = {
            "source": str(input_dir),
            "source_type": source_type,
            "n_source_images": len(image_files),
            "n_augmented": 0,
            "n_hard_negatives": 0,
            "n_failed": 0,
        }

        # Build the task list (hard-negative choice is made here, in the parent, for determinism)
        tasks = []
        for img_path in image_files:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if source_type == "val":
                tasks.append((str(img_path), str(lbl_path), str(out_img), str(out_lbl), "val", 0, False))
                continue
            make_neg = random.random() < self.hard_negative_rate
            for aug_i in range(n_aug):
                tasks.append((str(img_path), str(lbl_path), str(out_img), str(out_lbl),
                              source_type, aug_i, make_neg and aug_i == 0))

        def tally(result, is_neg):
            if result:
                stats["n_hard_negatives" if is_neg else "n_augmented"] += 1
            else:
                stats["n_failed"] += 1

        if self.workers > 1 and len(tasks) > 8:
            from concurrent.futures import ProcessPoolExecutor
            kw = dict(image_size=self.image_size, hard_negative_rate=self.hard_negative_rate,
                      rotate_limit=self.rotate_limit, allow_downscale=self.allow_downscale)
            with ProcessPoolExecutor(self.workers, initializer=_init_worker,
                                     initargs=(kw, self.seed)) as ex:
                for result, is_neg in tqdm(ex.map(_run_task, tasks, chunksize=16),
                                           total=len(tasks), desc=f"  {source_type}"):
                    tally(result, is_neg)
        else:
            for t in tqdm(tasks, desc=f"  {source_type}"):
                img_p, lbl_p, oi, ol, st_, ai, neg = t
                tally(self.augment_image(Path(img_p), Path(lbl_p), Path(oi), Path(ol),
                                         source_type=st_, aug_index=ai, is_hard_negative=neg), neg)

        return stats


# ─── Dataset Validation ───────────────────────────────────────────────────────
def validate_label_files(data_dir: Path) -> dict:
    """
    Sanity check all YOLO label files in a directory.
    Reports: missing labels, empty labels, invalid bboxes, class distribution.
    """
    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"

    image_files = (list(img_dir.glob("*.jpg")) +
                   list(img_dir.glob("*.jpeg")) +
                   list(img_dir.glob("*.png")))

    report = {
        "total_images": len(image_files),
        "missing_labels": [],
        "empty_labels": [],
        "invalid_bboxes": [],
        "class_distribution": {cls: 0 for cls in EWASTE_CLASSES},
        "class_ids_seen": set(),
    }

    for img_path in image_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        if not lbl_path.exists():
            report["missing_labels"].append(img_path.name)
            continue

        class_ids, bboxes = read_yolo_label(lbl_path)

        if not class_ids:
            report["empty_labels"].append(img_path.name)
            continue

        for cls_id, bbox in zip(class_ids, bboxes):
            if 0 <= cls_id < len(EWASTE_CLASSES):
                report["class_distribution"][EWASTE_CLASSES[cls_id]] += 1
                report["class_ids_seen"].add(cls_id)
            if not validate_bbox(bbox):
                report["invalid_bboxes"].append(
                    f"{img_path.name}: class={cls_id} bbox={bbox}"
                )

    report["class_ids_seen"] = sorted(report["class_ids_seen"])

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Label Validation: {data_dir}")
    print(f"{'='*60}")
    print(f"  Total images     : {report['total_images']}")
    print(f"  Missing labels   : {len(report['missing_labels'])}")
    print(f"  Empty labels     : {len(report['empty_labels'])}")
    print(f"  Invalid bboxes   : {len(report['invalid_bboxes'])}")
    print(f"\n  Class distribution:")
    for cls, count in sorted(report["class_distribution"].items(),
                              key=lambda x: -x[1]):
        bar = "█" * min(40, count // max(1, max(report["class_distribution"].values()) // 40))
        print(f"    {cls:<20} {count:6d}  {bar}")

    missing_classes = [cls for cls, count in report["class_distribution"].items()
                       if count == 0]
    if missing_classes:
        print(f"\n  ⚠  Missing classes (0 samples): {', '.join(missing_classes)}")

    return report


# ─── Mosaic Augmentation ──────────────────────────────────────────────────────
def apply_mosaic(image_paths: list, label_paths: list,
                 out_img: Path, out_lbl: Path,
                 image_size: int = 640) -> int:
    """
    Mosaic augmentation: combine 4 images into one 2x2 grid.

    WARNING (2026-10-02): this HALVES the resolution of every source image (each is resized
    into a quadrant), so a 12 px object becomes 6 px.  It is also redundant: Ultralytics'
    train-time mosaic (05 uses mosaic=0.8) composes the 768 px tiles at native scale (checked
    with Ultralytics 8.4.171: object-size distribution unchanged).  --add_mosaic is therefore
    refused unless you pass --allow_lossy_mosaic.
    Reduces memory usage vs large single images and exposes model to
    4x the object variety per training step.

    Returns number of mosaic images created.
    """
    assert len(image_paths) == len(label_paths)
    if len(image_paths) < 4:
        return 0

    n_mosaics = 0
    indices = list(range(len(image_paths)))
    random.shuffle(indices)

    # Group into quads
    for i in range(0, len(indices) - 3, 4):
        quad_idx = indices[i:i+4]
        half = image_size // 2

        # Build mosaic canvas
        mosaic = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_cls    = []

        positions = [
            (0, 0, half, half),       # top-left
            (half, 0, image_size, half),    # top-right
            (0, half, half, image_size),    # bottom-left
            (half, half, image_size, image_size),  # bottom-right
        ]

        valid = True
        for j, idx in enumerate(quad_idx):
            img = cv2.imread(str(image_paths[idx]))
            if img is None:
                valid = False
                break

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x1, y1, x2, y2 = positions[j]
            w, h = x2 - x1, y2 - y1

            resized = cv2.resize(img, (w, h))
            mosaic[y1:y2, x1:x2] = resized

            # Remap bboxes into mosaic coordinates
            cls_ids, bboxes = read_yolo_label(label_paths[idx])
            for cls_id, bbox in zip(cls_ids, bboxes):
                cx_abs = bbox[0] * w + x1
                cy_abs = bbox[1] * h + y1
                bw_abs = bbox[2] * w
                bh_abs = bbox[3] * h

                cx_n = cx_abs / image_size
                cy_n = cy_abs / image_size
                bw_n = bw_abs / image_size
                bh_n = bh_abs / image_size

                new_bbox = [cx_n, cy_n, bw_n, bh_n]
                if validate_bbox(new_bbox):
                    mosaic_bboxes.append(new_bbox)
                    mosaic_cls.append(cls_id)

        if not valid or not mosaic_bboxes:
            continue

        # Save mosaic
        mosaic_name = f"mosaic_{n_mosaics:06d}"
        out_img_path = out_img / f"{mosaic_name}.jpg"
        out_lbl_path = out_lbl / f"{mosaic_name}.txt"

        save = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_img_path), save, [cv2.IMWRITE_JPEG_QUALITY, 92])
        write_yolo_label(out_lbl_path, mosaic_cls, mosaic_bboxes)
        n_mosaics += 1

    return n_mosaics


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="EWADS Albumentations augmentation pipeline"
    )
    parser.add_argument("--input_dir",          type=str,
                        default="./data/raw")
    parser.add_argument("--output_dir",         type=str,
                        default="./data/augmented")
    parser.add_argument("--image_size",         type=int, default=None,
                        help="tile side in px; default = tile size recorded by 03b "
                             "(tiling_manifest.json), else 768.  Must equal the 03b --tile.")
    parser.add_argument("--allow_downscale",    action="store_true",
                        help="permit shrinking inputs larger than image_size (destroys small objects)")
    parser.add_argument("--rotate_limit",       type=int, default=ROTATE_LIMIT_DEFAULT,
                        help="max free rotation in degrees (0 = off; loosens boxes, see the table at the top)")
    parser.add_argument("--allow_lossy_mosaic", action="store_true")
    parser.add_argument("--workers",            type=int, default=0,
                        help="parallel processes (0 = all cores - 1)")
    parser.add_argument("--seed",               type=int, default=2026)
    parser.add_argument("--rfs_t",              type=float, default=0.1,
                        help="repeat-factor sampling threshold t (0 = off). Images containing a class "
                             "present in < t of images are repeated sqrt(t/f) times")
    parser.add_argument("--rfs_cap",            type=float, default=8.0)
    parser.add_argument("--n_aug_per_image",    type=int, default=3,
                        help="Augmented copies per training image")
    parser.add_argument("--hard_negative_rate", type=float, default=0.0)
    parser.add_argument("--add_mosaic",         action="store_true",
                        help="Generate mosaic augmentations from training set")
    parser.add_argument("--n_mosaics",          type=int, default=500)
    parser.add_argument("--validate_labels",    action="store_true",
                        help="Validate label files in output_dir and exit")
    parser.add_argument("--test_image",         type=str,
                        help="Test augmentation on single image (display output)")
    args = parser.parse_args()

    # Validation-only mode
    if args.validate_labels:
        validate_label_files(Path(args.output_dir) / "train")
        return

    # Single image test mode
    if args.test_image:
        _test_single_image(args.test_image, args.image_size or 768)
        return

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiling = {}
    mf = input_dir / "tiling_manifest.json"
    if mf.exists():
        tiling = json.loads(mf.read_text())
    if args.image_size is None:
        args.image_size = int(tiling.get("tile", 768))
    elif tiling and args.image_size != int(tiling["tile"]):
        print(f"  [ERROR] --image_size {args.image_size} != tile size {tiling['tile']} used by 03b. "
              f"Mismatch would resample tiles.")
        sys.exit(2)
    if not tiling:
        print("  [WARN] no tiling_manifest.json in input_dir: inputs are assumed to be tiles/crops "
              "already <= image_size (the downscale guard will stop you if they are not).")

    augmentor = EWasteAugmentor(
        image_size=args.image_size,
        hard_negative_rate=args.hard_negative_rate,
        rotate_limit=args.rotate_limit,
        allow_downscale=args.allow_downscale,
        workers=args.workers or max(1, (__import__("os").cpu_count() or 2) - 1),
        seed=args.seed,
    )
    random.seed(args.seed)

    all_stats = []

    # Expected input structure:
    # data/raw/
    #   synthetic/ → images/, labels/
    #   real/      → images/, labels/
    #   val/       → images/, labels/  ← no augmentation
    #   test/      → images/, labels/  ← no augmentation, no touching

    source_configs = [
        ("synthetic", "synthetic", args.n_aug_per_image),
        ("real",      "real",      args.n_aug_per_image),
        ("val",       "val",       1),   # resize only
    ]

    train_sources = []                       # recorded in dataset.yaml; 05 --synthetic_only checks it
    for split_name, source_type, n_aug in source_configs:
        split_input  = input_dir / split_name
        split_output = output_dir / ("train" if split_name != "val" else "val")

        if not split_input.exists():
            print(f"  [SKIP] {split_input} not found")
            continue

        stats = augmentor.process_split(
            input_dir=split_input,
            output_dir=split_output,
            source_type=source_type,
            n_aug=n_aug,
        )
        all_stats.append(stats)
        if split_name != "val":
            train_sources.append(split_name)

    # Mosaic augmentation (training set only)
    if args.add_mosaic and not args.allow_lossy_mosaic:
        print("\n  [SKIP] --add_mosaic halves image resolution (12 px objects -> 6 px) and is redundant:"
              "\n         Ultralytics' train-time mosaic (05: mosaic=0.8) places 768 px tiles at NATIVE scale"
              "\n         (measured on tiled data: median object 99.5 px before vs 94.3 px after, 0% < 8 px)."
              "\n         Pass --allow_lossy_mosaic to force the offline version anyway.")
        args.add_mosaic = False
    if args.add_mosaic:
        train_img_dir = output_dir / "train" / "images"
        train_lbl_dir = output_dir / "train" / "labels"
        img_files = sorted(train_img_dir.glob("*.jpg"))
        lbl_files = [train_lbl_dir / (f.stem + ".txt") for f in img_files]

        print(f"\n  Generating {args.n_mosaics} mosaic augmentations...")
        n_mosaic = apply_mosaic(
            img_files[:args.n_mosaics * 4],
            lbl_files[:args.n_mosaics * 4],
            out_img=train_img_dir,
            out_lbl=train_lbl_dir,
            image_size=args.image_size,
        )
        print(f"  Created {n_mosaic} mosaic images")

    # Validate output
    validate_label_files(output_dir / "train")

    # Write stats
    stats_path = output_dir / "augmentation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\n  Stats written: {stats_path}")

    # Repeat-factor sampling over the final training images (rare-class tiles are listed more often)
    rfs_list = None
    if args.rfs_t > 0:
        rfs_list = write_rfs_list(output_dir, args.rfs_t, args.rfs_cap)

    # Write YOLO dataset YAML
    _write_dataset_yaml(output_dir, tile=args.image_size, overlap=int(tiling.get("overlap", 128)),
                        train_list=rfs_list, sources=train_sources)
    print(f"\n  Dataset YAML written: {output_dir / 'dataset.yaml'}")


def _test_single_image(img_path: str, image_size: int):
    """Quick visual test of augmentation pipeline on a single image."""
    import os

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {img_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    policy  = build_synthetic_policy(image_size or 768)

    print(f"Applying 9 augmentation samples to {img_path}")
    print(f"Output saved to ./aug_test/")
    Path("./aug_test").mkdir(exist_ok=True)

    for i in range(9):
        result = policy(image=img_rgb, bboxes=[], class_labels=[])
        out = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"./aug_test/aug_{i:02d}.jpg", out,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Done. Check ./aug_test/")


def write_rfs_list(output_dir: Path, t: float, cap: float) -> Optional[Path]:
    """LVIS repeat-factor sampling over train images -> train_rfs.txt (one path per line, repeats
    listed multiple times; Ultralytics accepts a .txt image list).  Finishes the class balancing
    the renderer cannot fully do (e.g. cpu_chip is only resolvable at low altitude)."""
    img_dir, lbl_dir = output_dir / "train" / "images", output_dir / "train" / "labels"
    per_img = {}
    for p in sorted(img_dir.glob("*.jpg")):
        ids, _ = read_yolo_label(lbl_dir / (p.stem + ".txt"))
        per_img[str(p.resolve())] = set(ids)
    if not per_img:
        return None
    r_i, r_c = et.repeat_factors(per_img, t=t, cap=cap)
    listing = et.stochastic_repeats(r_i, seed=2026)
    out = output_dir / "train_rfs.txt"
    out.write_text("\n".join(listing) + "\n")
    print(f"\n  Repeat-factor sampling (t={t}, cap={cap}): {len(per_img)} train images -> "
          f"{len(listing)} entries ({len(listing) / len(per_img):.2f}x)")
    for c, r in sorted(r_c.items(), key=lambda kv: -kv[1])[:8]:
        if r > 1.0:
            n_img = sum(1 for cs in per_img.values() if c in cs)
            print(f"    {EWASTE_CLASSES[c]:<18} in {n_img:>6} images ({n_img / len(per_img):.2%})  repeat x{r:.2f}")
    return out


def _write_dataset_yaml(output_dir: Path, tile: int = 768, overlap: int = 128,
                        train_list: Optional[Path] = None, sources: Optional[list] = None):
    """Write YOLO-format dataset.yaml for RT-DETR / Ultralytics training."""
    train_ref = train_list.name if train_list else "train/images"
    yaml_content = f"""# EWADS Dataset Configuration
# Generated by 03_augmentation_pipeline.py

path: {output_dir.resolve()}
train: {train_ref}
val:   val/images
test:  ../test/images   # Never tiled, never augmented. Evaluate FULL-FRAME with 08_tiled_eval.py

# Read by 05/06/07/08: train imgsz == tile, and inference must tile the same way.
ewads:
  tile: {tile}
  overlap: {overlap}
  sources: {json.dumps(list(sources or []))}   # input splits in TRAIN (05 --synthetic_only refuses 'real')

nc: {len(EWASTE_CLASSES)}

names:
"""
    for i, cls in enumerate(EWASTE_CLASSES):
        yaml_content += f"  {i}: {cls}\n"

    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)


if __name__ == "__main__":
    main()
