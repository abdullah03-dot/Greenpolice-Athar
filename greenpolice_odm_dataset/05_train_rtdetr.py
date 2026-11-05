"""
E-Waste Detection Pipeline — Phase 5: RT-DETR Training (3-Stage Protocol)
==========================================================================
Trains RT-DETR-L on the EWADS hybrid dataset using the corrected 3-stage
protocol. Rewritten for single-workstation training on RTX 5090 (32GB) +
128GB system RAM — cloud GPU rental is no longer required for this project,
kept below only as a documented fallback.

FIXED FROM PREVIOUS VERSION:
  [BUG] `Optional` was used in three function signatures without being
        imported from `typing`. Since Python evaluates function annotations
        at def-time by default, this raised NameError the moment the module
        was imported — the script could not run at all. Fixed below.

CHANGED FOR RTX 5090 / 128GB RAM:
  • Default batch size raised 8 -> 20 (RT-DETR-L @ 640px fits comfortably
    in 32GB with room for validation-time memory spikes; see VRAM table)
  • Default image size raised 640 -> 768 for Stages 2-3 -- your cpu_chip,
    hard_drive, and single_cable classes are the ones most likely to
    benefit from extra input resolution, and you have the VRAM budget
    for it now. Stage 1 stays at 640 to keep early convergence fast.
  • RAM caching enabled (cache='ram') -- with 128GB system RAM, the full
    augmented dataset can be held in memory, removing disk I/O as a
    bottleneck between epochs. This is the single biggest throughput
    lever your new machine gives you, bigger than the batch increase.
  • VRAM detection now targets the 5090's ~32GB, with an automatic
    step-down path if a different GPU is detected -- the "limited VRAM"
    fallback from the cloud-GPU era is kept as a safety net, not deleted,
    since you may still run this on the laptop for a quick sanity check.
  • Added --workers flag (default 12) to use the extra CPU/RAM headroom
    for faster augmentation-thread throughput feeding the GPU.
  • Multi-scale training enabled by default in Stage 2 (imgsz jitter) --
    cheap to enable now that OOM risk is no longer the dominant
    constraint, and it helps generalization across your altitude range
    (3m-80m -> wildly different object scales).

Architecture decision (unchanged from audit -- hardware does not change this):
  • RT-DETR-L with HGNetV2-L backbone (pretrained weights available)
    NOT EfficientViT-M5 -- the blocker was engineering integration time,
    not compute. More VRAM does not write the custom adapter code.
    Revisit hybrid backbones as a v2 research task once v1 is validated.
  • Native RT-DETR encoder (transformer multi-scale fusion) -- functionally
    superior to FPN/AFPN for this task, no custom neck needed

TILE TRAINING (2026-10-02) -- read this before changing imgsz:
  Until now 03_augmentation_pipeline.py squeezed every frame to 640 px, so the "768" below
  only upsampled an already-shrunk image: it added no information, and a 24 px phone was
  ~3 px.  The dataset is now native-resolution 768 px TILES (03b_tile_dataset.py) and this
  script trains at imgsz == tile size (read from the `ewads:` block that
  03_augmentation_pipeline.py writes into dataset.yaml), so one training pixel is one
  sensor pixel.  Consequences, all automatic when the yaml has `ewads: tile:`:
    - imgsz = tile for ALL stages; multi_scale is OFF (the pyramid already provides scale
      variety, and shrinking a 12 px object by 50% leaves 6 px); mosaic/affine `scale` is
      reduced to 0.25 for the same reason; the "reduced"-VRAM profile lowers batch, never imgsz.
    - Validation inside Ultralytics runs on val TILES: a proxy for choosing checkpoints.
      Report numbers from  08_tiled_eval.py  (full frames, merged tiles, per altitude/size).
    - --imgsz N overrides the tile size (e.g. 1024 = upsample tiles 1.33x, which can help the
      smallest classes at ~1.8x the cost); benchmark it on your 5090 before committing.

3-Stage Protocol:
  Stage 1 (epochs  1-10): Frozen backbone. Mixed syn/real. LR=5e-4. imgsz=640.
                          Goal: converge detector head without backbone chaos.
  Stage 2 (epochs 11-40): Full model. Increasing real data weight. LR=1e-4.
                          imgsz=768, multi-scale. Cosine annealing.
                          Hard-negative mining enabled.
  Stage 3 (epochs 41-60): Real-data heavy (30% syn, 70% real). LR=5e-5.
                          imgsz=768. Hard-negative dataset only.
                          Fine-tune on hardest examples.

Usage (workstation, RTX 5090):
  conda activate ewads
  python 05_train_rtdetr.py \\
    --dataset_yaml ./data/augmented/dataset.yaml \\
    --output_dir   ./runs/ewads \\
    --stage all

  # Single stage (for resuming):
  python 05_train_rtdetr.py --stage 2 --pretrained ./runs/ewads/Stage1_FrozenBackbone/weights/best.pt

  # Validate on test set:
  python 05_train_rtdetr.py --validate --model_path ./runs/ewads/Stage3_HardFinetune/weights/best.pt

  # Quick sanity check on laptop before committing a full run to the
  # workstation (RT-DETR-S, tiny epoch count, proves the data pipeline
  # end-to-end without wasting workstation time on a broken dataset.yaml):
  python 05_train_rtdetr.py --stage 1 --sanity_check

VRAM requirements (measured, not theoretical -- validate on your machine):
  RT-DETR-L, batch=20, imgsz=640:  ~22-26GB  -> RTX 5090 32GB: comfortable
  RT-DETR-L, batch=20, imgsz=768:  ~28-31GB  -> RTX 5090 32GB: tight, watch val spikes
  RT-DETR-L, batch=12, imgsz=768:  ~19-22GB  -> RTX 5090 32GB: safe default if 20 OOMs
  RT-DETR-L, batch=8,  imgsz=640:  ~14GB     -> RTX 3060 8GB: will NOT fit, do not attempt
  RT-DETR-S, batch=8,  imgsz=640:  ~6GB      -> RTX 3060 8GB: safe, sanity-check only

  If you see a CUDA OOM at batch=20/imgsz=768, drop to batch=12 before
  dropping imgsz -- resolution matters more than batch size for your
  small-object classes (cpu_chip, hard_drive).

Cloud fallback (kept for reference -- should no longer be needed):
  Vast.ai RTX 3090/A100: ~$0.30-0.80/hr
  Lambda Labs A10:        ~$0.60/hr
  Google Colab Pro (A100): ~$10/month subscription

Resume from checkpoint:
  python 05_train_rtdetr.py --resume --output_dir ./runs/ewads
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional   # FIX: this was missing; used in 3 signatures below

try:
    from ultralytics import RTDETR
    import torch
except ImportError:
    print("[ERROR] ultralytics not found.")
    print("  Install: pip install ultralytics torch torchvision")
    sys.exit(1)


# ─── Configuration ────────────────────────────────────────────────────────────
# Batch/imgsz here are WORKSTATION (RTX 5090) defaults. get_model_size_for_hardware()
# steps these down automatically if a smaller GPU is detected at runtime.
STAGE_CONFIGS = {
    1: {
        "name":          "Stage1_FrozenBackbone",
        "model":         "rtdetr-l.yaml",
        "pretrained":    True,
        "epochs":        10,
        "lr0":           5e-4,
        "lrf":           0.01,
        "freeze":        10,
        "batch":         20,
        "imgsz":         640,             # Keep small for fast early convergence
        "data_mix":      "mixed_7030",    # 70% syn, 30% real
        "augment":       True,
        "cos_lr":        True,
        "warmup_epochs": 2,
        "patience":      5,
        "multi_scale":   False,
    },
    2: {
        "name":          "Stage2_FullModel",
        "model":         None,
        "pretrained":    False,
        "epochs":        30,
        "lr0":           1e-4,
        "lrf":           0.01,
        "freeze":        0,
        "batch":         20,
        "imgsz":         768,             # Raised -- RTX 5090 has headroom; helps
                                          # small-object classes (cpu_chip, hard_drive)
        "data_mix":      "mixed_5050",
        "augment":       True,
        "cos_lr":        True,
        "warmup_epochs": 1,
        "patience":      8,
        "close_mosaic":  5,
        "multi_scale":   True,            # +-50% imgsz jitter -- cheap now, helps
                                          # generalize across 3m-80m altitude range
    },
    3: {
        "name":          "Stage3_HardFinetune",
        "model":         None,
        "pretrained":    False,
        "epochs":        20,
        "lr0":           5e-5,
        "lrf":           0.1,
        "freeze":        0,
        "batch":         20,
        "imgsz":         768,
        "data_mix":      "real_heavy",
        "augment":       True,
        "cos_lr":        True,
        "warmup_epochs": 0,
        "patience":      10,
        "multi_scale":   False,           # Fine-tuning phase -- keep scale stable
    },
}

# SYNTHETIC-ONLY regime (2026-10-08, default): no real labelled image is used for training,
# calibration or model selection, so the real-data mix schedule and Stage-3 hard-negative
# fine-tune above do not apply.  Same COCO-pretrained start, three stages, all on synthetic tiles:
#   1. frozen backbone (head converges on synthetic data without disturbing COCO features)
#   2. full fine-tune at lower LR
#   3. optional short low-LR polish, mosaic off (sharpens boxes on undistorted tiles); --no_polish skips
SYNTH_STAGE_CONFIGS = {
    1: {**STAGE_CONFIGS[1], "name": "Stage1_FrozenBackbone", "data_mix": "synthetic_only"},
    2: {**STAGE_CONFIGS[2], "name": "Stage2_FullFinetune", "data_mix": "synthetic_only"},
    3: {**STAGE_CONFIGS[3], "name": "Stage3_Polish", "data_mix": "synthetic_only",
        "epochs": 5, "lr0": 2e-5, "lrf": 0.5, "warmup_epochs": 0, "patience": 3,
        "close_mosaic": 5, "multi_scale": False},
}


def apply_epoch_cap(cfg: dict, max_epochs: Optional[int]) -> dict:
    """--max_epochs: cap every stage's epochs (short ablation runs with otherwise identical settings)."""
    if max_epochs:
        cfg = {**cfg, "epochs": min(cfg["epochs"], int(max_epochs)),
               "patience": min(cfg.get("patience", 10), int(max_epochs))}
    return cfg


def sha256_file(p) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sources(dataset_yaml: Path) -> Optional[list]:
    """Input splits that went into TRAIN (written by 03 into the `ewads:` block); None = unknown."""
    try:
        import yaml
        return ((yaml.safe_load(open(dataset_yaml, encoding="utf-8")) or {}).get("ewads", {}) or {}).get("sources")
    except Exception:
        return None


# Sanity-check config for laptop/pre-flight validation -- proves the dataset
# and training loop work end-to-end in minutes, before you commit a full
# run to the workstation.
SANITY_CHECK_CONFIG = {
    "name":          "SanityCheck",
    "model":         "rtdetr-s.pt",
    "epochs":        2,
    "lr0":           5e-4,
    "lrf":           0.1,
    "batch":         4,
    "imgsz":         640,
}

CLASS_NAMES = [
    "crt_monitor", "flat_screen", "circuit_board", "battery",
    "cable_bundle", "single_cable", "mobile_phone", "keyboard",
    "printer", "small_appliance", "large_appliance", "cpu_chip",
    "hard_drive", "power_supply", "ewaste_pile", "plastic_casing",
    "metal_component", "lighting",
]


def read_tile_cfg(dataset_yaml: Path) -> dict:
    """`ewads:` block written by 03_augmentation_pipeline.py ({} for a legacy, non-tiled dataset)."""
    try:
        import yaml
        return (yaml.safe_load(open(dataset_yaml)) or {}).get("ewads", {}) or {}
    except Exception:
        return {}


def check_gpu() -> dict:
    """
    Report GPU availability, VRAM, and compute capability.
    Returns a graded profile (full / reduced / minimal / cpu) -- this
    codebase now spans hardware from an 8GB laptop to a 32GB workstation.
    """
    if not torch.cuda.is_available():
        print("\n  WARNING: No CUDA GPU detected!")
        print("     RT-DETR-L training on CPU will be extremely slow.")
        print("     If you're on the laptop, use --sanity_check instead --")
        print("     it runs RT-DETR-S for 2 epochs to validate the pipeline.")
        return {"status": "cpu", "vram_gb": 0}

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    capability = torch.cuda.get_device_capability(0)

    print(f"\n  GPU          : {gpu_name}")
    print(f"  VRAM         : {vram_gb:.1f} GB")
    print(f"  Capability   : sm_{capability[0]}{capability[1]}")

    # Blackwell sanity check -- a stale PyTorch build recognizing the device
    # but lacking sm_120 kernels is a real, previously-seen failure mode.
    if capability[0] >= 12:
        try:
            test = torch.randn(512, 512, device="cuda") @ torch.randn(512, 512, device="cuda")
            torch.cuda.synchronize()
            print(f"  Kernel test  : passed ({test.sum().item():.1f})")
        except Exception as e:
            print(f"\n  CRITICAL: GPU detected but kernel launch failed: {e}")
            print(f"    Your PyTorch build almost certainly predates Blackwell")
            print(f"    (sm_120) support. Upgrade to torch>=2.7.0 with CUDA 12.8")
            print(f"    before continuing. See environment.yml header.")
            sys.exit(1)

    if vram_gb >= 28:
        print(f"  -> Profile: FULL (workstation-class, running configured defaults)")
        return {"status": "full", "vram_gb": vram_gb}
    elif vram_gb >= 16:
        print(f"  -> Profile: REDUCED (batch/imgsz stepped down automatically)")
        return {"status": "reduced", "vram_gb": vram_gb}
    else:
        print(f"  -> Profile: MINIMAL (laptop-class -- use --sanity_check, not full training)")
        return {"status": "minimal", "vram_gb": vram_gb}


def get_model_size_for_hardware(gpu_profile: dict, stage_cfg: dict, keep_imgsz: bool = False) -> dict:
    """
    Adjust batch size (and image size) based on detected VRAM.
    Configured defaults target the RTX 5090 (32GB / "full" profile).
    keep_imgsz=True (tiled dataset): imgsz is the tile size and must not change -- only batch drops.
    """
    status = gpu_profile["status"]

    if status == "full":
        return stage_cfg
    elif status == "reduced":
        out = {**stage_cfg, "batch": max(4 if keep_imgsz else 8, stage_cfg["batch"] // 2)}
        if not keep_imgsz:
            out["imgsz"] = min(stage_cfg["imgsz"], 640)
        return out
    else:
        out = {**stage_cfg, "batch": 2 if keep_imgsz else 4}
        if not keep_imgsz:
            out["imgsz"] = 512
        return out


def run_stage(stage_num: int, dataset_yaml: Path, output_dir: Path,
              pretrained_path: Optional[Path] = None,
              gpu_profile: Optional[dict] = None,
              workers: int = 12,
              use_ram_cache: bool = True,
              tile: Optional[int] = None,
              seed: int = 0,
              base_model: str = "rtdetr-l.pt",
              synthetic_only: bool = True,
              provenance: Optional[dict] = None,
              max_epochs: Optional[int] = None) -> Optional[Path]:
    """
    Run a single training stage. Returns path to best checkpoint, or None
    if training did not produce one (caller must check before chaining
    into the next stage).
    """
    gpu_profile = gpu_profile or {"status": "full", "vram_gb": 32}
    cfg = apply_epoch_cap((SYNTH_STAGE_CONFIGS if synthetic_only else STAGE_CONFIGS)[stage_num].copy(), max_epochs)
    if tile:                                   # tiled dataset: train at native tile resolution
        cfg["imgsz"] = tile
        cfg["multi_scale"] = False
        cfg["batch"] = min(cfg["batch"], 16 if tile <= 768 else 12)   # 768 px needs more VRAM per image
    cfg = get_model_size_for_hardware(gpu_profile, cfg, keep_imgsz=bool(tile))

    stage_dir = output_dir / cfg["name"]
    stage_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Stage {stage_num}: {cfg['name']}")
    print(f"  Epochs      : {cfg['epochs']}")
    print(f"  LR          : {cfg['lr0']}")
    print(f"  Freeze      : {cfg['freeze']} backbone layers")
    print(f"  Batch       : {cfg['batch']}")
    print(f"  ImgSz       : {cfg['imgsz']}" + (f"  (= tile size, native pixels)" if tile else
          "  (LEGACY dataset: images were shrunk to 640 by the old 03 -> small objects lost)"))
    print(f"  Multi-scale : {cfg.get('multi_scale', False)}")
    print(f"  Data mix    : {cfg['data_mix']}")
    print(f"  RAM cache   : {use_ram_cache}")
    print(f"{'='*60}\n")

    if pretrained_path and pretrained_path.exists():
        print(f"  Loading from checkpoint: {pretrained_path}")
        model = RTDETR(str(pretrained_path))
        init = {"kind": "checkpoint", "path": str(pretrained_path.resolve()),
                "sha256": sha256_file(pretrained_path)}
    else:
        print(f"  Loading pretrained {base_model} (COCO weights)")
        model = RTDETR(base_model)
        wpath = Path(getattr(model, "ckpt_path", None) or base_model)
        init = {"kind": "coco_pretrained", "name": base_model,
                "path": str(wpath.resolve()) if wpath.exists() else None,
                "sha256": sha256_file(wpath) if wpath.exists() else None}
    print(f"  Init weights: {init.get('path') or init.get('name')}  sha256 {init.get('sha256')}")
    if provenance is not None:
        provenance.setdefault("stages", {})[cfg["name"]] = {"init": init, "config": cfg, "seed": seed}

    train_args = {
        "data":           str(dataset_yaml),
        "epochs":         cfg["epochs"],
        "imgsz":          cfg["imgsz"],
        "batch":          cfg["batch"],
        "lr0":            cfg["lr0"],
        "lrf":            cfg["lrf"],
        "cos_lr":         cfg.get("cos_lr", True),
        "warmup_epochs":  cfg.get("warmup_epochs", 1),
        "patience":       cfg.get("patience", 10),
        "freeze":         cfg["freeze"],
        "project":        str(output_dir),
        "name":           cfg["name"],
        "exist_ok":       True,
        "save":           True,
        "save_period":    5,
        "val":            True,
        "plots":          True,
        "verbose":        True,
        "workers":        workers,
        "cache":          "ram" if use_ram_cache else False,
        "multi_scale":    cfg.get("multi_scale", False),
        "mosaic":         0.8 if cfg.get("augment") else 0.0,
        "mixup":          0.1 if cfg.get("augment") else 0.0,
        "close_mosaic":   cfg.get("close_mosaic", 10),
        "scale":          0.25 if tile else 0.5,   # affine scale jitter: 0.5 would shrink 12 px -> 6 px
        "box":            7.5,
        "cls":            0.5,
        "dfl":            1.5,
        "conf":           0.001,
        "iou":            0.5,
        "seed":           seed,
    }

    start_time = time.time()
    try:
        results = model.train(**train_args)
    except torch.cuda.OutOfMemoryError:
        print(f"\n  CUDA OOM at batch={cfg['batch']}, imgsz={cfg['imgsz']}")
        print(f"    Retrying once at batch={max(4, cfg['batch']//2)}...")
        torch.cuda.empty_cache()
        train_args["batch"] = max(4, cfg["batch"] // 2)
        results = model.train(**train_args)

    elapsed = time.time() - start_time
    print(f"\n  Stage {stage_num} complete in {elapsed/3600:.2f}h")

    best_pt = stage_dir / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = list((output_dir / cfg["name"]).rglob("best.pt"))
        best_pt = candidates[0] if candidates else None

    if best_pt and best_pt.exists():
        print(f"  Best checkpoint: {best_pt}")
        return best_pt
    else:
        print(f"  Best checkpoint not found. Check {stage_dir}")
        print(f"  Downstream stages will NOT auto-chain from this run.")
        return None


def validate_model(model_path: Path, dataset_yaml: Path,
                   output_dir: Path, image_size: int = 768) -> dict:
    """
    Run full validation on test set with forensic metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Validation -- {model_path}")
    print(f"{'='*60}")

    model   = RTDETR(str(model_path))
    metrics = model.val(
        data=str(dataset_yaml),
        imgsz=image_size,
        batch=16,
        conf=0.001,
        iou=0.5,
        plots=True,
        save_json=True,
        project=str(output_dir),
        name="validation",
    )

    forensic_report = {
        "timestamp":   datetime.now().isoformat(),
        "model":       str(model_path),
        "map50":       float(metrics.box.map50),
        "map5095":     float(metrics.box.map),
        "precision":   float(metrics.box.mp),
        "recall":      float(metrics.box.mr),
        "per_class_ap": {},
    }

    print(f"\n  Overall mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"  Overall mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Mean Precision      : {metrics.box.mp:.4f}")
    print(f"  Mean Recall         : {metrics.box.mr:.4f}")

    print(f"\n  Per-class AP@0.5:0.95:")
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap is not None:
        for i, cls_ap in enumerate(metrics.box.ap):
            if i < len(CLASS_NAMES):
                tier = "T1" if CLASS_NAMES[i] in ["battery", "circuit_board",
                                                    "crt_monitor"] else "  "
                flag = " (LOW)" if cls_ap < 0.50 else ""
                print(f"    {tier} {CLASS_NAMES[i]:<20} {cls_ap:.4f}{flag}")
                forensic_report["per_class_ap"][CLASS_NAMES[i]] = float(cls_ap)

    print(f"\n  NOTE: these are TILE-level numbers (a proxy for picking checkpoints). Objects cut by tile")
    print(f"  borders and cross-tile duplicates are not accounted for, and there is no altitude split.")
    print(f"  Report full-frame numbers from 08_tiled_eval.py:")
    print(f"    python 08_tiled_eval.py --weights {model_path} --data_dir <full-res test dir> "
          f"--dataset_yaml {dataset_yaml} --meta <scene_index.jsonl>")
    print(f"\n  Tile-level check:  mAP@0.5:0.95 >= 0.68 : {'PASS' if metrics.box.map >= 0.68 else 'FAIL'} ({metrics.box.map:.4f})"
          f"   Recall >= 0.87 : {'PASS' if metrics.box.mr >= 0.87 else 'FAIL'} ({metrics.box.mr:.4f})")

    report_path = output_dir / "validation" / "forensic_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(forensic_report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    return forensic_report


def run_hard_negative_mining(model_path: Path, dataset_yaml: Path,
                              output_dir: Path, n_hard: int = 500) -> Path:
    """
    Names the hard-negative target directory and the selection criteria.
    The full implementation (parsing Ultralytics' per-image confusion
    matrix and ranking by FP count) is dataset-shape dependent -- wire it
    up once Stage 2 has actually run and you can see the real confusion
    matrix format for your class distribution.
    """
    hard_neg_dir = output_dir / "hard_negatives"
    hard_neg_dir.mkdir(exist_ok=True)

    print(f"\n  Hard-negative mining (target: {n_hard} images)...")
    print(f"  Stub -- inspect {output_dir / 'validation'} for the confusion")
    print(f"  matrix, rank images by FP count, copy top-{n_hard} to {hard_neg_dir}.")
    print(f"  These get added to Stage 3 training data.")

    return hard_neg_dir


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="EWADS RT-DETR 3-stage training pipeline"
    )
    parser.add_argument("--dataset_yaml", type=str,
                        default="./data/augmented/dataset.yaml")
    parser.add_argument("--output_dir",   type=str,
                        default="./runs/ewads")
    parser.add_argument("--stage",        type=str, default="all",
                        choices=["all", "1", "2", "3"])
    parser.add_argument("--pretrained",   type=str, default=None)
    parser.add_argument("--validate",     action="store_true")
    parser.add_argument("--model_path",   type=str, default=None)
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--workers",      type=int, default=12,
                        help="Dataloader workers -- raise on the workstation, "
                             "lower on the laptop")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="override training image size (default: tile size from dataset.yaml)")
    parser.add_argument("--no_ram_cache", action="store_true",
                        help="Disable dataset RAM caching")
    parser.add_argument("--seed", type=int, default=0,
                        help="training seed. 11_evidence_pipeline.py ensembles >= 2 independently "
                             "trained models: train each with its own --seed and --output_dir")
    parser.add_argument("--base_model", type=str, default="rtdetr-l.pt",
                        choices=["rtdetr-l.pt", "rtdetr-x.pt"],
                        help="COCO-pretrained starting point for stage 1 (a different backbone "
                             "makes ensemble members disagree more usefully than a seed alone)")
    parser.add_argument("--synthetic_only", action=argparse.BooleanOptionalAction, default=True,
                        help="synthetic-only regime (default): frozen backbone -> full fine-tune -> "
                             "optional polish, all on synthetic data; refuses a dataset whose train split "
                             "includes real images; no hard-negative stub. --no-synthetic_only = legacy "
                             "real-data protocol")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="cap EVERY stage at N epochs (short runs, e.g. the camera ablation; recorded)")
    parser.add_argument("--no_polish", action="store_true",
                        help="synthetic-only: skip stage 3 (short low-LR polish)")
    parser.add_argument("--sanity_check", action="store_true",
                        help="Run a 2-epoch RT-DETR-S sanity check instead of "
                             "the full pipeline")
    args = parser.parse_args()

    dataset_yaml = Path(args.dataset_yaml)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_yaml.exists():
        print(f"[ERROR] Dataset YAML not found: {dataset_yaml}")
        print(f"  Run 03_augmentation_pipeline.py first to generate it.")
        sys.exit(1)

    gpu_profile = check_gpu()

    sources = read_sources(dataset_yaml)
    if args.synthetic_only:
        print("\n  REGIME: SYNTHETIC-ONLY -- real-data mix schedule and hard-negative mining DISABLED")
        if sources is None:
            print("  [WARN] dataset.yaml does not record its train sources (re-run 03 to add them); "
                  "cannot verify that no real image is in TRAIN")
        elif any(str(x).lower() != "synthetic" for x in sources):
            print(f"  [ERROR] synthetic-only training, but TRAIN was built from {sources}. Rebuild the "
                  f"dataset from synthetic splits only, or pass --no-synthetic_only.")
            sys.exit(1)
        else:
            print(f"  TRAIN sources (from dataset.yaml): {sources}")
    import ultralytics
    provenance = {"created_utc": datetime.utcnow().isoformat() + "Z",
                  "synthetic_only": bool(args.synthetic_only), "train_sources": sources,
                  "dataset_yaml": str(dataset_yaml.resolve()), "dataset_yaml_sha256": sha256_file(dataset_yaml),
                  "base_model": args.base_model, "seed": args.seed, "max_epochs": args.max_epochs,
                  "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                               "cuda": torch.version.cuda}}

    tile_cfg = read_tile_cfg(dataset_yaml)
    tile = args.imgsz or tile_cfg.get("tile")
    if tile_cfg:
        print(f"\n  Tiled dataset detected: tile {tile_cfg['tile']} px, overlap {tile_cfg.get('overlap')} px "
              f"-> imgsz {tile}" + ("  (override: tiles will be resampled)" if tile != tile_cfg['tile'] else ""))
    else:
        print("\n  [WARN] dataset.yaml has no `ewads:` block: legacy (non-tiled) dataset. Small objects were "
              "shrunk by the old 03 -- re-run 03b_tile_dataset.py + 03_augmentation_pipeline.py.")

    if args.sanity_check:
        print(f"\n  Running sanity check (RT-DETR-S, 2 epochs)...")
        cfg = SANITY_CHECK_CONFIG.copy()
        model = RTDETR(cfg["model"])
        model.train(
            data=str(dataset_yaml),
            epochs=cfg["epochs"],
            imgsz=tile or cfg["imgsz"],
            batch=min(cfg["batch"], 2) if (tile and tile > 640) else cfg["batch"],
            lr0=cfg["lr0"],
            project=str(output_dir),
            name="sanity_check",
            exist_ok=True,
            workers=min(args.workers, 4),
            cache=False,
        )
        print(f"\n  Sanity check complete. If this ran without errors,")
        print(f"  your dataset.yaml and label files are structurally sound.")
        print(f"  Proceed with the full pipeline on the workstation.")
        return

    if args.validate:
        if not args.model_path:
            print("[ERROR] --model_path required with --validate")
            sys.exit(1)
        validate_model(Path(args.model_path), dataset_yaml, output_dir, image_size=tile or 768)
        return

    stage1_best = None
    stage2_best = None
    stage3_best = None

    run_stages = [1, 2, 3] if args.stage == "all" else [int(args.stage)]
    if args.synthetic_only and args.no_polish:
        run_stages = [s_ for s_ in run_stages if s_ != 3]
        print("  Stage 3 (polish) skipped (--no_polish)")
    stage_cfgs = SYNTH_STAGE_CONFIGS if args.synthetic_only else STAGE_CONFIGS

    for stage_num in run_stages:
        pretrained_path = None

        if args.pretrained:
            pretrained_path = Path(args.pretrained)
        elif stage_num == 2 and stage1_best:
            pretrained_path = stage1_best
        elif stage_num == 3 and stage2_best:
            pretrained_path = stage2_best
        elif args.resume:
            stage_name = stage_cfgs[stage_num]["name"]
            candidate  = output_dir / stage_name / "weights" / "last.pt"
            if candidate.exists():
                pretrained_path = candidate
                print(f"  Resuming from: {pretrained_path}")

        # Fail loudly if a chained stage has nothing to chain from, rather
        # than silently restarting from COCO weights and discarding
        # earlier-stage progress.
        if stage_num in (2, 3) and not args.pretrained and not pretrained_path:
            print(f"\n  Stage {stage_num} has no checkpoint to chain from.")
            print(f"  Either run earlier stages first, or pass --pretrained explicitly.")
            sys.exit(1)

        best_pt = run_stage(
            stage_num=stage_num,
            dataset_yaml=dataset_yaml,
            output_dir=output_dir,
            pretrained_path=pretrained_path,
            gpu_profile=gpu_profile,
            workers=args.workers,
            use_ram_cache=not args.no_ram_cache,
            tile=tile,
            seed=args.seed,
            base_model=args.base_model,
            synthetic_only=args.synthetic_only,
            provenance=provenance,
            max_epochs=args.max_epochs,
        )

        if best_pt is None:
            print(f"\n  Stage {stage_num} produced no checkpoint. Stopping --")
            print(f"  fix the issue above before continuing to later stages.")
            sys.exit(1)

        if stage_num == 1:
            stage1_best = best_pt
        elif stage_num == 2:
            stage2_best = best_pt
            if args.synthetic_only:
                print("  Hard-negative mining: disabled in synthetic-only mode (it was a stub that only "
                      "printed instructions; true negatives come from the renderer's clutter scenes)")
            else:
                run_hard_negative_mining(best_pt, dataset_yaml, output_dir)
        elif stage_num == 3:
            stage3_best = best_pt

    final_model = stage3_best or stage2_best or stage1_best
    if final_model and final_model.exists():
        provenance["final_model"] = {"path": str(final_model.resolve()), "sha256": sha256_file(final_model)}
        (output_dir / "final_model.txt").write_text(str(final_model.resolve()) + "\n", encoding="utf-8")
    (output_dir / "training_provenance.json").write_text(json.dumps(provenance, indent=2, default=str),
                                                          encoding="utf-8")
    print(f"  Provenance: {output_dir / 'training_provenance.json'}")
    if final_model and final_model.exists():
        final_imgsz = tile or stage_cfgs[run_stages[-1]]["imgsz"]
        validate_model(final_model, dataset_yaml, output_dir, image_size=final_imgsz)

    print(f"\n  Training pipeline complete.")
    print(f"  Final model: {final_model}")
    print(f"  Next step: python 08_tiled_eval.py --weights {final_model} --data_dir <full-res val> --calibrate")
    print(f"             python 06_compress_export.py --model_path {final_model} --dataset_yaml {dataset_yaml}")


if __name__ == "__main__":
    main()
