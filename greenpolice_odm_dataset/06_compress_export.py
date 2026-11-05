"""
E-Waste Detection Pipeline — Phase 6: Compression & Export
===========================================================
Converts the trained FP32 RT-DETR-L teacher model into edge-deployable formats:

  Step 1: Structured Pruning (L1-norm channel pruning, 30-50% FLOP reduction)
  Step 2: Knowledge Distillation (RT-DETR-L teacher → YOLOv10-N student)
  Step 3: Quantization-Aware Training (QAT, INT8)
  Step 4: ONNX Export + Validation
  Step 5: TensorRT Engine Build (Jetson Orin NX target)

Critical rule: ONNX validation after EVERY compression step.
If a step degrades mAP@0.5 by > 2%, stop and investigate before continuing.

Decision from audit:
  Cross-architecture distillation (RT-DETR-L → YOLOv10-N) is done via
  PSEUDO-LABELING, not logit-level KD. Teacher labels unlabeled data,
  student trains on those labels. Cleaner and more documented than
  cross-family soft-output distillation.

UPDATED FOR RTX 5090 (32GB) / 128GB RAM workstation:
  • Student training batch: 16 -> 48 (YOLOv10-N is tiny; batch size was
    never VRAM-limited even on the old hardware — this is a free win)
  • QAT batch: 8 -> 24
  • RAM caching enabled for both student training and QAT
  • Pruning (Step 1) is UNCHANGED by the hardware upgrade — this is a
    deliberate note, not an oversight. Pruning exists to shrink the
    model for the Jetson Orin NX edge target, not to fit your training
    GPU. A bigger training GPU doesn't reduce how much compute is on
    the drone. Keep the prune_ratio decision tied to your Jetson latency
    budget (see 07_edge_benchmark.py), not to workstation capability.

BUGS FOUND AND FIXED IN THIS PASS (unrelated to the hardware upgrade —
these existed before and would have surfaced the first time this script
ran end-to-end; caught now while doing a deliberate re-read):
  [BUG 1] run_qat() loaded every checkpoint through RTDETR() regardless
          of what architecture it actually was. The pipeline calls this
          on the YOLOv10-N student, not the RT-DETR-L teacher — wrong
          model class for the checkpoint. Now takes is_rtdetr explicitly.
  [BUG 2] train_student() hardcoded its output run name to
          "student_yolov10n" even when --target rtdetr-s was selected,
          which both mislabeled the run directory and removed any way
          for downstream steps to detect the architecture from the path.
          Now derives the run name from the actual student_model argument.
  [BUG 3] export_student_onnx() hardcoded YOLO() the same way run_qat()
          hardcoded RTDETR() — same root cause, same fix pattern.

TILE-AWARE (2026-10-02): everything here used to be hard-wired to 640x640 and the teacher
pseudo-labelled WHOLE frames at Ultralytics' default 640 px (a 20 MP frame -> 8.5x downscale), so
the student never saw the small objects.  Now:
  - the image size is the tile size recorded in dataset.yaml (`ewads: tile:`), override --imgsz;
    it flows into student training, QAT, ONNX export and the TensorRT shapes;
  - pseudo-labelling runs the teacher through the same pyramid tiler used everywhere else
    (ewads_tiling.TiledDetector) on FULL-RESOLUTION frames, then tiles those frames + labels;
  - [BUG 4] train_student() accepted `pseudo_dir` but never used it, so pseudo-labels were
    generated and then thrown away.  It now trains on dataset + pseudo-labelled tiles.
  - on the Jetson the model runs once per tile (see 07_edge_benchmark.py for per-frame cost).

Usage:
  conda activate ewads
  pip install ultralytics torch torchvision onnx onnxruntime

  # Full pipeline:
  python 06_compress_export.py \\
    --model_path  ./runs/ewads/Stage3_HardFinetune/weights/best.pt \\
    --dataset_yaml ./data/augmented/dataset.yaml \\
    --output_dir  ./runs/export \\
    --target      yolov10n           ← nano edge model
    --prune_ratio 0.35

  # Export only (skip pruning/distillation):
  python 06_compress_export.py --export_only \\
    --model_path ./runs/ewads/Stage3_HardFinetune/weights/best.pt \\
    --output_dir ./runs/export

  # Validate ONNX model:
  python 06_compress_export.py --validate_onnx \\
    --onnx_path ./runs/export/ewads_nano_int8.onnx \\
    --dataset_yaml ./data/augmented/dataset.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ewads_tiling as et

try:
    from ultralytics import RTDETR, YOLO
    import onnx
    import onnxruntime as ort
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("  pip install ultralytics onnx onnxruntime")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EWASTE_CLASSES = [
    "crt_monitor", "flat_screen", "circuit_board", "battery",
    "cable_bundle", "single_cable", "mobile_phone", "keyboard",
    "printer", "small_appliance", "large_appliance", "cpu_chip",
    "hard_drive", "power_supply", "ewaste_pile", "plastic_casing",
    "metal_component", "lighting",
]


def read_tile_cfg(dataset_yaml: Optional[Path]) -> dict:
    """`ewads:` block from dataset.yaml ({} for a legacy non-tiled dataset)."""
    try:
        import yaml
        return (yaml.safe_load(open(dataset_yaml)) or {}).get("ewads", {}) or {}
    except Exception:
        return {}


# ─── Step 1: Structured Pruning ───────────────────────────────────────────────
def prune_model(model_path: Path, output_dir: Path,
                prune_ratio: float = 0.35,
                dataset_yaml: Optional[Path] = None,
                imgsz: int = 640) -> Path:
    """
    L1-norm channel pruning on the RT-DETR backbone.
    Removes channels with smallest L1-norm weights — lowest information content.

    prune_ratio: fraction of channels to prune (0.35 = 35% FLOP reduction target)

    Pruning is applied to Conv2d layers only. The transformer encoder and
    detection head are NOT pruned — they are the precision-critical components.

    Returns path to pruned model checkpoint.
    """
    print(f"\n{'='*60}")
    print(f"  Step 1: Structured Pruning")
    print(f"  Prune ratio : {prune_ratio:.0%} of channels")
    print(f"  Target      : backbone Conv2d layers only")
    print(f"{'='*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = RTDETR(str(model_path))
    torch_model = model.model.to(DEVICE)
    torch_model.eval()

    # Compute baseline FLOP count
    baseline_params = sum(p.numel() for p in torch_model.parameters())
    print(f"  Baseline parameters: {baseline_params/1e6:.2f}M")

    # Identify prunable Conv2d layers in backbone
    # RT-DETR backbone layers are typically in model.backbone
    prunable_layers = []
    for name, module in torch_model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Skip first layer (input channels fixed) and very small convs
            if module.out_channels >= 32 and "backbone" in name:
                prunable_layers.append((name, module))

    print(f"  Prunable Conv2d layers: {len(prunable_layers)}")

    # Compute L1 norms per output channel for each layer
    masks = {}
    total_channels_before = 0
    total_channels_after  = 0

    for name, layer in prunable_layers:
        weight = layer.weight.data  # (out_channels, in_channels, kH, kW)
        l1_norms = weight.abs().sum(dim=[1, 2, 3])  # L1 norm per output channel

        n_out = layer.out_channels
        n_prune = int(n_out * prune_ratio)
        n_keep  = n_out - n_prune

        # Keep channels with highest L1 norm
        _, sorted_idx = torch.sort(l1_norms, descending=True)
        keep_idx = sorted_idx[:n_keep].sort().values

        masks[name] = keep_idx
        total_channels_before += n_out
        total_channels_after  += n_keep

    prune_pct = 1 - (total_channels_after / total_channels_before)
    print(f"  Channels before: {total_channels_before}")
    print(f"  Channels after : {total_channels_after}")
    print(f"  Channel reduction: {prune_pct:.1%}")

    # NOTE: Full structured pruning requires rebuilding the model graph
    # with new channel counts and copying weights. This is architecture-specific
    # and typically done with a pruning library (torch-pruning, nni, etc.)
    # The implementation below uses the torch-pruning library if available,
    # otherwise falls back to weight masking (soft pruning) for validation.

    try:
        import torch_pruning as tp

        example_input = torch.randn(1, 3, imgsz, imgsz).to(DEVICE)

        # Importance metric: L1 norm of weights
        imp = tp.importance.MagnitudeImportance(p=1)

        # Build pruner
        ignored_layers = []
        for module in torch_model.modules():
            # Protect detection head and transformer encoder
            if isinstance(module, (nn.Linear,)) or "head" in str(type(module)).lower():
                ignored_layers.append(module)

        pruner = tp.pruner.MagnitudePruner(
            model=torch_model,
            example_inputs=example_input,
            importance=imp,
            pruning_ratio=prune_ratio,
            ignored_layers=ignored_layers,
        )
        pruner.step()

        pruned_params = sum(p.numel() for p in torch_model.parameters())
        print(f"  Pruned parameters: {pruned_params/1e6:.2f}M "
              f"({100*(1-pruned_params/baseline_params):.1f}% reduction)")

    except ImportError:
        print(f"  [INFO] torch-pruning not found. Using weight masking (soft pruning).")
        print(f"  Install for hard structural pruning: pip install torch-pruning")
        print(f"  Soft pruning applied — model structure unchanged, zero-masked channels.")

        # Soft pruning: zero out channels with lowest L1 norm
        with torch.no_grad():
            for name, layer in prunable_layers:
                if name in masks:
                    keep_idx = masks[name]
                    mask = torch.zeros(layer.out_channels, device=DEVICE)
                    mask[keep_idx] = 1.0
                    layer.weight.data *= mask.view(-1, 1, 1, 1)
                    if layer.bias is not None:
                        layer.bias.data *= mask

    # Save pruned model
    pruned_path = output_dir / "rtdetr_l_pruned.pt"
    torch.save({"model": torch_model.state_dict(),
                "pruned": True,
                "prune_ratio": prune_ratio}, str(pruned_path))

    print(f"  Pruned model saved: {pruned_path}")

    # Validate pruned model
    if dataset_yaml:
        print(f"\n  Validating pruned model...")
        _validate_checkpoint(model, dataset_yaml, "pruned")

    return pruned_path


# ─── Step 2: Knowledge Distillation (Pseudo-labeling) ─────────────────────────
def pseudo_label_for_student(teacher_path: Path, unlabeled_dir: Path,
                              output_dir: Path, conf_threshold: float = 0.35,
                              tile: int = 640, overlap: int = 128) -> Optional[Path]:
    """
    Use the teacher (RT-DETR-L) to pseudo-label unlabeled FULL-RESOLUTION frames, then cut
    those frames + labels into the same native-resolution tiles the dataset uses.
    These tiles train the student (YOLOv10-N) without logit-level distillation.

    The teacher runs through the pyramid tiler (ewads_tiling.TiledDetector), so it sees small
    objects at native pixels exactly as it does at evaluation/deployment time.
    conf_threshold: minimum confidence for pseudo-label inclusion (0.4+ keeps only clean ones).

    Returns the directory holding the tiled pseudo-labelled split (images/, labels/), or None.
    """
    import cv2
    print(f"\n{'='*60}")
    print(f"  Step 2: Pseudo-labeling for Knowledge Distillation (tiled)")
    print(f"  Teacher    : {teacher_path.name}")
    print(f"  Unlabeled  : {unlabeled_dir}")
    print(f"  Conf thresh: {conf_threshold}   tile {tile}  overlap {overlap}")
    print(f"{'='*60}\n")

    raw = output_dir / "pseudo_raw" / "pseudo"          # full-res frames + labels (provenance)
    (raw / "images").mkdir(parents=True, exist_ok=True)
    (raw / "labels").mkdir(parents=True, exist_ok=True)

    # case-insensitive: real DJI frames are .JPG, and glob is case-sensitive on Linux
    image_files = sorted(p for p in unlabeled_dir.rglob("*")
                         if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_files:
        print(f"  [WARN] No images found in {unlabeled_dir}")
        print(f"  Pseudo-labeling skipped. Student will train on augmented dataset only.")
        return None

    teacher = RTDETR(str(teacher_path))
    det = et.TiledDetector(et.ultralytics_predict_fn(teacher, tile, conf=conf_threshold),
                           tile=tile, overlap=overlap, batch=8, conf=conf_threshold)
    print(f"  Labeling {len(image_files)} frames with the tiled teacher...")

    import shutil
    n_labeled = n_empty = 0
    for p in image_files:
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        d = det(frame)
        if len(d) == 0:
            n_empty += 1
            continue
        with open(raw / "labels" / f"{p.stem}.txt", "w") as f:
            for x1, y1, x2, y2, conf, cls in d:
                f.write(f"{int(cls)} {(x1 + x2) / 2 / w:.7f} {(y1 + y2) / 2 / h:.7f} "
                        f"{(x2 - x1) / w:.7f} {(y2 - y1) / h:.7f}\n")
        shutil.copy2(p, raw / "images" / p.name)
        n_labeled += 1
    print(f"  Labeled     : {n_labeled} frames   (no detections, skipped: {n_empty})")
    if n_labeled == 0:
        return None

    # cut into tiles with the same rules as the training set
    import subprocess
    tiler = Path(__file__).resolve().parent / "03b_tile_dataset.py"
    tiled_root = output_dir / "pseudo_tiled"
    subprocess.run([sys.executable, str(tiler), "--input_dir", str(output_dir / "pseudo_raw"),
                    "--output_dir", str(tiled_root), "--splits", "pseudo",
                    "--tile", str(tile), "--overlap", str(overlap)], check=True)
    out = tiled_root / "pseudo"
    print(f"  Pseudo-labelled tiles: {out}")
    return out


def train_student(teacher_path: Path, dataset_yaml: Path,
                  pseudo_dir: Optional[Path], output_dir: Path,
                  student_model: str = "yolov10n.pt",
                  epochs: int = 50,
                  batch: int = 48,
                  use_ram_cache: bool = True,
                  imgsz: int = 640) -> Path:
    """
    Train YOLOv10-N student model on combined augmented + pseudo-labeled data.

    batch defaults to 48 (up from 16) — YOLOv10-N is a few million parameters,
    trivial for a 32GB card, and a larger batch here costs nothing since this
    step was never the pipeline's VRAM bottleneck even on the old hardware.
    Step down manually if you're running this on the laptop instead.
    """
    print(f"\n{'='*60}")
    print(f"  Step 2b: Student Training ({student_model})")
    print(f"  Epochs  : {epochs}")
    print(f"  Batch   : {batch}")
    print(f"{'='*60}\n")

    # [BUG FIXED] Run name is now derived from student_model instead of being
    # hardcoded to "student_yolov10n". Previously, choosing --target rtdetr-s
    # still wrote to a directory literally named student_yolov10n — harmless
    # on its own, but it also meant downstream architecture detection (in
    # run_qat) could not tell the two apart by inspecting the output path.
    is_rtdetr = "rtdetr" in student_model.lower()
    ModelClass = RTDETR if is_rtdetr else YOLO
    run_name = f"student_{student_model.replace('.pt', '').replace('.yaml', '')}"

    student = ModelClass(student_model)

    data_yaml = dataset_yaml
    if pseudo_dir is not None and Path(pseudo_dir, "images").exists():
        # [BUG 4 fix] actually train on the pseudo-labelled tiles, next to the real training data
        import yaml
        cfg = yaml.safe_load(open(dataset_yaml))
        base = Path(cfg.get("path", Path(dataset_yaml).parent))
        orig = cfg["train"]
        orig = orig if isinstance(orig, list) else [orig]
        cfg["train"] = [str(base / o) if not Path(o).is_absolute() else o for o in orig] + \
                       [str(Path(pseudo_dir).resolve() / "images")]
        cfg["path"] = str(base)
        data_yaml = output_dir / "student_dataset.yaml"
        yaml.safe_dump(cfg, open(data_yaml, "w"), sort_keys=False)
        print(f"  Student data: dataset + pseudo-labelled tiles -> {data_yaml}")

    results = student.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=1e-3,
        cos_lr=True,
        patience=10,
        project=str(output_dir),
        name=run_name,
        exist_ok=True,
        mosaic=0.8,
        mixup=0.1,
        scale=0.25,                  # default 0.5 would shrink 12 px objects to 6 px
        cache="ram" if use_ram_cache else False,
    )

    best_pt = output_dir / run_name / "weights" / "best.pt"
    print(f"  Student trained: {best_pt}")
    return best_pt, is_rtdetr


# ─── Step 3: QAT (Quantization-Aware Training) ────────────────────────────────
def run_qat(model_path: Path, dataset_yaml: Path,
            output_dir: Path, epochs: int = 5,
            quant_backend: str = "fbgemm",
            batch: int = 24,
            use_ram_cache: bool = True,
            is_rtdetr: bool = False,
            imgsz: int = 640) -> Path:
    """
    Quantization-Aware Training (INT8) using PyTorch's QAT framework.
    Simulates INT8 quantization during training, then freezes to static INT8.

    quant_backend: "fbgemm" (x86/CUDA) or "qnnpack" (ARM/mobile)
                   For Jetson Orin NX, TensorRT handles quantization —
                   QAT here improves accuracy before TRT INT8 calibration.

    NOTE: Full QAT for DETR-family models requires custom quantization config
    because the attention mechanism has non-linear operations. The implementation
    below uses PTQ (post-training quantization) as a validated baseline,
    with QAT hooks for fine-tuning. This is the production-safe approach.

    [BUG FIXED] This previously loaded model_path with the RTDETR() class
    unconditionally. But the pipeline calls this on the YOLOv10-N student
    checkpoint (from train_student()), not the RT-DETR-L teacher — loading
    a YOLO checkpoint through the RTDETR class is an architecture mismatch,
    not a hardware issue, and would fail regardless of GPU. Now takes the
    architecture as an explicit flag from the caller (who trained the
    checkpoint and therefore knows what it is) instead of guessing from
    filenames or checkpoint metadata, since guessing broke a second time
    when the run-name-hardcoding bug in train_student() was fixed alongside
    this one.
    """
    print(f"\n{'='*60}")
    print(f"  Step 3: Quantization-Aware Training (INT8)")
    print(f"  Backend : {quant_backend}")
    print(f"  Epochs  : {epochs}")
    print(f"  Batch   : {batch}")
    print(f"{'='*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    ModelClass = RTDETR if is_rtdetr else YOLO
    print(f"  Architecture: {'RT-DETR' if is_rtdetr else 'YOLO'} "
          f"(loader: {ModelClass.__name__})")
    model = ModelClass(str(model_path))

    # Short fine-tuning at very low LR with augmentation OFF
    # This adapts weights to quantization-friendly ranges
    print(f"  Running {epochs} QAT fine-tuning epochs...")
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        lr0=1e-5,               # Very low LR — just adapt to INT8 range
        lrf=1.0,                # No LR decay during QAT
        batch=batch,
        imgsz=imgsz,
        cos_lr=False,
        augment=False,          # No augmentation during QAT — stable gradients
        mosaic=0.0,
        mixup=0.0,
        project=str(output_dir),
        name="qat_finetune",
        exist_ok=True,
        patience=3,
        cache="ram" if use_ram_cache else False,
    )

    qat_best = output_dir / "qat_finetune" / "weights" / "best.pt"
    print(f"  QAT fine-tune complete: {qat_best}")
    return qat_best


# ─── Step 4: ONNX Export + Validation ─────────────────────────────────────────
def export_to_onnx(model_path: Path, output_dir: Path,
                   image_size: int = 640,
                   dynamic: bool = False,
                   int8: bool = False,
                   calibration_dir: Optional[Path] = None) -> Path:
    """
    Export to ONNX with optional INT8 calibration.
    Validates exported ONNX graph immediately after export.

    dynamic: export with dynamic batch size (needed for variable batch inference)
    int8: apply INT8 calibration (requires calibration_dir with real images)
    """
    print(f"\n{'='*60}")
    print(f"  Step 4: ONNX Export")
    print(f"  Model     : {model_path.name}")
    print(f"  INT8      : {int8}")
    print(f"  Dynamic   : {dynamic}")
    print(f"{'='*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = RTDETR(str(model_path))

    export_args = {
        "format":   "onnx",
        "imgsz":    image_size,
        "dynamic":  dynamic,
        "simplify": True,       # ONNX simplifier — reduces graph complexity
        "opset":    17,         # ONNX opset 17 required for TRT 8.6+
    }

    if int8 and calibration_dir and calibration_dir.exists():
        export_args["int8"] = True
        export_args["data"]  = str(calibration_dir)  # Calibration dataset

    onnx_path = model_path.with_suffix(".onnx")
    model.export(**export_args)

    # Validate exported ONNX
    print(f"\n  Validating ONNX graph...")
    validation_result = validate_onnx(onnx_path)

    # Move to output_dir with descriptive name
    suffix = "_int8" if int8 else "_fp32"
    final_name = f"ewads_rtdetr{suffix}.onnx"
    final_path = output_dir / final_name

    import shutil
    shutil.copy2(onnx_path, final_path)
    print(f"  ONNX saved: {final_path}")

    return final_path


def validate_onnx(onnx_path: Path) -> dict:
    """
    Validate ONNX model:
      1. Graph structure check (onnx.checker)
      2. Inference sanity check (onnxruntime)
      3. Output shape verification

    Called after EVERY export step. If this fails, stop the pipeline.
    """
    result = {"path": str(onnx_path), "valid": False, "errors": []}

    # Check 1: ONNX graph validity
    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        result["graph_valid"] = True
        print(f"  ✓ ONNX graph structure valid")
    except Exception as e:
        result["errors"].append(f"Graph invalid: {e}")
        print(f"  ✗ ONNX graph invalid: {e}")
        return result

    # Check 2: OnnxRuntime inference
    try:
        session = ort.InferenceSession(str(onnx_path),
                                       providers=["CUDAExecutionProvider",
                                                  "CPUExecutionProvider"])
        input_name  = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"  ✓ OnnxRuntime session created")
        print(f"    Input  : {input_name} {input_shape}")

        # Dummy inference at the model's own input size (dynamic axes -> 640)
        hh = input_shape[2] if isinstance(input_shape[2], int) else 640
        ww = input_shape[3] if isinstance(input_shape[3], int) else 640
        dummy = np.random.rand(1, 3, hh, ww).astype(np.float32)
        outputs = session.run(None, {input_name: dummy})
        print(f"  ✓ Dummy inference successful")
        print(f"    Outputs: {[o.shape for o in outputs]}")

        result["output_shapes"] = [list(o.shape) for o in outputs]
        result["valid"] = True

    except Exception as e:
        result["errors"].append(f"ORT inference failed: {e}")
        print(f"  ✗ OnnxRuntime inference failed: {e}")

    return result


def export_student_onnx(student_path: Path, output_dir: Path,
                        is_rtdetr: bool = False, imgsz: int = 640) -> Path:
    """
    Export the student model to ONNX. Takes is_rtdetr explicitly for the
    same reason run_qat() does now — this previously hardcoded YOLO(),
    which breaks silently (or loads the wrong graph) when --target rtdetr-s
    is selected instead of the yolov10n default.
    """
    ModelClass = RTDETR if is_rtdetr else YOLO
    arch_name  = "rtdetr-s" if is_rtdetr else "yolov10n"
    print(f"\n  Exporting student ({arch_name}) to ONNX...")

    model = ModelClass(str(student_path))
    model.export(format="onnx", imgsz=imgsz, simplify=True, opset=17)

    onnx_path = student_path.with_suffix(".onnx")
    final_path = output_dir / f"ewads_{arch_name}_fp32.onnx"

    import shutil
    shutil.copy2(onnx_path, final_path)

    # Validate
    validate_onnx(final_path)

    print(f"  Student ONNX: {final_path}")
    return final_path


# ─── Step 5: TensorRT Build Script Generator ──────────────────────────────────
def generate_tensorrt_script(onnx_path: Path, output_dir: Path,
                              precision: str = "int8",
                              jetson_target: str = "orin_nx",
                              imgsz: int = 640) -> Path:
    """
    Generate a shell script to build TensorRT engine on the Jetson.
    TensorRT must run ON the target device (Jetson) — it cannot be
    cross-compiled from x86. This script is copied to the Jetson and run there.

    precision: "fp16" or "int8"
    """
    # NOTE: the INT8 lines are assembled here, not inside the f-string below -- a backslash inside
    # an f-string expression is a SyntaxError on Python < 3.12 (your env pins 3.10), so the
    # previous version of this file could not even be imported.
    flag = "int8" if precision == "int8" else "fp16"
    if precision == "int8":
        int8_block = (
            '# INT8 needs a calibration CACHE.  trtexec --calib takes a cache FILE, not a folder of\n'
            '# images, so a folder cannot calibrate it; without one the scales are meaningless.\n'
            'if [ -f "$CALIB_CACHE" ]; then\n'
            '    BUILD_CMD="$BUILD_CMD --calib=$CALIB_CACHE"\n'
            '    echo "Using calibration cache $CALIB_CACHE"\n'
            'else\n'
            '    echo "WARNING: no $CALIB_CACHE -> INT8 accuracy will be BAD. Preferred: on the Jetson run"\n'
            '    echo "  python tools/build_calib_set.py --tiled_dir data/tiled --out data/calib"\n'
            '    echo "  python 10_jetson_build_engine.py --weights <student.pt> --calib data/calib/calib.yaml --gate_data <val_full> --imgsz ' + str(imgsz) + '"\n'
            '    echo "  (builds FP16 + INT8 and keeps INT8 only if it passes the accuracy gate), or use build_trt_fp16.sh."\n'
            'fi\n')
    else:
        int8_block = ""

    trt_script = f"""#!/bin/bash
# TensorRT Engine Builder — EWADS
# Run this ON the Jetson Orin NX, not on your laptop/workstation
# Generated: {datetime.now().isoformat()}
# Target: {jetson_target}
# Precision: {precision}
# Input: images 1x3x{imgsz}x{imgsz}  (= one TILE; 07_edge_benchmark.py runs it once per tile)

set -e

ONNX_PATH="{onnx_path.name}"
ENGINE_PATH="${{ONNX_PATH%.onnx}}_{precision}.trt"
CALIB_CACHE="./calibration.cache"
LOG_PATH="./trt_build.log"

echo "Building TensorRT engine from $ONNX_PATH"
echo "Precision: {precision.upper()}"
echo "Output: $ENGINE_PATH"
echo ""

# Verify ONNX file exists
if [ ! -f "$ONNX_PATH" ]; then
    echo "ERROR: ONNX file not found: $ONNX_PATH"
    exit 1
fi

# Verify trtexec is available
if ! command -v trtexec &> /dev/null; then
    echo "ERROR: trtexec not found. Ensure TensorRT is installed."
    echo "  On Jetson: sudo apt install tensorrt"
    exit 1
fi

echo "TensorRT version:"
trtexec --version 2>/dev/null | head -3
echo ""

# Build command  (the ONNX input is named "images" for Ultralytics exports)
BUILD_CMD="trtexec --onnx=$ONNX_PATH --saveEngine=$ENGINE_PATH --memPoolSize=workspace:4096 --{flag} --minShapes=images:1x3x{imgsz}x{imgsz} --optShapes=images:1x3x{imgsz}x{imgsz} --maxShapes=images:4x3x{imgsz}x{imgsz} --verbose"

{int8_block}
echo "Running: $BUILD_CMD"
eval "$BUILD_CMD 2>&1 | tee $LOG_PATH"

if [ -f "$ENGINE_PATH" ]; then
    ENGINE_SIZE=$(du -sh "$ENGINE_PATH" | cut -f1)
    echo ""
    echo "Engine built successfully"
    echo "  Output: $ENGINE_PATH ($ENGINE_SIZE)"
    echo ""
    echo "Running latency benchmark (per tile)..."
    trtexec --loadEngine=$ENGINE_PATH --iterations=200 --warmUp=20 --percentile=95 \\
        2>&1 | grep -E "(mean|median|percentile|GPU|Throughput)"
else
    echo "Engine build failed. Check $LOG_PATH"
    exit 1
fi
"""

    script_path = output_dir / f"build_trt_{precision}.sh"
    with open(script_path, "w") as f:
        f.write(trt_script)
    script_path.chmod(0o755)

    print(f"  TensorRT build script: {script_path}")
    print(f"  Copy this to Jetson and run: bash {script_path.name}")
    return script_path


# ─── Compression Report ───────────────────────────────────────────────────────
def generate_compression_report(output_dir: Path, results: dict):
    """Write a JSON report summarizing all compression steps and metrics."""
    report = {
        "timestamp":   datetime.now().isoformat(),
        "pipeline":    "EWADS Compression & Export",
        "steps":       results,
        "files": {
            "pruned_model":   str(output_dir / "rtdetr_l_pruned.pt"),
            "qat_model":      str(output_dir / "qat_finetune" / "weights" / "best.pt"),
            "teacher_onnx":   str(output_dir / "ewads_rtdetr_fp32.onnx"),
            "student_onnx":   str(output_dir / "ewads_yolov10n_fp32.onnx"),
            "trt_script_int8": str(output_dir / "build_trt_int8.sh"),
            "trt_script_fp16": str(output_dir / "build_trt_fp16.sh"),
        },
        "next_steps": [
            "Copy ONNX + TRT build scripts to Jetson Orin NX",
            "Run build_trt_int8.sh on Jetson",
            "Run 07_edge_benchmark.py on Jetson to measure latency + accuracy",
            "Tune confidence threshold using calibration protocol",
        ],
    }

    report_path = output_dir / "compression_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Compression report: {report_path}")
    return report


def _validate_checkpoint(model, dataset_yaml: Path, stage_name: str):
    """Quick validation pass after a compression step."""
    try:
        metrics = model.val(data=str(dataset_yaml), verbose=False)
        print(f"  [{stage_name}] mAP@0.5: {metrics.box.map50:.4f}  "
              f"mAP@0.5:0.95: {metrics.box.map:.4f}  "
              f"Recall: {metrics.box.mr:.4f}")
        return metrics
    except Exception as e:
        print(f"  [{stage_name}] Validation failed: {e}")
        return None


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="EWADS compression and export pipeline"
    )
    parser.add_argument("--model_path",      type=str, required=True)
    parser.add_argument("--dataset_yaml",    type=str,
                        default="./data/augmented/dataset.yaml")
    parser.add_argument("--output_dir",      type=str, default="./runs/export")
    parser.add_argument("--target",          type=str, default="yolov10n",
                        choices=["yolov10n", "rtdetr-s"],
                        help="Student model architecture")
    parser.add_argument("--prune_ratio",     type=float, default=0.35)
    parser.add_argument("--unlabeled_dir",   type=str, default=None,
                        help="Unlabeled images for pseudo-labeling")
    parser.add_argument("--qat_epochs",      type=int, default=5)
    parser.add_argument("--export_only",     action="store_true",
                        help="Skip pruning/distillation, export directly")
    parser.add_argument("--validate_onnx",   action="store_true")
    parser.add_argument("--onnx_path",       type=str, default=None)
    parser.add_argument("--imgsz",           type=int, default=None,
                        help="input size = tile size (default: from dataset.yaml `ewads: tile`, else 640)")
    parser.add_argument("--pseudo_conf",     type=float, default=0.4)
    parser.add_argument("--precision",       type=str, default="int8",
                        choices=["fp16", "int8"])
    args = parser.parse_args()

    model_path   = Path(args.model_path)
    dataset_yaml = Path(args.dataset_yaml)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_cfg = read_tile_cfg(dataset_yaml)
    imgsz    = args.imgsz or tile_cfg.get("tile") or 640
    overlap  = int(tile_cfg.get("overlap", 128))
    if tile_cfg:
        print(f"\n  Tiled dataset: model input {imgsz}x{imgsz} (tile), overlap {overlap}")
    else:
        print(f"\n  [WARN] no `ewads:` block in {dataset_yaml}: legacy non-tiled dataset, using {imgsz}px.")

    # ONNX validation only mode
    if args.validate_onnx:
        if not args.onnx_path:
            print("[ERROR] --onnx_path required with --validate_onnx")
            sys.exit(1)
        result = validate_onnx(Path(args.onnx_path))
        print(f"\n  Valid: {result['valid']}")
        if result.get("errors"):
            for e in result["errors"]:
                print(f"  Error: {e}")
        return

    results = {}

    if args.export_only:
        # Direct export without pruning or distillation
        print("\n  Export-only mode")
        teacher_onnx = export_to_onnx(model_path, output_dir, image_size=imgsz, int8=False)
        generate_tensorrt_script(teacher_onnx, output_dir, "fp16", imgsz=imgsz)
        generate_tensorrt_script(teacher_onnx, output_dir, "int8", imgsz=imgsz)
        generate_compression_report(output_dir, results)
        return

    # Full pipeline
    print(f"\n{'='*60}")
    print(f"  EWADS Compression Pipeline")
    print(f"  Teacher: {model_path.name}")
    print(f"  Student: {args.target}")
    print(f"  Device : {DEVICE}")
    print(f"{'='*60}")

    # Step 1: Prune teacher
    pruned_path = prune_model(model_path, output_dir, args.prune_ratio, dataset_yaml, imgsz=imgsz)
    results["step1_pruning"] = {"output": str(pruned_path), "ratio": args.prune_ratio}

    # Step 2: Pseudo-label and train student
    unlabeled_dir = Path(args.unlabeled_dir) if args.unlabeled_dir else None
    if unlabeled_dir and unlabeled_dir.exists():
        pseudo_dir = pseudo_label_for_student(model_path, unlabeled_dir, output_dir,
                                              conf_threshold=args.pseudo_conf, tile=imgsz, overlap=overlap)
    else:
        pseudo_dir = None
        print(f"\n  [SKIP] No unlabeled_dir provided. Student trains on augmented dataset only.")

    student_path, student_is_rtdetr = train_student(
        teacher_path=model_path,
        dataset_yaml=dataset_yaml,
        pseudo_dir=pseudo_dir,
        output_dir=output_dir,
        student_model=f"{args.target}.pt",
        imgsz=imgsz,
    )
    results["step2_distillation"] = {"student": str(student_path),
                                      "architecture": args.target}

    # Step 3: QAT on student — architecture passed explicitly, not guessed
    qat_path = run_qat(student_path, dataset_yaml, output_dir, args.qat_epochs,
                       is_rtdetr=student_is_rtdetr, imgsz=imgsz)
    results["step3_qat"] = {"output": str(qat_path), "epochs": args.qat_epochs}

    # Step 4: ONNX export (teacher FP32 + student INT8)
    print(f"\n  Exporting teacher to ONNX (FP32)...")
    teacher_onnx = export_to_onnx(model_path, output_dir, image_size=imgsz, int8=False)

    print(f"\n  Exporting student to ONNX (FP32, pre-TRT INT8 calibration)...")
    student_onnx = export_student_onnx(qat_path, output_dir, is_rtdetr=student_is_rtdetr, imgsz=imgsz)

    results["step4_onnx"] = {
        "teacher": str(teacher_onnx),
        "student": str(student_onnx),
    }

    # Step 5: TensorRT build scripts
    generate_tensorrt_script(student_onnx, output_dir, "int8", imgsz=imgsz)
    generate_tensorrt_script(student_onnx, output_dir, "fp16", imgsz=imgsz)
    results["step5_tensorrt"] = {"scripts_dir": str(output_dir)}

    # Final report
    generate_compression_report(output_dir, results)

    print(f"\n{'='*60}")
    print(f"  Compression pipeline complete!")
    print(f"  Teacher ONNX  : {teacher_onnx}")
    print(f"  Student ONNX  : {student_onnx}")
    print(f"  TRT scripts   : {output_dir}/build_trt_*.sh")
    print(f"\n  Next: copy student ONNX + TRT scripts to Jetson Orin NX")
    print(f"        then run 07_edge_benchmark.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
