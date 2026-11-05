# EWADS — E-Waste Aerial Detection System

> **Start with [HANDOFF.md](HANDOFF.md) → "Workstation runbook" (top of the file).**
> This README describes the original design. The pipeline has since changed
> (native-resolution tiling `03b`, evaluation `08`, mission planner `09`, Jetson
> engine build `10`, evidence pipeline `11`) and its Pipeline Map / Quick Start
> are out of date — the runbook is the current, ordered procedure. Scripts live
> in the project root (there is no `scripts/` folder).

## Pipeline README

---

## Revision Log

**v2 — Workstation upgrade (RTX 5090 32GB + 128GB DDR5)**
- Removed cloud GPU as the primary RT-DETR-L training path; workstation
  now handles scripts 05 and 06 entirely
- Raised training batch sizes and Stage 2-3 image resolution to use the
  new VRAM budget (see `05_train_rtdetr.py` header for exact numbers)
- Enabled RAM dataset caching throughout (128GB makes this free)
- Added `environment_laptop.yml` as a separate, non-interchangeable
  profile — the workstation's CUDA 12.8/Blackwell build will not run
  correctly on the laptop's Ampere GPU and vice versa
- **Fixed 4 real bugs found during this revision pass**, independent of
  the hardware change (see below) — these would have surfaced the first
  time the affected code paths actually ran, hardware upgrade or not

**Bugs fixed in this pass:**
| # | File | Bug | Impact |
|---|------|-----|--------|
| 1 | `05_train_rtdetr.py` | `Optional` used in 3 signatures, never imported from `typing` | Script crashed with `NameError` on import — could not run at all |
| 2 | `06_compress_export.py` | `run_qat()` loaded every checkpoint via `RTDETR()` regardless of actual architecture | Would fail or silently misload when called on the YOLOv10-N student |
| 3 | `06_compress_export.py` | `train_student()` hardcoded output run name to `student_yolov10n` even when `--target rtdetr-s` was selected | Misleading run directory; broke architecture auto-detection downstream |
| 4 | `06_compress_export.py` | `export_student_onnx()` hardcoded `YOLO()` the same way #2 hardcoded `RTDETR()` | Same failure mode as #2, different call site |

Bug #1 would have blocked you immediately. Bugs #2-4 would only have
surfaced once you ran the full `06_compress_export.py` pipeline without
`--export_only` — i.e., exactly when you got to the nano prototype step.
Worth internalizing: a script that runs cleanly through `py_compile` or
even partway through execution is not the same as a script that's
correct — annotation-only `NameError`s and architecture-mismatch bugs
like these don't show up until the specific code path executes.

---

## What this is

A production-grade aerial e-waste object detection pipeline for legal enforcement
and environmental forensics. Detects 18 classes of electronic waste from drone
imagery (DJI Phantom 4 Pro / Pixhawk). Outputs GPS-tagged, timestamped, hash-chained
detection logs admissible as forensic evidence.

**Target:** mAP@0.5:0.95 ≥ 0.68 | Recall ≥ 0.87 | FP rate < 0.05

---

## Hardware Routing (updated — workstation upgraded to RTX 5090 32GB + 128GB RAM)

Cloud GPU rental is no longer required for this project. Everything runs on
two local machines now. The laptop's role shrank to data-pipeline work only;
the workstation absorbed everything that used to need a cloud A100.

| Task                              | Machine                      | Why                                    |
|------------------------------------|-------------------------------|-----------------------------------------|
| BlenderProc synthetic generation  | Laptop (RTX 3060 8GB)         | I/O and CPU-bound, not GPU-bound        |
| Objaverse model fetching          | Laptop                        | Network-bound, not compute-bound        |
| AdaIN training + inference        | Laptop                        | ~5.5GB VRAM at batch=4, fits easily     |
| Albumentations augmentation       | Laptop                        | CPU-bound                               |
| RT-DETR-L 3-stage training        | **Workstation (RTX 5090)**    | Fits batch=20 @ 768px in 32GB now       |
| YOLOv10-N student + pseudo-label  | **Workstation**               | Faster iteration with RAM-cached data   |
| QAT + pruning                     | **Workstation**               | Same GPU as training, no reload cost    |
| ONNX export                       | **Workstation**               | No VRAM needed but co-located with rest |
| TRT engine build (script 06 output)| **Jetson Orin NX**            | Must build on target device — unchanged |
| Edge benchmark (script 07)        | **Jetson Orin NX**            | Real deployment validation — unchanged  |

**Before your first workstation run**, verify the RTX 5090 is actually
recognized by PyTorch — Blackwell (sm_120) needs CUDA 12.8+ and PyTorch
2.7+, and a stale install can report `cuda.is_available() == True` while
every real kernel call fails:

```bash
conda env create -f environment.yml   # workstation profile
conda activate ewads
python -c "
import torch
print('torch:', torch.__version__)
print('capability:', torch.cuda.get_device_capability(0))
x = torch.randn(1024,1024,device='cuda') @ torch.randn(1024,1024,device='cuda')
torch.cuda.synchronize()
print('kernel launch OK')
"
```

The laptop uses a **separate** environment file (`environment_laptop.yml`,
different conda env name `ewads-laptop`) — don't install the workstation's
CUDA 12.8 toolkit on the laptop, and don't try to run the workstation's
`environment.yml` there expecting it to "just work at reduced speed." The
laptop's older CUDA build simply has no Blackwell kernels; it isn't a
slower fallback, it's a different target.

```bash
# Laptop
conda env create -f environment_laptop.yml
conda activate ewads-laptop

# Workstation
conda env create -f environment.yml
conda activate ewads
```

**Sanity-check before committing a full run.** Before you ship your
augmented dataset over to the workstation for a multi-hour RT-DETR-L run,
prove the dataset.yaml and label files are structurally sound with a
2-epoch RT-DETR-S run on the laptop:

```bash
python 05_train_rtdetr.py --sanity_check \
  --dataset_yaml ./data/augmented/dataset.yaml
```

If that fails, you just saved yourself from discovering a broken label
path three hours into a workstation run.

---

## Pipeline Map

```
PHASE 0: Environment
  └── conda env create -f environment.yml

PHASE 1: 3D Model Acquisition
  ├── 02b_objaverse_fetch.py      ← Download from Objaverse (free, scriptable)
  └── blenderproc run _generate_piles.py  ← Procedural ewaste_pile + cable_bundle

PHASE 2: Synthetic Data Generation
  └── blenderproc run 02_blenderproc_synthetic_v2.py
        Inputs:  assets/models/, assets/textures/, assets/haven/
        Outputs: data/raw/synthetic/ (images + labels + scene_index.jsonl)

PHASE 3: Style Transfer Training
  └── python 04_adain_transfer.py train
        Inputs:  data/raw/synthetic/, data/raw/real/images/
        Outputs: models/adain/adain_best.pth

PHASE 4: Style Transfer Application
  └── python 04_adain_transfer.py transfer
        Inputs:  data/raw/synthetic/, models/adain/adain_best.pth
        Outputs: data/raw/synthetic_styled/

PHASE 5: Augmentation
  └── python 03_augmentation_pipeline.py
        Inputs:  data/raw/synthetic_styled/, data/raw/real/, data/raw/val/
        Outputs: data/augmented/ (train + val + dataset.yaml)

PHASE 6: Training (cloud/workstation)
  └── python 05_train_rtdetr.py --stage all
        Inputs:  data/augmented/dataset.yaml
        Outputs: runs/ewads/Stage3_HardFinetune/weights/best.pt

PHASE 7: Compression & Export (laptop)
  └── python 06_compress_export.py
        Inputs:  runs/ewads/.../best.pt
        Outputs: runs/export/ (ONNX + TRT build scripts)

PHASE 8: Jetson Deployment
  ├── Copy ONNX + scripts to Jetson
  ├── bash build_trt_int8.sh      ← Run ON Jetson
  └── python 07_edge_benchmark.py --all
        Outputs: benchmark_results/ (latency + accuracy + calibrated threshold)

PHASE 9: Forensic Inference (production)
  └── python 07_edge_benchmark.py --forensic
        Outputs: forensic_output/ (visualizations + audit.jsonl)
```

---

## Quick Start (laptop → workstation, full pipeline)

With the RTX 5090 in play, there's no reason to route around RT-DETR-L
anymore — the "train YOLOv10-N only because RT-DETR-L doesn't fit"
workaround from the laptop-only era is gone. This is the real pipeline,
split across both machines by what each is good at.

```bash
# ══ ON THE LAPTOP ══════════════════════════════════════════════════════
conda env create -f environment_laptop.yml
conda activate ewads-laptop

# 1. Fetch 3D models
python 02b_objaverse_fetch.py --n_per_class 15
python 02b_objaverse_fetch.py --gen_piles --n_piles 30

# 2. Validate BlenderProc pipeline (dry run — no rendering)
blenderproc run 02_blenderproc_synthetic_v2.py \
  --n_scenes 5 --dry_run

# 3. Generate synthetic scenes — laptop handles this fine, it's CPU/IO-bound
blenderproc run 02_blenderproc_synthetic_v2.py \
  --n_scenes 5000 --output_dir ./data/raw/synthetic

# 4. Train AdaIN on your real captures (~2-3 hrs, ~5.5GB VRAM)
python 04_adain_transfer.py train \
  --content_dir ./data/raw/synthetic \
  --style_dir ./data/raw/real/images \
  --epochs 20 --batch_size 4

# 5. Apply style transfer
python 04_adain_transfer.py transfer \
  --model_path ./models/adain/adain_best.pth \
  --input_dir ./data/raw/synthetic \
  --style_dir ./data/raw/real/images \
  --output_dir ./data/raw/synthetic_styled

# 6. Augment
python 03_augmentation_pipeline.py \
  --n_aug_per_image 3 --add_mosaic

# 7. Sanity-check the dataset BEFORE shipping it to the workstation —
#    this is a 2-epoch RT-DETR-S run, takes minutes, catches broken
#    label paths and malformed dataset.yaml before you waste workstation time
python 05_train_rtdetr.py --sanity_check \
  --dataset_yaml ./data/augmented/dataset.yaml

# 8. Ship the augmented dataset to the workstation
rsync -avz ./data/augmented/ workstation:/path/to/ewads/data/augmented/


# ══ ON THE WORKSTATION (RTX 5090) ══════════════════════════════════════
conda env create -f environment.yml
conda activate ewads

# Verify Blackwell support BEFORE the first real run (see Hardware Routing
# section above) — this is not optional, a silent CPU fallback here means
# you find out 8 hours later that "training" ran on the CPU.

# 9. Full RT-DETR-L 3-stage training — batch=20, imgsz up to 768,
#    dataset cached in RAM. Expect single-digit hours total, not days.
python 05_train_rtdetr.py \
  --dataset_yaml ./data/augmented/dataset.yaml \
  --output_dir ./runs/ewads \
  --stage all

# 10. Full compression pipeline: prune teacher, pseudo-label + train
#     YOLOv10-N student, QAT, export both to ONNX, generate TRT scripts
python 06_compress_export.py \
  --model_path ./runs/ewads/Stage3_HardFinetune/weights/best.pt \
  --dataset_yaml ./data/augmented/dataset.yaml \
  --output_dir ./runs/export \
  --target yolov10n \
  --prune_ratio 0.35


# ══ ON THE JETSON ORIN NX ══════════════════════════════════════════════
# 11. Copy student ONNX + build script over, then build the TRT engine
#     ON the Jetson — TensorRT engines are not portable across devices
scp ./runs/export/ewads_yolov10n_fp32.onnx jetson:/opt/ewads/models/
scp ./runs/export/build_trt_int8.sh jetson:/opt/ewads/
ssh jetson "cd /opt/ewads && bash build_trt_int8.sh"

# 12. Benchmark + calibrate on real hardware
ssh jetson "cd /opt/ewads && python 07_edge_benchmark.py \
  --mode trt --engine_path ./models/ewads_yolov10n_int8.trt \
  --test_dir ./data/test --all"
```

If you want a fast sanity-tested prototype before the full 60-epoch
3-stage run finishes, you can still stop after Stage 1 (`--stage 1`,
~10 epochs) and run it through compression — it won't hit your mAP
target, but it validates the entire pipeline end-to-end, including the
Jetson deployment step, while Stage 2/3 keep training.

---

## Script Reference

| Script | Purpose | Runs On |
|--------|---------|---------|
| `02_blenderproc_synthetic_v2.py` | Synthetic scene generation | Laptop |
| `02b_objaverse_fetch.py` | 3D model acquisition from Objaverse | Laptop |
| `03_augmentation_pipeline.py` | Albumentations augmentation | Laptop |
| `04_adain_transfer.py` | Synthetic-to-real style transfer | Laptop |
| `05_train_rtdetr.py` | RT-DETR-L 3-stage training | **Workstation** |
| `06_compress_export.py` | Pruning + KD + QAT + ONNX export | **Workstation** |
| `07_edge_benchmark.py` | Latency + accuracy + forensic pipeline | Jetson |

---

## Key Decisions (from architectural audit — hardware-independent unless noted)

1. **RT-DETR-L with HGNetV2-L backbone** — not EfficientViT-M5
   (the blocker is engineering integration time, not compute; the 5090
   doesn't change this — see script 05's docstring)

2. **AdaIN over CycleGAN** — 75% of the quality gain, fraction of the training time

3. **Pseudo-labeling for KD** — not logit-level distillation across architectures
   (RT-DETR-L labels data → YOLoV10-N trains on those labels)

4. **Stage 1 uses mixed data from start** — prevents synthetic overfit before fine-tune

5. **ONNX validation after every export step** — mandatory, not optional

6. **Confidence threshold tuned for recall, not mAP** — forensic requirement

7. **Per-class AP targets** — aggregate mAP hides class-level failures in forensics

8. **Pruning ratio is tied to the Jetson's latency budget, not the training
   GPU.** The RTX 5090 upgrade makes training faster; it does not make the
   Jetson Orin NX faster. Don't let workstation headroom tempt you into
   skipping or under-doing the pruning step — the drone runs on the same
   hardware constraint it always did. Set `--prune_ratio` by measuring
   latency in `07_edge_benchmark.py` and working backward, not by feel.

9. **Training resolution raised 640→768 for Stages 2-3** (workstation only).
   This is a genuine hardware-enabled improvement, not just "more compute
   because we can" — your smallest classes (cpu_chip, hard_drive) are the
   ones most likely to be under-resolved at 640px from altitude, and 768px
   fits the 5090's VRAM budget with room to spare. Validate this actually
   helped by comparing per-class AP for those two classes specifically
   between a 640px and 768px run before assuming it's a free win.

---

## Data Directory Structure

```
data/
  raw/
    synthetic/          ← BlenderProc output (images + labels)
    synthetic_styled/   ← After AdaIN style transfer
    real/               ← Real drone captures (images + labels from CVAT)
    val/                ← Validation set (real, labeled, set aside BEFORE training)
    test/               ← HELD-OUT TEST SET (never touched until final eval)
  augmented/
    train/              ← Mixed augmented training set
    val/                ← Resized validation set
    dataset.yaml        ← Ultralytics dataset config

assets/
  models/               ← 3D models per class (from Objaverse)
  textures/             ← CC textures (blenderproc download cc_textures)
  haven/                ← HDRIs (blenderproc download haven)

models/
  adain/                ← AdaIN checkpoints

runs/
  ewads/                ← RT-DETR-L training runs
  export/               ← ONNX + TRT build scripts
  prototype/            ← YOLOv10-N prototype runs

benchmark_results/
  forensic_output/      ← Visualizations + audit log
  benchmark_report.json
```

---

## Annotation Workflow (CVAT)

1. Capture 200-500 real drone images at operational altitudes (30m, 60m, 90m)
2. Upload to CVAT project with the 18-class taxonomy
3. Two annotators independently label — adjudicate disagreements
4. Export as YOLO format
5. Place in `data/raw/real/` (train split) and `data/raw/val/` (val split)
6. **Set aside 100-200 images as test set BEFORE any training** → `data/test/`

The test set must be geographically diverse. If your training data is all from
one location, your test set must include at least 2-3 different locations.

---

## Forensic Output Format

Every detection event in `forensic_audit.jsonl` (fields updated 2026-10-08):
```json
{
  "record_index": 0,
  "timestamp_utc": "2025-09-01T10:23:45.123456+00:00",
  "capture_time_utc": "2025-09-01T06:20:30.250000+00:00",
  "capture_time_local": "2025-09-01T10:20:30.250000",
  "capture_time_source": "exif DateTimeOriginal+OffsetTimeOriginal",
  "model_version": "ewads-v1.0.0|tile768|overlap128|levelsall",
  "model_sha256": "5c1e...",
  "image_sha256": "a3f9...",
  "gps": {"latitude": 25.123456, "longitude": 55.654321, "altitude_m": 60.0},
  "conf_threshold": 0.312,
  "detections": [
    {"class_name": "battery", "confidence": 0.874, "bbox_xyxy": [...], "tier": 1}
  ],
  "tier1_detected": true,
  "prev_hash": "7d2a...",
  "record_hash": "f8c1..."
}
```

- `timestamp_utc` is the **processing time** (when the record was written), not when the image was taken.
- `capture_time_utc` is when the pixels were captured. It is only filled when the source carries a UTC
  offset (EXIF `OffsetTimeOriginal`, an XMP date with a zone) or the operator asserted the camera
  clock's offset (`--capture_utc_offset`, stated in `capture_time_source`); otherwise it is `null` and the
  camera-local time is kept in `capture_time_local`. No source at all: `capture_time_source = "unavailable"`.
- `model_sha256` is the SHA-256 of the engine / ONNX / weights file actually loaded (11: list of the
  ensemble members).

Chain can be verified at any time:
```bash
python 07_edge_benchmark.py --verify_chain \
  --audit_log ./benchmark_results/forensic_audit.jsonl
```
Opening an existing log to append re-verifies the whole chain first and refuses a broken one.
The chain alone cannot reveal tail truncation or a full rebuild; sign its head and verify against a
trusted key fingerprint (`tools/sign_chain_head.py`, see HANDOFF §15).

---

## Idea Backlog

- [ ] Wire bundle counting: SAM2 + counting head (secondary model, post-v1)
- [ ] DJI SDK live telemetry integration (replaces SRT file parsing)
- [ ] Altitude-adaptive confidence thresholds (per altitude band)
- [ ] ConvNeXt-V2-Base backbone swap (v2 research question)
- [ ] AFPN neck (relevant once backbone is CNN-based)
- [ ] Multi-temporal change detection (identify new dump sites between flights)
