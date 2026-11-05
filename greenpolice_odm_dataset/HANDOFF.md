# EWADS / ATHAR — Handoff

**Since 2026-10-08 the project is SYNTHETIC-ONLY for training.** Follow the
**Synthetic-only runbook** directly below. The older "Workstation runbook"
after it is kept for its setup phases (0–3, 9) and for reference; its real-data
training/validation steps (Phases 5–8) are superseded. Sections 1–13 are the
history and reasoning.

---

# Synthetic-only runbook (ATHAR, 2026-10-08)

**Regime.** No real labelled image is used for training, calibration or model
selection. Real imagery appears only as (a) unlabelled ground-texture **style
photos** for the AdaIN arm and (b) the **real test set** (staged scenes from
measured heights 8–25 m) plus **1 m reference-target photos** (H2).
Every selection step uses synthetic **validation**: `main/val` (and
`main/val_seq` for `11`). `08 --calibrate`, `10 --gate_data` and `11` (when it
derives stage thresholds) **refuse** any `*heldout*` / `test*` folder.

Run from the project root in the `ewads` env. **Check** lines must pass before
going on. Setup first: old runbook Phases 0–1 (transfer, env, tests) and 3
(assets + review round 3).

## S1 — Targets (before anything is trained)
Review `targets_v2.json`, then **[you]** say "freeze" (or run
`python tools/freeze_targets.py --targets targets_v2.json --by "<name>"`).
**Check:** `python tools/freeze_targets.py --targets targets_v2.json --check` → FROZEN.

## S2 — Render
```bash
# main set: all environments, all altitudes (+ low-altitude top-up), single frames
blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 5000 --environment all --seed 1
blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 1000 --environment all \
    --altitude_mode low_evidence --start_id 5000 --seed 2
# validation SEQUENCES for 11 (overlapping frames, spacing = 11's scheduler); distinct ids
blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 150 --environment all \
    --altitude_mode low_evidence --sequence_frames 5 --start_id 900000 --seed 3 \
    --output_dir ./data/raw/synthetic_valseq
# held-out SEQUENCES for scoring 11 on the evidence targets (final evaluation only); distinct ids
blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 150 --environment all \
    --altitude_mode low_evidence --sequence_frames 5 --start_id 950000 --seed 4 \
    --output_dir ./data/raw/synthetic_heldoutseq
```
**Check:** no class with 0 labels in the printed table; `synthetic_valseq/*/images` and
`synthetic_heldoutseq/*/images` hold `*_f00…f04.jpg`.

## S3 — Splits (by scene, stratified; LOEO folds)
```bash
python tools/make_synth_splits.py --synthetic_dir data/raw/synthetic \
    --val_seq_dir data/raw/synthetic_valseq --heldout_seq_dir data/raw/synthetic_heldoutseq --out_dir data/splits
```
Creates `data/splits/main/{synthetic,val,heldout,val_seq,heldout_seq}` and
`data/splits/loeo_<env>/{synthetic,val,heldout}` plus `split_manifest.json`
(per-file SHA-256). **Never** use `*/heldout` for any choice.
**Check:** the tool prints no LEAKAGE error; every split non-empty.

## S4 — Tile, AdaIN arm, augment (main)
```bash
python 03b_tile_dataset.py --input_dir data/splits/main --output_dir data/tiled/main --splits synthetic val --dry_run
python 03b_tile_dataset.py --input_dir data/splits/main --output_dir data/tiled/main --splits synthetic val
# arm A: no AdaIN
python 03_augmentation_pipeline.py --input_dir data/tiled/main --output_dir data/augmented/main --rfs_t 0.1
# arm B: AdaIN with real ground-texture style photos (unlabelled; never test-site photos)
python tools/check_style_photos.py --photos <style photo folder> --exclude_dir data/test/images --copy_usable data/style_photos
python 04_adain_transfer.py train --content_dir data/tiled/main/synthetic --style_dir data/style_photos \
    --output_dir models/adain --epochs 20 --batch_size 16
python 04_adain_transfer.py transfer --model_path models/adain/adain_best.pth \
    --input_dir data/tiled/main/synthetic --style_dir data/style_photos --output_dir data/tiled_styled/main/synthetic
cp -r data/tiled/main/val data/tiled/main/tiling_manifest.json data/tiled_styled/main/
python 03_augmentation_pipeline.py --input_dir data/tiled_styled/main --output_dir data/augmented/main_adain --rfs_t 0.1
```
**Check:** both `dataset.yaml` files show `sources: ["synthetic"]`.

## S5 — Train (synthetic-only is the default mode of 05)
```bash
python 05_train_rtdetr.py --sanity_check --dataset_yaml data/augmented/main/dataset.yaml
python 05_train_rtdetr.py --dataset_yaml data/augmented/main/dataset.yaml       --output_dir runs/A_noadain --seed 0
python 05_train_rtdetr.py --dataset_yaml data/augmented/main_adain/dataset.yaml --output_dir runs/A_adain   --seed 0
```
Stages: frozen backbone → full fine-tune → short polish (`--no_polish` skips).
Each run writes `final_model.txt` and `training_provenance.json` (base-weights
SHA-256, per-stage init SHA-256, dataset hash, seed, versions).
**Check:** the log says `REGIME: SYNTHETIC-ONLY` and lists `TRAIN sources: ['synthetic']`.

## S6 — Choose the AdaIN arm and the threshold (synthetic validation only)
```bash
for arm in A_noadain A_adain; do
  python 08_tiled_eval.py --weights $(cat runs/$arm/final_model.txt) --data_dir data/splits/main/val \
      --meta data/splits/main/val/scene_index.jsonl --calibrate --targets "" --out_dir eval/val_$arm
done
```
Keep the arm with the higher `map50` in `eval/val_<arm>/eval_report.json` (this
comparison is the AdaIN decision record — keep both reports). Its calibrated
`conf` is THE operating threshold for every later step. Then the ensemble's
second member on the winning dataset:
```bash
python 05_train_rtdetr.py --dataset_yaml <winning dataset.yaml> --output_dir runs/B --base_model rtdetr-x.pt --seed 1
```

## S7 — LOEO folds (one per environment)
```bash
for env in desert european_urban european_rural; do
  python 03b_tile_dataset.py --input_dir data/splits/loeo_$env --output_dir data/tiled/loeo_$env --splits synthetic val
  python 03_augmentation_pipeline.py --input_dir data/tiled/loeo_$env --output_dir data/augmented/loeo_$env --rfs_t 0.1
  python 05_train_rtdetr.py --dataset_yaml data/augmented/loeo_$env/dataset.yaml --output_dir runs/loeo_$env --seed 0
  python 08_tiled_eval.py --weights $(cat runs/loeo_$env/final_model.txt) --data_dir data/splits/loeo_$env/val \
      --calibrate --targets "" --out_dir eval/loeo_${env}_val          # fold threshold, on the fold's VAL
done
```
(If the AdaIN arm won in S6, run the fold through 04 the same way before 03.)
Scoring each fold on its held-out environment is part of final evaluation (Task 8).

## S8 — Evidence-stage thresholds (11) on synthetic validation sequences
```bash
python 11_evidence_pipeline.py --weights $(cat runs/A_<winner>/final_model.txt) $(cat runs/B/final_model.txt) \
    --data_dir data/splits/main/val_seq --tta flips --out_dir abl/val_seq
```
`abl/val_seq/<run>/stage_thresholds.json` is passed with `--stage_thresholds` to
every later 11 run (held-out synthetic sequences and real data).

## S9 — Jetson (INT8 gate on synthetic validation)
Old runbook Phase 9, with `--gate_data data/splits/main/val`
(`tools/build_calib_set.py --tiled_dir data/tiled/main` for the calibration set).

## S10 — Final evaluation: only on the explicit "run final evaluation"
Synthetic held-out (`main/heldout`, `--targets_dataset synthetic_heldout`), LOEO
held-out environments (`loeo_<env>/heldout`, `--targets_dataset sim_to_sim_loeo`),
real test set (`--targets_dataset sim_to_real`), all with `--conf` from S6 / the
fold's val. Not before; see the Task 8 instructions.

---

# Workstation runbook (2026-10-05; Phases 5–8 superseded by the synthetic-only runbook)

Conventions: run everything from the project root, in the `ewads` env, unless a
step says **[Jetson]** or **[you]** (needs your hands, not the computer).
`blenderproc run …` steps start Blender; everything else is plain `python`.
Each phase ends with a **check** — don't start the next phase until it passes.

## Phase 0 — Transfer and verify

1. Copy the whole `greenpolice_odm_dataset/` folder (or unzip
   `ewads_workstation_handoff.zip`) to the workstation.
2. Verify nothing was corrupted in transit (Git Bash / Linux):
   ```bash
   sha256sum -c MANIFEST.sha256 --quiet && echo "all files intact"
   ```
   **Check:** prints `all files intact`. Any `FAILED` line = re-copy that file.

## Phase 1 — Environment

```bash
nvidia-smi                       # driver must be >= 570.x (CUDA 12.8)
conda env create -f environment.yml
conda activate ewads
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
conda deactivate && conda activate ewads    # picks up PYTHONUTF8=1 from environment.yml
```
The `uninstall` is not optional: `ultralytics` pulls in a PyPI torch while the
env is created (CPU-only on Windows), and the cu128 install would otherwise
report "Requirement already satisfied" and keep it.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0)); x=torch.randn(1024,1024,device='cuda')@torch.randn(1024,1024,device='cuda'); torch.cuda.synchronize(); print('kernel launch OK')"
python -c "import os; print('PYTHONUTF8 =', os.environ.get('PYTHONUTF8'))"
```
**Check:** torch ≥ 2.7, capability `(12, 0)`, `kernel launch OK`, `PYTHONUTF8 = 1`.

Run the test suite (CPU logic checks, ~5 min):
```bash
for t in test_survey test_hier_eval test_metrics_vs_coco test_tiled_pipeline test_rotation_boxes test_onnx_parity test_trt_backend test_fusion; do python tests/$t.py > /dev/null 2>&1 && echo "PASS $t" || echo "FAIL $t"; done
```
**Check:** 8 × `PASS`. (Checked 2026-10-05 on Windows: all pass except
`test_metrics_vs_coco`, which only needs `pycocotools`, now in `environment.yml`.)
On a `FAIL`, run that test alone to see its traceback.
(`test_mosaic_scale` needs a tiled dataset — run it after Phase 5.)

## Phase 2 — Decide and freeze the targets **[you]** — before any training

1. Open `targets_v1.json` and read every number and its `why`. Change what you
   disagree with.
2. Decide whether `lighting` (fluorescent tubes / CFLs — mercury) is tier-1.
   Tier-1 is defined **once**, in `ewads_taxonomy.py` → `TIER_1`; 07, 08, 11 and
   target E3 all read it from there.
3. Freeze:
   ```bash
   python tools/freeze_targets.py --targets targets_v1.json --by "<your name>"
   ```
   Put the printed hash somewhere you cannot silently edit (a git commit, an
   email to yourself or your supervisor). The freeze also hashes
   `ewads_taxonomy.py`, so changing tier-1 later is detected.

**Check:** `python tools/freeze_targets.py --check` prints FROZEN.

## Phase 3 — Synthetic assets

```bash
rm -rf data/raw/synthetic        # the 162 old frames were made by the pre-v3 renderer
python tools/fetch_ground_textures.py --list_only && python tools/fetch_ground_textures.py
python tools/fetch_hdris.py --list_only && python tools/fetch_hdris.py
blenderproc run tools/generate_procedural_models.py --n 40     # 6 thin classes, 40 each
python tools/fetch_distractors.py --per_category 6             # optional real clutter models
```

Review round 3 (new procedural models + distractors). Same process as rounds 1–2:
```bash
blenderproc run tools/render_model_thumbnails.py --out_dir assets/models/_review_v3 \
    --skip_reviewed assets/models/_review_v2/index.json
python tools/make_review_sheets.py --thumbs_dir assets/models/_review_v3
```
Look at every `assets/models/_review_v3/sheet_<class>.jpg`, write the decisions
to `tools/reviews/round3.json` (copy the format of `round2_current.json`; set
`"index": "assets/models/_review_v3/index.json"`), then:
```bash
python tools/apply_model_review.py tools/reviews/round3.json           # dry run: read every line
python tools/apply_model_review.py tools/reviews/round3.json --apply
rm -rf assets/models/ewaste_pile     # generate_piles.py skips files that exist -- old piles lack the new models
blenderproc run tools/generate_piles.py --kind ewaste_pile --n 30      # rebuild piles from reviewed models
```
**Check:** every class folder in `assets/models/` has ≥ 10 models; the piles
sheet (`render_model_thumbnails.py` on `ewaste_pile`) shows heaps, not specks.

## Phase 4 — Render

1. Test 3 scenes per environment first and **look at them**:
   ```bash
   for e in desert european_urban european_rural; do
     blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 3 --environment $e --output_dir ./scratch/render_test
   done
   ```
   **Check:** the log says `Device NVIDIA GeForce RTX 5090 of type OPTIX found and used`
   (if it falls back to CPU, see section 3); images show textured ground,
   clutter, varied sky; `labels/*.txt` exist; no `✗` lines. Then `rm -rf scratch`.
2. Full run + low-altitude top-up (writes `data/raw/synthetic/<environment>/`):
   ```bash
   blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 5000 --environment all
   blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 1000 --environment all \
       --altitude_mode low_evidence --start_id 5000
   ```
   **Check:** the class table printed at the end (also
   `generation_stats_<start_id>.json`) — no class with 0 labels.

## Phase 5 — Real drone data **[you]** — the critical path; Phases 3–4 can run meanwhile

**Fly.** At least 2–3 *different sites* (the test set must come from sites
training never saw). Plan altitudes with:
```bash
python 09_mission_planner.py --class_table
python 09_mission_planner.py --legs 60:10 25:8 8:3
```
Fly survey (~60 m), confirmation (15–25 m) and evidence (5–8 m) legs. Record
**4K video with the `.SRT` telemetry file switched on**, plus stills. Each
evidence pass should be a *sequence* of overlapping frames (11 fuses views).
Also photograph a 1 m target from a known height once (checks 09's GSD).

**Label** in CVAT with exactly this class order (the ids are baked into the code):
```
0 crt_monitor  1 flat_screen  2 circuit_board  3 battery  4 cable_bundle  5 single_cable
6 mobile_phone  7 keyboard  8 printer  9 small_appliance  10 large_appliance  11 cpu_chip
12 hard_drive  13 power_supply  14 ewaste_pile  15 plastic_casing  16 metal_component  17 lighting
```
Export as YOLO 1.1. A frame with no e-waste gets an **empty** `.txt` (true negative).

**Split by site/flight, never by frame** (neighbouring frames are near-duplicates —
splitting by frame leaks test into train). Decide the split *before* training
and never move frames between splits afterwards:

| Folder | What | Used by |
|---|---|---|
| `data/raw/real/{images,labels}` | training sites | 03b → training |
| `data/raw/val/{images,labels}` | 1 site, full-res, sequences | 03b (training-time val); also pass the **same folder** as `val_full`/`val_seq` below |
| `data/test/{images,labels}` | held-out sites, full-res, sequences | 08 / 11 final scores only |
| `data/style_photos/` | unlabelled real frames (no test sites) | 04 AdaIN |

The handoff text (sections 10–12) names separate `val_full` / `val_seq` /
`test_seq` folders. With a few hundred real frames, splitting val three ways
leaves each part too small to trust, so reuse the val sequences for all three
roles and keep `test` as sequences too. Below, `data/raw/val` stands for
`val_full` and `val_seq`, and `data/test` for `test` and `test_seq`.

Altitude for each val/test frame (08 `--meta`): a CSV `data/test/altitudes.csv`
with columns `stem,altitude_m` (and the same for val). Take the altitude from
the DJI XMP `RelativeAltitude` (above take-off), **never** EXIF `GPSAltitude`:
```bash
python -c "import csv,sys,pathlib,ewads_survey as s; d=pathlib.Path(sys.argv[1]); w=csv.writer(open(d/'altitudes.csv','w',newline='')); w.writerow(['stem','altitude_m']); [w.writerow([p.stem, s.read_dji_xmp(p).get('rel_alt_m')]) for p in sorted((d/'images').iterdir()) if p.suffix.lower() in ('.jpg','.jpeg')]" data/test
```
(check the CSV has no empty altitudes; on sloping sites measure AGL yourself).

Vet the style photos:
```bash
python tools/check_style_photos.py --photos D:/flights --exclude_dir data/test/images --copy_usable data/style_photos
```

## Phase 6 — Tile, style, augment

```bash
python 03b_tile_dataset.py --input_dir data/raw --output_dir data/tiled --dry_run   # disk estimate + per-class table
python 03b_tile_dataset.py --input_dir data/raw --output_dir data/tiled
```
**Check:** in the per-class table `whole>=90%` is ~100% for every class; the
disk estimate fits; synthetic shows `reading 3 sub-folders`.

AdaIN must *earn* its place, so build both datasets and train both (Phase 7):
```bash
# (a) without AdaIN
python 03_augmentation_pipeline.py --input_dir data/tiled --output_dir data/augmented --rfs_t 0.1
# (b) with AdaIN: style the synthetic tiles, then assemble a folder 03 can read
python 04_adain_transfer.py train --content_dir data/tiled/synthetic/images --style_dir data/style_photos \
    --output_dir models/adain --epochs 20 --batch_size 16
python 04_adain_transfer.py transfer --model_path models/adain/adain_best.pth \
    --input_dir data/tiled/synthetic --style_dir data/style_photos --output_dir data/tiled_styled/synthetic
cp -r data/tiled/real data/tiled/val data/tiled/tiling_manifest.json data/tiled_styled/
python 03_augmentation_pipeline.py --input_dir data/tiled_styled --output_dir data/augmented_styled --rfs_t 0.1
```
**Check:** `data/tiled_styled/synthetic/adain_fidelity.json` shows no erasure
flags for 8–32 px objects; `python tests/test_mosaic_scale.py data/tiled/synthetic 768` passes.

## Phase 7 — Train

```bash
python 05_train_rtdetr.py --sanity_check --dataset_yaml data/augmented/dataset.yaml       # minutes
python 05_train_rtdetr.py --dataset_yaml data/augmented/dataset.yaml        --output_dir runs/A_noadain --stage all
python 05_train_rtdetr.py --dataset_yaml data/augmented_styled/dataset.yaml --output_dir runs/A_adain   --stage all
```
Compare the two on **real** val (Phase 8, step 1). Keep the better dataset, then
train a second, independent model on it for the ensemble — a different
backbone disagrees more usefully than a different seed:
```bash
python 05_train_rtdetr.py --dataset_yaml <winning dataset.yaml> --output_dir runs/B --stage all \
    --base_model rtdetr-x.pt --seed 1
```

## Phase 8 — Evaluate and produce evidence

1. Per model, on val (calibrates the operating threshold — **never on test**):
   ```bash
   python 08_tiled_eval.py --weights runs/A_noadain/Stage3_HardFinetune/weights/best.pt \
       --data_dir data/raw/val --dataset_yaml data/augmented/dataset.yaml \
       --meta data/raw/val/altitudes.csv --calibrate --out_dir eval/val_A_noadain
   ```
2. Evidence pipeline ablation on val, then test with val's thresholds frozen:
   ```bash
   python 11_evidence_pipeline.py --weights runs/A/.../best.pt runs/B/.../best.pt \
       --data_dir data/raw/val --tta flips --out_dir abl/val
   python 11_evidence_pipeline.py --weights <same> --data_dir data/test --tta flips \
       --stage_thresholds abl/val/<run>/stage_thresholds.json --out_dir abl/test
   python 08_tiled_eval.py --data_dir data/test --cache abl/test/<run>/detections_multiview.npz \
       --conf <multiview threshold from abl/val> --meta data/test/altitudes.csv --out_dir eval/evidence
   ```
   **Check:** the report header says `FROZEN` (not NOT FROZEN / MODIFIED).
   These are the numbers you report.
3. Compress + export for the Jetson:
   ```bash
   python 06_compress_export.py --model_path runs/<best>/Stage3_HardFinetune/weights/best.pt \
       --dataset_yaml <winning dataset.yaml> --output_dir runs/export --target yolov10n
   ```

## Phase 9 — Jetson **[Jetson]**

```bash
python tools/build_calib_set.py --tiled_dir data/tiled --out calib      # on the workstation, copy calib/ over
python 10_jetson_build_engine.py --weights <best.pt> --calib calib/calib.yaml --gate_data data/raw/val
python 07_edge_benchmark.py --benchmark ...        # command printed by 10
python 09_mission_planner.py --bench_report <07's benchmark_report.json> --legs 60:10 25:8 8:3
```
`10` keeps INT8 only if it passes the accuracy gate (writes
`engine_decision.json`). Then fly, and run `07 --video FLIGHT.MP4 --srt_file FLIGHT.SRT`
live; afterwards run `11 --images <evidence pass>` on the 5090 for the evidence report.

## Not verified anywhere yet (first time each runs, watch it)

GPU training and any converged model; RT-DETR's real output format through
06/07 (`(1,300,6)` was checked on CPU only); TensorRT on the Orin NX; AdaIN
with real VGG weights and photos; DJI SRT/XMP parsing on *your* firmware
(check one real file in Phase 5); SIFT registration on low-texture ground.

---

# History and reasoning (reference)

## 1. What is in this package

| Path | Contents |
|---|---|
| `*.py` (root) | Pipeline scripts 02–07. **Scripts live in the project root, not `scripts/`** as the README commands assume. |
| `tools/` | New helper scripts (model review, thumbnails, pile/cable generation) + `tools/reviews/*.json` review decisions |
| `assets/models/<class>/` | **219 reviewed Objaverse models** + 30 `ewaste_pile` + 30 `cable_bundle` procedural assets |
| `assets/models/_review*/` | Review contact sheets + `index.json` (tile → model) for each review round |
| `assets/models/_review/review_log.jsonl` | Timestamped log of every model move/reject (225 actions) |
| `data/raw/synthetic/` | First synthetic batch: **162 labelled images, 529 YOLO boxes** (3 environments) |
| `MANIFEST.sha256` | SHA-256 of every file in the package — verify after transfer |

Not included (still on the laptop): `assets/models/_rejected/` (153 rejected
models, 1.2 GB — recorded in `review_log.jsonl`), the `ewads-laptop` conda env,
the Objaverse download cache (`~/.objaverse`).

## 2. Verify the transfer

```bash
# Linux / Git Bash, from the project root
sha256sum -c MANIFEST.sha256 --quiet && echo "all files intact"
```

## 3. Workstation setup

```bash
nvidia-smi                                  # driver must be >= 570.x for CUDA 12.8
conda env create -f environment.yml
conda activate ewads
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then run the Blackwell kernel check from the README ("Hardware Routing"
section). Expect `capability: (12, 0)` and `kernel launch OK`.

**BlenderProc on the 5090 — verify before any long render.** BlenderProc 2.8
downloads Blender 4.2.1 on first `blenderproc run`. Confirm Cycles actually
uses the 5090 via OptiX (the log prints `Device NVIDIA GeForce RTX 5090 of
type OPTIX found and used`) with a 3-scene test into a scratch folder:

```bash
blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 3 \
  --environment desert --output_dir /tmp/ewads_test
```

If it falls back to CPU, BlenderProc can use a newer Blender via
`--custom-blender-path`.

## 4. What was done on the laptop

1. Created and GPU-verified the `ewads-laptop` environment.
2. **Model acquisition + two review rounds.** Keyword-based Objaverse search was
   ~50% wrong-class (a plate of cookies as `flat_screen`, a boar statue as
   `power_supply`, ...). Added an `--lvis` mode to `02b_objaverse_fetch.py`
   that uses Objaverse-LVIS human-verified categories, then reviewed every
   model visually. Decisions: `tools/reviews/round1_initial.json`,
   `round2_current.json`, `round2_restore.json`.
3. Replaced the broken primitive pile generator with
   `tools/generate_piles.py` (physics-dropped piles of reviewed models;
   procedural tangled/coiled cable bundles).
4. Rendered 200 scenes → 162 kept (38 skipped: no object above the size
   threshold). 0 failures.

### Bugs fixed (would have corrupted data or crashed)

`02b_objaverse_fetch.py`
- Tag search crashed on every query (tags are dicts) → returned 0 models.
- 800K annotations reloaded per search term (now cached once).
- Truncated downloads hung the worker pool forever → serial + skip.
- Pile generator used an OBJ exporter removed in Blender 4.x.

`02_blenderproc_synthetic_v2.py`
- `import blenderproc` must be line 1 (BlenderProc refuses to run otherwise).
- Camera pointed at the horizon (`-90°` pitch); Blender cameras already look down.
- Labels assigned by assumed instance order → wrong objects labelled.
- glTF root EMPTY treated as the model → meshes never placed or labelled.
- No real-world scaling of Objaverse models (arbitrary units).
- `set_sky_background`, `Light.set_direction`, `sensor_width=` don't exist in BlenderProc 2.8; `"Specular"` socket renamed in Blender 4.
- `bproc.init()` called once per environment → crash in `--environment all`.
- Sky + sun double-lit every frame to pure white.
- **Texture leak**: Cycles reused cached textures across models with the same embedded image name (a battery label appeared on TVs).
- **Blank segmentation from scene 2 onward**: object pass indices were only assigned for scene 0.
- **Frame counter** not reset → scene N rendered N frames and saved the wrong one.
- Skipped scenes were reported as unexplained failures (now logged to `skipped_scenes.jsonl` with reasons).

## 5. Open issues — fix before the full 5,000-scene render

> **All four items below are addressed in section 9** (items 1–3 on 2026-09-28,
> item 4 + HDRIs on 2026-10-01).

1. **Flat single-colour terrain.** Every scene is a smooth plain colour.
   The detector will learn "anything on smooth ground = e-waste", hurting the
   FP < 0.05 target on real footage. Needed: (a) a curated ~40-material
   ambientCG ground set (sand, gravel, asphalt, grass, mud) — the full
   `blenderproc download cc_textures` is tens of GB and its file names don't
   match the script's texture tags (`sand_01`, `desert_ground`, ...), so the
   tag matching must be rewritten too; (b) non-e-waste distractors (rocks,
   bushes, tyres, bags, wood) so the model learns negatives. HDRIs: none
   downloaded; a procedural Nishita sky is used.
2. **Unlabelled small objects.** `mobile_phone` and `cpu_chip` got 0 labels:
   at 20–80 m they are 5–30 px, below `MIN_PIXEL_AREA` (150–400 px), so they
   are rendered but not labelled — the model is taught to ignore them, which
   hurts recall. Lower the thresholds so every visible object is labelled,
   and restrict tiny classes to altitudes where they are resolvable.
3. **Thin classes.** `hard_drive` 2, `power_supply` 3, `single_cable` 4,
   `cpu_chip` 6, `metal_component` 7, `plastic_casing` 8 models. No LVIS
   category exists for these; source curated models (Sketchfab / BlenderKit)
   and review them with the tools below.
4. **Class balance** in the first batch is skewed (`ewaste_pile` 109 boxes,
   `metal_component` 6). Revisit `CLASS_SAMPLING_WEIGHTS` once 1–3 are fixed.

## 6. Still blocked (needs you)

Steps 4–7 of the README (AdaIN, augmentation, sanity-check training) need real
drone captures labelled in CVAT: `data/raw/real/images/` + labels, with
`data/raw/val/` and the held-out `data/test/` split off **before** any training.

## 7. Reviewing models (for new or re-fetched models)

```bash
blenderproc run tools/render_model_thumbnails.py \
  --out_dir assets/models/_review_v3 --skip_reviewed assets/models/_review_v2/index.json
python tools/make_review_sheets.py --thumbs_dir assets/models/_review_v3
# write tools/reviews/round3.json (see existing rounds for the format), then:
python tools/apply_model_review.py tools/reviews/round3.json            # dry run
python tools/apply_model_review.py tools/reviews/round3.json --apply
```

Regenerate piles after any model change (they are built from reviewed models):
`blenderproc run tools/generate_piles.py --kind ewaste_pile --n 30`

## 8. README corrections

- Script paths: `scripts/0X_*.py` → `0X_*.py` (project root).
- Laptop GPU is the **6 GB** RTX 3060 Laptop variant, not 8 GB.
- "BlenderProc synthetic generation is CPU/IO-bound" is wrong: Cycles renders
  on the GPU via OptiX (~38 s/scene on the 3060 laptop). The workstation is the
  better machine for rendering and AdaIN training too.
- The README's pile step (`blenderproc run _generate_piles.py`) is superseded
  by `tools/generate_piles.py`.

Laptop versions used: BlenderProc 2.8.0, Blender 4.2.1 LTS, torch 2.4.1+cu121,
ultralytics 8.4.163, objaverse 0.1.7.

---

## 9. 2026-09-28 / 10-01 — fixes for section 5 items 1–4 (renderer v3)

### Run order on the workstation

```bash
# 1. Ground textures (~40 curated ambientCG scans) + outdoor HDRIs (Poly Haven).
#    Both CC0; both write a manifest.json to cite in the paper.
python tools/fetch_ground_textures.py --list_only      # check the picks first
python tools/fetch_ground_textures.py
python tools/fetch_hdris.py --list_only
python tools/fetch_hdris.py
# 2. Models for the six thin classes (40 each), then review them like any round
blenderproc run tools/generate_procedural_models.py --n 40
blenderproc run tools/render_model_thumbnails.py --out_dir assets/models/_review_v3 \
    --skip_reviewed assets/models/_review_v2/index.json
python tools/make_review_sheets.py --thumbs_dir assets/models/_review_v3
#    -> tools/reviews/round3.json, then apply_model_review.py (dry run, then --apply)
# 3. Regenerate piles so they include the new models
blenderproc run tools/generate_piles.py --kind ewaste_pile --n 30
# 4. (optional) real clutter models; review their thumbnails too
python tools/fetch_distractors.py --per_category 6
# 5. Look at 3 scenes per environment BEFORE the long run
blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 3 --environment desert \
    --output_dir /tmp/ewads_test
# 6. Full run, then a low-altitude top-up (the balancer reads the labels already
#    on disk and spends the top-up on the classes that are still short)
blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 5000 --environment all
blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 1000 --environment all \
    --altitude_mode low_evidence --start_id 5000
```
Read the table each run prints (class, labels, visible-unlabelled, all-time
share vs target) — it is also saved as `generation_stats_<start_id>.json`.

### What changed in `02_blenderproc_synthetic_v2.py`

1. **Ground.** Subdivided terrain with real relief (baked, so ray casts match the
   render); PBR material blending two ambientCG scans at physical scale with a
   noise mask + macro colour drift (breaks tiling). Without textures it falls back
   to a multi-tone procedural ground and prints a warning — never flat again.
   **Clutter**: rocks, shrubs, sticks/logs, tyres, cardboard boxes, bags,
   bottles/cans, rope/hose (hard negative for cables), concrete blocks —
   rendered, never labelled; density per environment. `--negative_fraction`
   (default 6%) renders clutter-only scenes with empty label files.
2. **Small objects.** Camera = real Phantom 4 Pro (13.2 mm sensor, fixed 8.8 mm,
   73.7° HFOV; the old 17.3 mm/35–90° FOV was wrong). Ground sampling distance is
   computed first; a model spawns only if it will be ≥ 16 px long (cables ≥ 1 px
   wide), and every visible e-waste object with ≥ 12 visible px is labelled.
   Objects are scattered inside the camera footprint. Resolution mix: 4K video
   55%, 20 MP stills 30%, 1080p 15%.
   Consequence to know: a 4 cm CPU chip is only resolvable below ~6 m (4K) /
   ~9 m (stills), so chips appear only in low-altitude scenes. That is physics,
   not a bug — report cpu_chip AP per altitude band in the paper.
3. **Thin classes.** `tools/generate_procedural_models.py` builds hard drives
   (3.5"/2.5"/SSD/opened), PSUs (ATX/laptop brick/wall adapter/transformer),
   single cables (power/USB/Ethernet/AV/bare), chips (LGA/PGA/QFP/BGA), metal
   parts (heat sinks, tower coolers, panels, brackets, drive cages) and plastic
   casings (open shells, monitor backs, panels, broken fragments) at real
   dimensions with embedded label/PCB/vent textures. Each has a sidecar
   `<model>.json` with its true size, which the renderer now honours.
4. **Size bugs found while testing:** `lighting` models (mostly LVIS
   "lightbulb") were scaled to 1.2 m tube length, and AA cells in `battery` to
   25 cm. Both classes are now sized by shape (elongated vs compact).
5. **Removed no-ops:** world mist (Cycles only writes it to a Mist pass) and
   `use_motion_blur` on a static single frame never changed pixels. Haze, blur
   and sensor noise come from `03_augmentation_pipeline.py`.

6. **Class balance (item 4).** Fixed weights can't fix it: the imbalance comes
   from eligibility (a pile is resolvable in every scene, a 4 cm chip in ~1 in
   20). The sampler now tracks labels written per class (seeded from labels
   already on disk) and corrects toward the tier-weighted target: weights ×
   (target/observed share), and over-represented picks are thinned. A
   simulation of 5,000 scenes + 1,000 low-altitude top-up on the test model
   set: rarest class 297 → 655 labels, `ewaste_pile` 8,808 → 4,325, distance
   from target 0.328 → 0.165. Exact shares are not reachable for chips without
   distorting the altitude mix, so finish balancing at TRAINING time with
   repeat-factor sampling (LVIS-style: oversample images containing rare
   classes). `--no_balance` gives the fixed-weight baseline for an ablation.
7. **HDRIs.** `tools/fetch_hdris.py` (Poly Haven, outdoor, daylight, per
   environment). 60% of scenes use an HDRI, 40% the Nishita sky + sun. HDRI
   scenes no longer add the sun lamp + fill light (the HDRI contains the sun;
   the old branch would have double-lit, and would likely have crashed: the
   world created by `bproc.clean_up()` has no node tree). In the Nishita path
   the sky's sun now matches the lamp's position (they were sampled separately,
   so sky colour and shadows disagreed).
8. **Thin-class source.** Section 5 suggested curated Sketchfab/BlenderKit
   models; the procedural generator was chosen instead because it is
   scriptable, correctly scaled, licence-free and unlimited. Curated models are
   still welcome: drop the .glb into the class folder (optional sidecar .json
   with `real_size_m`) and it goes through the same review.
   Procedural laptop bricks gate on the brick body (`gate_frac`), not the
   cable-inclusive extent — otherwise they spawned at altitudes where only
   their cable was resolvable.

### Also changed

- `03_augmentation_pipeline.py`: `--hard_negative_rate` default 0.10 → 0. It
  cleared the labels of images that still contained e-waste (teaching the model
  e-waste = background). Use the renderer's true negatives instead.

### Still open

- ~~Resize kills the small-object fix~~ — **fixed 2026-10-02, see section 10.**
- Model review round 3 (procedural models + distractors) is needed before the run.


## 10. 2026-10-02 — native-resolution tiling (the "resize" blocker)

**Problem.** Three steps destroyed the detail the renderer now produces: `03` shrank
every frame to 640 px (20 MP → 8.5x), `04` AdaIN squashed to 512x512 and upsampled the
result back (an 8x blur on 4K), and `07` squashed the whole frame to a square at
inference. A 24 px phone became ~3 px, and the "imgsz 768" in `05` only upsampled an
already-shrunk image. Nothing in the renderer fix could be learned.

**New pipeline order**

```
BlenderProc / real frames (native res)
  -> 03b_tile_dataset.py      cut native-resolution 768 px tiles (+ coarser pyramid levels)
  -> 04_adain_transfer.py     style transfer ON TILES, native resolution (no resize)
  -> 03_augmentation_pipeline.py   augment tiles, repeat-factor list for rare classes
  -> 05_train_rtdetr.py       imgsz = tile (read from dataset.yaml `ewads:` block)
  -> 08_tiled_eval.py         FULL-FRAME mAP / recall / FP, per class / size / altitude
  -> 06_compress_export.py    ONNX/TRT at the tile size; tiled pseudo-labelling
  -> 07_edge_benchmark.py     tiled inference + per-tile and per-frame latency
```

```bash
# test set stays FULL-RES and untouched (never tiled, never augmented)
python 03b_tile_dataset.py --input_dir data/raw --output_dir data/tiled --dry_run   # sizes only
python 03b_tile_dataset.py --input_dir data/raw --output_dir data/tiled
#   read its per-class table: "whole>=90%" must be ~100% for every class
python 04_adain_transfer.py train    --content_dir data/tiled/synthetic --style_dir data/raw/real/images ...
python 04_adain_transfer.py transfer --input_dir data/tiled/synthetic --output_dir data/tiled_styled/synthetic ...
python 03_augmentation_pipeline.py --input_dir data/tiled --output_dir data/augmented --rfs_t 0.1
python 05_train_rtdetr.py --dataset_yaml data/augmented/dataset.yaml --stage all
python 08_tiled_eval.py --weights runs/ewads/Stage3_HardFinetune/weights/best.pt \
    --data_dir data/val_full --dataset_yaml data/augmented/dataset.yaml --calibrate --out_dir eval/val
python 08_tiled_eval.py --weights <same> --data_dir data/test --dataset_yaml data/augmented/dataset.yaml \
    --conf <threshold from val> --meta data/test/scene_index.jsonl --out_dir eval/test
```
Hold a FULL-RESOLUTION `val_full` set (frames never tiled for training) for calibration,
and a separate `test` set for reporting. If `04` is skipped, point `03` at `data/tiled`.

### How the tiling works (`ewads_tiling.py`)

- Level 0 = native pixels cut into overlapping 768 px tiles (overlap 128). Further
  levels shrink the frame by <= 2x each until it fits one tile, so objects larger than
  a tile are still seen whole. 4K = 33 tiles, 20 MP = 74 (1.3x the native tiles alone).
- Any object up to `overlap` px is whole in some tile of its level; every object is whole
  in the last level; neighbouring levels' size bands overlap, so every size >= 8 px is
  covered. Round-trip test: 2,100 simulated objects (10-2,500 px), 100% recovered,
  0 duplicates, 0 false positives.
- Training labels: keep a box if >= 25% visible, >= 8 px long, clipped >= 4 px.
  A tile with a visible-but-unlabelled object is never used as a clean negative.
  Clean empty tiles (clutter, true-negative frames) are kept at 12% as hard negatives.
- Inference merge: drop detections touching an interior tile border (fragments),
  class-aware NMS, nested-duplicate suppression.
- Tiles are split with their source frame, so there is no train/val leakage.
  `tile_index.jsonl` maps every tile to its frame, level and offset (provenance).

### What else changed

- **`03`**: policies ported to Albumentations **2.x** (old 1.x names — `var_limit`,
  `value`, `quality_lower`, `ImageCompression.ImageType` — crash or are silently
  ignored on 2.x, which is what `albumentations>=1.4.0` installs). Free rotation off by
  default (axis-aligned boxes re-fit around rotated boxes get up to ~1.4x looser per
  side for non-rectangular objects: cables, piles, bulbs; `--rotate_limit 10` to
  re-enable). CoarseDropout replaced by a label-aware cutout (it could erase a 12 px
  object and keep its label). 6-decimal label rounding (-5e-7) crashed Albumentations
  2.x; class ids were written as `7.0`. `--workers` (it was single-process). Refuses to
  downscale (`--allow_downscale`), `--add_mosaic` refused (halves resolution; Ultralytics
  mosaics at train time anyway), repeat-factor sampling -> `train_rfs.txt`.
- **`05`**: imgsz = tile for all stages, `multi_scale` off and `scale` 0.25 on tiles
  (50% shrink turns 12 px into 6 px), the "reduced VRAM" profile lowers batch not imgsz,
  default batch 16 at 768.
- **`06`**: [syntax] the file could not be imported on Python 3.10 (backslash inside an
  f-string); fixed. Image size flows to QAT/ONNX/TensorRT; TensorRT input name is
  `images` (was `input`, which would fail); INT8 now warns and points to Ultralytics'
  calibrated export (trtexec `--calib` wants a cache file, not a folder); pseudo-labels
  are produced by the tiled teacher on full frames and **are now actually used** (the
  student ignored `pseudo_dir` before).
- **`07`**: tiled inference path (`--no_tiled` for the old one), per-tile and per-frame
  latency, tiling parameters written into the audit-log `model_version`.
  New raw-output decoder checked against Ultralytics: **current Ultralytics exports
  RT-DETR as (1,300,6) `cx cy w h conf cls` in pixels**, not `x1 y1 x2 y2` — the old
  parser would have put every deployed box in the wrong place.
- **`08`**: new. COCO metrics in numpy, verified equal to pycocotools (diff 1e-16).

### Honest limits / still open

- **Tested** on my sandbox renders (11 frames), synthetic pixel-detector frames up to
  20 MP, and CPU-only Ultralytics (RT-DETR-L trains 1 epoch on the tiles; ONNX parity
  with `predict()` exact for RT-DETR and a trained YOLOv8 head). **Not tested:** GPU
  training, a real converged model, TensorRT/Jetson, `04` AdaIN (needs torch + your real
  style photos), and `03b` on the full 5,000-scene set.
- **Latency.** The model runs once per tile: ~33 calls for a 4K frame, 74 for 20 MP.
  Measure on the Jetson (`07 --benchmark` prints per-frame cost). 10 FPS on 4K is not
  possible with tiling; a survey does not need it (1 Hz at survey speed), or use
  `--max_levels 1`.
- **Objects < 8 px** (renderer labels down to 4) are intentionally unlabelled in tiles;
  `03b` reports them per class (`<8px` column). Expect near-zero cpu_chip recall above
  ~6 m — that is physics; report per-altitude AP.
- **AdaIN** must be retrained on native-resolution crops (the old checkpoint, if any,
  was trained on 512 px squashes). It also needs real style photos first.
- ~~**INT8 calibration** on the Jetson still needs a calibration cache (see `06`).~~ → section 11.
- `environment.yml`: pin `albumentations>=2.0,<3` and add `pyyaml`.
- Tests: `python tests/test_metrics_vs_coco.py`, `tests/test_tiled_pipeline.py`,
  `tests/test_onnx_parity.py [yolo_checkpoint.pt]`.

## 11. 2026-10-02 (second pass) — the seven tiling caveats

Each item from the "Honest limits" list above, what it turned out to be, and what now handles it.
Two of them were not real problems once measured; two hid bugs.

| # | Caveat | Verdict | Now |
|---|---|---|---|
| 1 | 33–74 model calls per frame, no 10 FPS | **wrong question** — a survey needs every patch of ground *seen*, not 30 fps | `09_mission_planner.py`, `07 --video` coverage scheduling, `--auto_levels`, batched tiles |
| 2 | objects < 8 px, cpu_chip dead above ~6 m | **physics** — plan for it and say so in the evidence | per-class max altitude (09); every audit record lists assessable / NOT-assessable classes; 08 reports the ≥16 px envelope next to all-object numbers |
| 3 | AdaIN untested, needs real photos | **had a bug** | `04` training clamp fixed; object-preserving blend + detail-retention report; `tools/check_style_photos.py` |
| 4 | INT8 needs a calibration cache | **07 could not load the engine at all** | `tools/build_calib_set.py`, `10_jetson_build_engine.py` (FP16 + INT8 + accuracy gate); 07 TensorRT backend rewritten |
| 5 | free rotation off | **measured, stays off** | numbers in `03` header (`tests/test_rotation_boxes.py`); `--rotate_limit` now uses the tighter `ellipse` box method |
| 6 | `--add_mosaic` refused | **not a problem** — Ultralytics' train-time mosaic is already native-scale | measured (`tests/test_mosaic_scale.py`); message explains |
| 7 | disk unknown | **measured** | `03b --dry_run` encodes sample tiles and estimates 03b + 04 + 03 totals vs free space |

### New run order (additions in **bold**)

```
02 render  ->  03b --dry_run (disk)  ->  03b tile
  ->  **tools/check_style_photos.py** (vet real photos)  ->  04 train / transfer (on tiles)
  ->  03 augment  ->  05 train  ->  08 eval (all + envelope)
  ->  06 compress/export ONNX
  ->  [on the Jetson] **tools/build_calib_set.py** -> **10_jetson_build_engine.py** (FP16 vs INT8 gate)
  ->  07 --benchmark  ->  **09_mission_planner.py --bench_report**  ->  fly  ->  **07 --video --srt_file**
```

### 1 — speed: what rate is actually needed (`ewads_survey.py`, `09`, `07 --video`)

The processed-frame rate is set by flight geometry: a frame must be processed each time the
ground has moved `(along-track footprint − largest object) / views`. Phantom 4 Pro, 4K video,
1/1000 s shutter, `views=2`, objects ≤ 1 m (09's default legs; tile cost assumed 20 ms — **measure yours**):

| leg | GSD | process rate | tiles/s | duty @ 20 ms/tile | assessable (smallest realistic instance ≥ 16 px) |
|---|---|---|---|---|---|
| 60 m, 10 m/s | 2.34 cm | 0.40 Hz | 12.5 | 25% | piles, large appliances, flat screens only |
| 25 m, 8 m/s | 0.98 cm | 0.80 Hz | 26 | 53% | everything except battery (AA), single cable, cpu_chip |
| 8 m, 3 m/s (blur-limited) | 0.31 cm | 1.04 Hz | 34 | 69% | all but battery/cpu_chip (marginal) |

`07 --video FLIGHT.MP4 --srt_file FLIGHT.SRT` does this live: motion from SRT speed/GPS, else
image phase correlation against the last processed frame; an unreliable estimate = process
(fails towards recall); `--max_gap_s` forces a frame at least every 2 s. Each processed frame's
record holds the video SHA-256, frame index/time, decoded-pixel SHA-256, why it was scheduled and
how many frames were skipped before it. Verified on a synthetic flight: telemetry and image
motion pick the same frames, and every 1 m object lands in ≥ 2 processed frames.
**Low altitude is blur-limited, not compute-limited**: at 8 m a 1/1000 s shutter allows ~3 m/s.

### 2 — tiny objects: plan altitude, record assessability

- `python 09_mission_planner.py --class_table` — max altitude per class and camera mode.
  cpu_chip (2.8 cm smallest): 4.5 m on 4K video, 6.4 m on 20 MP stills. AA battery: 7 / 10 m.
- Every `07` audit record now has `context.classes = {assessable, marginal, not_assessable,
  unknown}` plus altitude source, GSD, gimbal pitch (oblique > 10° flagged), exposure, blur.
  **Altitude for GSD = DJI XMP `RelativeAltitude` / SRT `H` (above take-off), never EXIF
  `GPSAltitude` (above sea level)** — using that would make GSD wrong everywhere above sea level.
  On sloping sites pass `--altitude_agl`.
- `08` prints both **all objects** and the **assessable envelope** (objects ≥ `--assess_px` 16,
  COCO-style ignore) with PASS/FAIL for each. Quote the envelope only together with its definition.
- `tests/test_survey.py` parses `02` (via `ast`) and fails if the class sizes, thresholds, camera
  or class order drift from `ewads_survey.py`.

### 3 — AdaIN (`04`)

- **Bug:** training clamped the decoder output to [0, 1] in ImageNet-*normalised* space
  (≈ −2.1…2.6) → decoder trained on a narrow grey band, no gradient outside it → washed-out
  styles. Removed. The every-5-epoch checkpoint also saved the *best* weights under the
  wrong epoch name; fixed.
- `transfer` keeps 65% of the original pixels inside labelled objects (`--object_alpha 0.35`,
  `--no_protect_objects` to disable): AdaIN rebuilds the image from 1/8-resolution features,
  where a 16 px object is 2×2 cells. It writes `adain_fidelity.json` (detail retained per size
  bucket) and flags erasure of 8–32 px objects. Ran end to end on CPU (random-init VGG — this
  machine cannot download weights): sizes preserved, labels byte-identical; the report showed
  0.00 detail without protection and 0.65 with (untrained decoder, so only the mechanism is proven).
- `tools/check_style_photos.py --photos <flights> --exclude_dir <real test images> --copy_usable data/style_photos`:
  rejects blurry, oblique, over/under-exposed, too-small, duplicate and **test-set-leaking**
  photos, and counts usable photos per renderer altitude profile.
- Still true: AdaIN must earn its place — train 05 with and without it and compare on **real**
  labelled val frames.

### 4 — INT8 and the TensorRT backend

- **Bug (would have failed on day one):** 07's TensorRT class used the binding API removed in
  TensorRT 10, could not read Ultralytics `.engine` files (4-byte length + JSON header before the
  engine), allocated GPU memory on every call, and ran one tile per call. Rewritten: TRT 8.5+/10
  tensor API with legacy fallback, header handling, static or dynamic batch (Ultralytics dynamic
  engines are dynamic in H/W too — tile size comes from the opt shape / metadata), buffers once,
  torch for device memory (no pycuda). `tests/test_trt_backend.py` checks the logic against a
  fake TensorRT (both APIs × static/dynamic × header/no header) — **not** a substitute for the Jetson.
- ONNX backend batches too (static exports are padded); batch-4 vs batch-1 detections identical.
- `tools/build_calib_set.py`: tiles (never full frames), every class, ~10% empty ground, real
  tiles first, pyramid-level mix of the tiler, refuses test splits; writes `calib.yaml`
  (checked to load in Ultralytics 8.4.171, which uses `quantize=8`, not `int8=True`).
- `10_jetson_build_engine.py` (Jetson only): exports FP16 and INT8 (dynamic batch ≤ 8), scores
  both with `08 --engine` on full-res val frames at the same threshold, times both with
  `07 --benchmark`, and keeps INT8 only if envelope mAP and recall drop ≤ 0.01 **and** no Tier-1
  class loses > 0.03 AP. Writes `engine_decision.json` and prints the 09/07 commands.

### 5, 6, 7

- Rotation: ±10° costs mean box IoU 0.80–0.95 (thin objects: 5th percentile 0.36) — label noise for
  mAP@.5:.95. The renderer's uniform yaw + exact flips/90° turns already cover orientation.
- Mosaic: Ultralytics 8.4.171 places 768 px tiles at native scale (median object 99.5 → 94.3 px,
  0% < 8 px). Offline mosaic stays refused.
- Disk: `03b --dry_run` JPEG-encodes ≤ 24 sample frames' tiles in memory and prints KB/tile, the
  03b total, and what 04 (`--n_styles`) and 03 (`--n_aug`) will add, against free space. Real runs
  record the bytes actually written. Sandbox renders came out at 64 KB/tile (q95; low texture —
  expect 2–4× that with real ambientCG ground).

### Still not verified (needs your hardware)
GPU training and a converged model; any accuracy number; TensorRT on the Orin NX (07 backend,
10's export + gate); AdaIN with real VGG weights and real photos; DJI SRT/XMP on *your*
firmware (parsers handle the documented P4P and newer formats — check one real file);
4K-video sensor crop on the P4P (09 assumes none; photograph a 1 m target at known height once
and compare the measured px with 09's GSD).

## 12. 2026-10-03 — the 0.68 problem: offline evidence pipeline (b) + honest evaluation (a)

**Why.** mAP@0.5:0.95 averages IoU 0.50…0.95. A 1-px box error on a 32 px object already fails
IoU 0.95; on a 16 px object it fails 0.80. Survey frames (Jetson, INT8, high altitude) cannot reach
0.68 on *all* objects, and the published state of the art on aerial small objects sits far below
it. So the work is split: the **Jetson finds sites**, the **RTX 5090 produces the evidence**, and the
targets say which number applies where — fixed *before* any results.

### (b) Offline evidence pipeline — `11_evidence_pipeline.py` + `ewads_fusion.py`

| Stage | What it does | What it can / cannot fix |
|---|---|---|
| D4 TTA (`--tta flips` 4 views, `d4` 8) | flips/90° rotations, boxes mapped back **exactly**; all in 03's training augs | box jitter (AP75+); NOT systematic confusions |
| `--scales 1.0 1.5` | bicubic up-scaling of the frame (no AI super-resolution) | 8–20 px objects; cost ×2.25 per scale |
| Ensemble (2–3 `--weights`) | independently trained models | errors that differ between models |
| WBF | boxes averaged, not NMS-discarded; `n_runs` = "found in k of N runs"; **one box per run per cluster** and `--wbf_pad_px 1.5` (plain WBF split small objects into duplicates — measured) | localisation, per-run FPs |
| Multi-view | SIFT homography (phase correlation / GPS fallback) links overlapping frames; tracks static objects; score = consistency over the frames that covered the object | single-frame glints/blur FPs (22 → 2 in the test); NOT objects the model always gets wrong |
| Review | duplicate tracks merged; crops cut from **hash-verified** source pixels; `review.csv` for the analyst | — |
| Audit | per-run folder `run_<UTC>_<manifest>`; manifest = model + code SHA-256 + config; hash-chained log of frames + candidate objects (observed class/conf **and** track vote) | re-runs never overwrite earlier evidence |

Toy-detector ablation (`tests/test_fusion.py`, NOT a prediction for the real model):
mAP .285 → TTA .571 → TTA+ensemble .656 → multi-view .655; FP share at recall ≥ 0.87:
.315 → .104 → .091 → .098; single-frame glints 22 → 2. `multiview+fill` *raised* FPs in the toy —
keep it off unless your VAL ablation says otherwise.

Run order (5090, after training ≥ 2 models with different seeds/backbones):
```
# 1. ablation on a labelled VAL *sequence* (overlapping frames) -> per-stage thresholds
python 11_evidence_pipeline.py --weights runs/a/best.pt runs/b/best.pt --data_dir data/val_seq --tta flips --out_dir abl/val
# 2. same on TEST with the VAL thresholds frozen (prints a warning if any fusion setting differs)
python 11_evidence_pipeline.py --weights ... --data_dir data/test_seq --tta flips \
       --stage_thresholds abl/val/<run>/stage_thresholds.json --out_dir abl/test
# 3. score the final stage against the pre-registered targets
python 08_tiled_eval.py --data_dir data/test_seq --cache abl/test/<run>/detections_multiview.npz \
       --conf <multiview threshold from step 1> --meta data/test_seq/scene_index.jsonl --out_dir eval/evidence
# 4. a real flight's evidence pass
python 11_evidence_pipeline.py --weights ... --images D:/flights/<date>/low --tta flips \
       --report_conf <multiview threshold> --out_dir evidence/<date>
```
`--reuse_cache` re-fuses without re-running models (settings of the cached runs are enforced).
`--dry_run` prints tile inferences per frame (2 models × 4 flips on 20 MP = 592).

### (a) Evaluation — `ewads_taxonomy.py`, `ewads_hier_eval.py`, 08, `targets_v1.json`

- **Taxonomy**: 18 classes → 8 groups (display, appliance, computer_component, board_chip,
  cell_device, cable, lighting, pile) → binary. 08 reports mAP/recall/FP at all three levels:
  group ≫ fine means the errors are confusions between look-alikes, not misses.
- **Tier-1** (battery, circuit_board, crt_monitor) reported on its own.
- **Altitude × size matrix** of mAP (needs `--meta`).
- **Sites** (survey view): objects within 3 m = one site; site recall and false alerts per frame.
- **Pre-registered targets** (`targets_v1.json`): evidence E1 mAP@.5:.95 ≥ 0.68 on objects ≥ 32 px,
  E2 mAP@.5 ≥ 0.85 ≥ 16 px, E3 tier-1 recall ≥ 0.87, E4 FP share < 0.05; survey S1 site recall ≥ 0.95,
  S2 false alerts < 0.05/frame. Each section is scored only on frames in its altitude scope; without
  altitudes there is no verdict unless you assert the scope with `--targets_section`.
- **Freeze before training**: `python tools/freeze_targets.py --targets targets_v1.json --by "<name>"`,
  then put the printed hash in a git commit / email. 08 prints FROZEN / NOT FROZEN /
  MODIFIED AFTER FREEZE on every report. A change after freezing = `targets_v2.json`, report both.

### Verified / not verified
- Verified (CPU, Python 3.10 and 3.11): WBF identical to `ensemble_boxes` (60 random cases);
  D4 exact on pixels; registration ≤ 1 px incl. rotation+scale and mixed frame sizes; GPS heading
  convention; end-to-end stills + lossless video with a pixel-based fake detector; crops byte-identical
  and refused when the source changed; hierarchy/sites/targets/freeze logic; 08 keys used by 10 intact.
  An independent review found 7 defects (mixed-size registration, run overwrite, unverified crops,
  frame records relabelled by the track vote, threshold scale in `track` mode, mixed caches,
  out-of-scope verdicts) — all fixed and covered by tests.
- NOT verified: gains on a real model (only your VAL/TEST ablation can say); SIFT on real
  low-texture ground (sand, water) — check `registration` methods in the frame records; DJI
  `GimbalYawDegree` = image-top heading (GPS fallback only); `tests/test_trt_backend.py` needs torch.
- Lighting (fluorescent tubes / CFLs contain mercury) is NOT tier-1 in 07 — decide before freezing.

## 13. 2026-10-08 — synthetic-only regime, Task 1: targets v2 (NOT frozen)

Project is now synthetic-only for training/calibration/model selection (real data = AdaIN style
photos + a held-out real staged test set + 1 m reference-target photos).

- **`targets_v2.json`** (new, `schema_version: 2`, DRAFT): sections `synthetic_heldout.evidence`
  (E1–E4, = v1), `synthetic_heldout.survey` (S1–S2, = v1), `sim_to_sim_loeo` (L1–L4, every fold
  must pass), `sim_to_real` (R1–R5: tier-1 recall@IoU0.5, binary recall, site recall, false
  alerts/frame, mAP@.5 ≥16 px — mAP@.5:.95 report-only), `h2_gsd` (H2), `h3_tamper` (T1–T7).
  Every target has `metric`, `scope`, `value`, `why`; new values are marked PROPOSAL.
  Each section names its `dataset` and `evaluated_by` tool.
- **Decision: `lighting` is NOT tier-1** (recorded in `targets_v2.json` → `decisions`): the class
  does not separate mercury lamps from ordinary bulbs. Lighting recall is reported per class.
- **`ewads_hier_eval.py`**: `validate_spec()` (v1 + v2 schema), `select_sections()`; `check_targets`
  keeps a v2 target's declared scope as `scope_declared` (it was overwritten by the applied scope).
- **`08_tiled_eval.py`**: default `--targets` is now `targets_v2.json`; new `--targets_dataset
  {synthetic_heldout, sim_to_sim_loeo, sim_to_real}` — a v2 file only scores that dataset's
  sections (H2/H3 never by 08). Without it the run completes and targets are skipped with a message.
  v1 works unchanged (`--targets targets_v1.json`).
- **`tools/freeze_targets.py`**: validates the file first; an invalid file is never frozen (exit 2).
- **Tests**: new `tests/test_targets_v2.py`. `tests/test_hier_eval.py` now passes
  `--targets targets_v1.json` explicitly. `tests/test_onnx_parity.py`: seeded, skips degenerate
  random-init outputs, and compares the top-10 detections tie-robustly (it failed intermittently:
  random-init scores tie and the two sorts broke ties differently).
- Note for later: `07`/`08` decode (1,300,6) heads with `parse_raw_output(fmt="auto")`; that is
  sound for a trained model but cannot classify a fully degenerate output. Passing the format
  explicitly from the model type would remove the heuristic (not changed).
- Backups: `backups/*.bak_20261008`.

## 14. 2026-10-08 — synthetic-only regime, Task 2: training configuration, splits, selection on synthetic val

- **`05_train_rtdetr.py`**: `--synthetic_only` (default on; `--no-synthetic_only` = legacy). Stages
  `Stage1_FrozenBackbone` → `Stage2_FullFinetune` (lower LR) → `Stage3_Polish` (5 epochs, LR 2e-5,
  mosaic off; `--no_polish` skips). The real-data `data_mix` labels and the hard-negative-mining stub
  are disabled (and logged as such). Refuses a dataset whose TRAIN sources include anything but
  `synthetic`. Writes `training_provenance.json` (COCO base weights path + SHA-256, each stage's init
  checkpoint SHA-256, dataset.yaml SHA-256, seed, torch/ultralytics/CUDA versions, final model SHA-256)
  and `final_model.txt`.
- **`03_augmentation_pipeline.py`**: dataset.yaml `ewads:` block records `sources` (input splits in TRAIN).
- **`tools/make_synth_splits.py`** (new): scene-level (sequence-level) split stratified by environment ×
  altitude profile → `main/{synthetic,val,heldout,val_seq}` and `loeo_<env>/{synthetic,val,heldout}`;
  hard links; `split_manifest.json` with per-file SHA-256 lists and list hashes; leakage check.
- **`02_blenderproc_synthetic_v2.py`**: records `capture_mode` per scene; `--sequence_frames N`
  renders N overlapping along-track frames per scene (spacing = 11's CoverageScheduler, via the new
  `ewads_survey.sequence_offsets_m`), files `<stem>_fNN.jpg`, index fields `sequence_id`,
  `frame_index`, `n_frames`, `camera_offset_m`; a sequence that would leave the terrain is skipped
  with a reason. N=1 (default) keeps the previous names and behaviour. Verified by a real low-res
  Blender render on the laptop: the middle sequence frame's labels are identical to the single-frame
  render of the same seed, and the 3 frames overlap as expected.
- **Selection guard** (`ewads_hier_eval.selection_data_problem`): `08 --calibrate`, `10 --gate_data`
  and `11` (when deriving stage thresholds) refuse folders whose last three path parts contain
  `test*` / `*heldout*`; `--allow_selection_on_heldout` overrides (printed). 10 and 11 check before
  any engine build / inference.
- **Synthetic-only runbook** added at the top of this file.
- **Tests**: new `tests/test_synth_splits.py` (splits, LOEO, manifest hashes, 03b layout contract,
  03 sources + 05 refusal, sequence geometry, selection guard incl. 08/10/11).
- Not unit-tested: the renderer's multi-frame render loop itself (Blender-only); covered by the live
  render check above.
- Backups: `backups/*.bak_20261008` (HANDOFF pre-task-2: `HANDOFF.md.bak_20261008_task2`).

## 15. 2026-10-08 — forensic hardening, Task 3 (07 / 11 / tools)

- **Capture time in every audit record** (`07` stills + video, `11` stills + video): `capture_time_utc`,
  `capture_time_local`, `capture_time_source` (new `ewads_survey.capture_time_from_image` /
  `capture_time_from_srt`; `parse_dji_srt` now keeps a per-frame `datetime_text` when the firmware writes
  one). Sources: EXIF `DateTimeOriginal` (+`SubSecTimeOriginal`, +`OffsetTimeOriginal`), XMP
  `CreateDate`/`DateTimeOriginal`/`DateCreated`, DJI SRT per-frame date-time. **UTC is only claimed when the
  source has an offset**; otherwise `capture_time_utc = null` and the camera-local time is kept (source says
  "zone unknown"). `--capture_utc_offset +04:00` (07, 11) asserts the camera clock's zone; such records say
  "ASSERTED by operator". Nothing found → `"unavailable"`. The Phantom 4 Pro SRT carries no date-time.
- **`timestamp_utc` = processing time** (documented in 07's logger and README).
- **`model_sha256` in every record**: 07 hashes the engine / ONNX file it actually loaded (also in
  `benchmark_report.json → model`); 11 records the list of ensemble members' SHA-256.
- **Verify on resume**: `ForensicAuditLogger.__init__` runs `verify_chain()` before appending to an existing
  log and raises if it is broken ("preserve as evidence, start a new log"). `07 --verify_chain` opens with
  `verify_on_open=False` so it can still report. `verify_chain` now reports an unparseable / truncated
  record instead of crashing (found by the new test).
- **Signed chain head**: `tools/sign_chain_head.py keygen | sign | verify` (Ed25519, `cryptography`, now in
  `environment.yml`). The head pins record count, first/last record hash and the whole-file SHA-256; `sign`
  refuses a broken chain and keeps earlier heads; `verify` only passes against a **trusted** fingerprint or
  public key. `11 --signing_pubkey <pem>` writes the key fingerprint into `run_manifest.json`.
  **Production:** keep the private key on a hardware token (smart card / HSM / YubiKey PIV), never in a file
  or in this repository (keygen refuses repo paths), and record the fingerprint outside the system. **Next
  step: RFC 3161 trusted timestamping of each signed head.**
- Also fixed: `07 run_forensic_inference` listed images case-sensitively (`*.jpg`), missing DJI `.JPG` on Linux.
- Tests: new `tests/test_forensic_hardening.py`.
- Backups: `backups/*.bak_20261008`, `*.bak_20261008_task3`.

## 16. 2026-10-08 — H3 tamper experiment, Task 4

- **`tools/tamper_experiment.py`** (new). Builds a hash-chained log with 07's own logger from
  deterministic synthetic records whose media files exist (or uses a COPY of a real log: `--log`,
  `--media_root`), signs its head with an ephemeral Ed25519 key (temp dir, never in the repo), then
  applies N trials (default 200, seeded) per tamper type — field_edit, reorder, insert, middle_deletion,
  tail_truncation, full_rebuild, source_swap — and checks each at three levels: `chain`, `chain+head`,
  `chain+head+rehash`. Writes `h3_tamper.json` (counts, rates, targets_v2 H3 PASS/FAIL **with the
  targets' freeze status**, code SHA-256), `h3_tamper.csv`, `h3_tamper_trials.jsonl`.
- Safety: tampered media are written to a per-trial overlay; originals (incl. a real log and its media)
  are only read (test proves byte-identical afterwards). Without `--media_root` a real log's source_swap is
  reported not-applicable rather than touching media. An untampered baseline and an untampered control copy
  must pass all checks or the run aborts.
- `tools/sign_chain_head.py`: 07 is loaded once per process (speed; logic unchanged).
- A 20-trial smoke run (scratch, not a result) showed the designed pattern. **The official run is NOT done:**
  it waits until `targets_v2.json` is frozen. Command:
  `python tools/tamper_experiment.py --n_trials 200 --out_dir results_bundle`
- Tests: new `tests/test_tamper_experiment.py`.

## 17. 2026-10-08 — H2 GSD / assessability validation, Task 5

- **`tools/gsd_validation.py`** (new). Inputs: reference-target photos, `heights.csv`
  (`stem,height_m,camera`), and the target's two endpoints from an endpoints CSV, CVAT XML (polyline /
  2 points) or YOLO-pose keypoints. Predicted length = `L / ewads_survey.gsd_m(...)` (the same function as
  the renderer / planner / 07). Intrinsics per image, source recorded: a camera profile (mode inferred from
  the image size; uses `width_crop`), else EXIF `FocalLength × FocalPlaneXResolution`, else
  `FocalLengthIn35mmFilm` (diagonal convention), else `FocalLength` + `--sensor_width_mm`; images without any
  are skipped with the reason. Records DJI gimbal pitch (flag if |pitch+90| > 5°) and the target's offset
  from centre. Outputs `h2_gsd.json` (per image, summaries by camera / height / mode, signed error,
  **implied video sensor-width fraction / crop factor** for profile cameras in video modes, H2 check with the
  targets' freeze status), `h2_gsd.csv`, `h2_gsd_scatter.png` (camera·mode series, ±10% band, identity line;
  palette validated with the dataviz validator, marker shapes as secondary encoding).
- If 4K video shows a crop: set `ewads_survey.CAMERAS["phantom4pro"].width_crop["video_4k"]` to the reported
  fraction (this changes the renderer's resolvability rules too — decide BEFORE rendering / freezing).
- Tests: new `tests/test_gsd_validation.py` (synthetic photos with known geometry: exact errors, EXIF paths,
  crop 0.8 recovered, oblique flag, three annotation formats agree).
- Not run on real data (none yet). Command once the photos exist:
  `python tools/gsd_validation.py --photos <dir> --heights <dir>/heights.csv --cvat_xml <dir>/annotations.xml`

## 18. 2026-10-08 — baselines and confidence intervals, Task 6

- **`tools/zeroshot_baseline.py`** (new) + **`tools/zeroshot_prompts.json`** (DRAFT prompts, one per class;
  file SHA-256 recorded per run). YOLO-World through the same `TiledDetector` + `ultralytics_predict_fn`
  path as `08 --weights`, tile/overlap from dataset.yaml, output = 08's `detections.npz` cache format
  (score it with `08 --cache`), `zeroshot_manifest.json` (weights path/size/SHA-256, prompts, versions).
  Uses the LARGEST local `*-worldv2.pt`; downloads only with an explicit `--download yolov8x-worldv2.pt`.
  `set_classes()` needs Ultralytics' CLIP text encoder (one-off fetch on first use). Refuses `test*` folders
  without `--final_evaluation`. Its threshold is calibrated on synthetic validation like the trained model's.
  **Not run yet** (prompts to be confirmed; no weights downloaded).
- **Bootstrap CIs** (`ewads_hier_eval.bootstrap_ci`, used by `08`): image-level percentile bootstrap,
  `--bootstrap 1000` (default; 0 = off), `--bootstrap_seed 2026`; CIs for mAP@.5, mAP@.5:.95, recall, tier-1
  recall, FP share, site recall (`eval_report.json → ci95`), and a `ci95` on every pre-registered target row
  (on that target's own frames/filters). `per_class_n` (instance count per class) added to the report.
- `08` closes its detection cache after loading (Windows file lock).
- **Camera-mismatch ablation (6.2): not started** — optional, waits for your go-ahead.
- Tests: new `tests/test_bootstrap_zeroshot.py`.

## 19. 2026-10-08 — camera-mismatch ablation (Task 6.2)

**Question:** does rendering with the TRUE camera geometry matter? Two arms, identical except the camera:
`true` = Phantom 4 Pro (8.8 mm, 24 mm-equiv) and `mis` = same sensor + resolutions with an 18.333 mm lens
(50 mm-equivalent; objects render ~2.1x larger per altitude, so size-vs-altitude statistics and the
resolvability gate are wrong). Both trained with the same short schedule, both evaluated on the SAME
`main/heldout` at the true geometry, compared with a paired bootstrap.

Code: `02 --camera_override p4p_mismatch_50mm_equiv` (or `--override_focal_mm/--override_sensor_width_mm`;
every scene records `camera_profile`, `focal_mm`, `sensor_width_mm`; the log flags "ABLATION: not the real
camera"); `05 --max_epochs N` (caps every stage; in provenance); `ewads_hier_eval.paired_bootstrap_diff`;
`tools/compare_evals.py`. Verified in Blender: GSD × 0.480 (= 8.8/18.333) for the same scene.

Run order (workstation; after S1–S3 of the synthetic-only runbook):
```bash
for arm in true mis; do
  cam=$([ $arm = mis ] && echo "--camera_override p4p_mismatch_50mm_equiv")
  blenderproc run 02_blenderproc_synthetic_v2.py --n_scenes 1000 --environment all --seed 21 \
      --start_id 700000 $cam --output_dir ./data/raw/abl_$arm
  python tools/make_synth_splits.py --synthetic_dir data/raw/abl_$arm --out_dir data/splits/abl_$arm \
      --heldout_frac 0 --no_loeo
  python 03b_tile_dataset.py --input_dir data/splits/abl_$arm/main --output_dir data/tiled/abl_$arm --splits synthetic val
  python 03_augmentation_pipeline.py --input_dir data/tiled/abl_$arm --output_dir data/augmented/abl_$arm --rfs_t 0.1
  python 05_train_rtdetr.py --dataset_yaml data/augmented/abl_$arm/dataset.yaml --output_dir runs/abl_$arm \
      --seed 0 --max_epochs 10 --no_polish
  # each arm's threshold on ITS OWN validation (what you would do without knowing the geometry is wrong)
  python 08_tiled_eval.py --weights $(cat runs/abl_$arm/final_model.txt) --data_dir data/splits/abl_$arm/main/val \
      --calibrate --targets "" --bootstrap 0 --out_dir eval/abl_${arm}_val
done
```
Scoring on `main/heldout` and `tools/compare_evals.py` (→ `results_bundle/ablation_camera.json`) belong to
final evaluation (Task 8). Tests: new `tests/test_ablation.py`.

## 20. 2026-10-08 — dataset statistics, Task 7

- **`tools/dataset_stats.py`** (new) → `results_bundle/dataset_stats.json` (+ `dataset_stats_objaverse_attributions.csv`).
  Counted from files only: scenes vs frames per environment / altitude profile / capture mode, labelled
  instances per class (total and per environment), objects per native pixel-size bucket (`<8`, 8-16 … 128+),
  negative-scene share, frames without labels, the renderer's visible-but-unlabelled counts, tiles per split
  (+ empty tiles) from `tile_index.jsonl`, per-split frames/scenes from `split_manifest.json`, assets per
  class (Objaverse **reviewed** = in a review round's `index.json`, else flagged UNREVIEWED; procedural;
  rejected), distractors, and licences: ground textures and HDRIs from their manifests, procedural = generated,
  Objaverse per-object licence from the LOCAL metadata cache, then from the repo snapshot
  `assets/models/_licences/objaverse_attributions.csv` (no network; anything in neither = "unknown").
- **Licence finding (from the laptop's Objaverse cache, 219 reviewed models):** CC BY 192, CC BY-SA 8,
  **CC BY-NC-SA 14, CC BY-NC 5**. CC BY* → attribution (creator + link) required wherever models or renders
  are published; the attribution list is in the snapshot CSV. **19 NonCommercial models: check with counsel
  before a GDRFA / agency deployment** — excluding them is a model-review decision (move to `_rejected`,
  regenerate piles) and must happen BEFORE rendering.
- Run it after S2–S6 of the synthetic-only runbook (it needs the renders and tiles to exist):
  `python tools/dataset_stats.py --extra_render_dirs data/raw/synthetic_valseq --tiled_dirs data/tiled/main`
- Tests: new `tests/test_dataset_stats.py`.

## 21. 2026-10-08 — licence exclusion (round 3a) and the Task 8 final-evaluation plan

**NonCommercial models excluded** (user instruction): the 19 Objaverse models licensed CC BY-NC / CC BY-NC-SA
were moved to `assets/models/_rejected/<class>/` — decisions `tools/reviews/round3a_licence_nc.json`
(`apply_model_review.py` now also accepts `reject_stems`), every move logged with its reason in
`assets/models/_review/review_log.jsonl`. The 30 `ewaste_pile` assets had been built from a pool that
included those models and do not record their ingredients, so all 30 were moved to
`assets/models/_rejected/ewaste_pile__built_from_pool_with_nc_models/` — **regenerate piles** (runbook
Phase 3 already does, after review round 3). Remaining Objaverse licences: CC BY 192, CC BY-SA 8 (attribution
list: `assets/models/_licences/objaverse_attributions.csv`). **Distractors fetched on the workstation are
also Objaverse models: run `tools/dataset_stats.py` after fetching and reject any NonCommercial ones the same
way before rendering.** Thin classes after the exclusion: hard_drive 1, power_supply 2 Objaverse models —
the procedural generator (Phase 3) fills them.

**Task 8 — final evaluation. Only after the user types "run final evaluation" and the targets are FROZEN.**
Output paths below are the ones `tools/results_bundle_config.json` expects. `<conf>` = the threshold each
model calibrated on synthetic validation (S6 / the fold's val); `<best>` = `$(cat runs/<winner>/final_model.txt)`.
```bash
python tools/freeze_targets.py --targets targets_v2.json --check            # must print FROZEN
# synthetic held-out, single model (all objects + envelope, class/group/binary, altitude x size, CIs)
python 08_tiled_eval.py --weights <best> --data_dir data/splits/main/heldout --conf <conf> \
    --meta data/splits/main/heldout/scene_index.jsonl --targets_dataset synthetic_heldout --out_dir eval/final/synthetic_heldout
# synthetic held-out, evidence pipeline on held-out sequences (thresholds frozen from val_seq, S8)
python 11_evidence_pipeline.py --weights <A> <B> --data_dir data/splits/main/heldout_seq --tta flips \
    --stage_thresholds abl/val_seq/<run>/stage_thresholds.json --out_dir abl/heldout_seq
python 08_tiled_eval.py --data_dir data/splits/main/heldout_seq --cache abl/heldout_seq/<run>/detections_multiview.npz \
    --conf <multiview threshold> --meta data/splits/main/heldout_seq/scene_index.jsonl \
    --targets_dataset synthetic_heldout --out_dir eval/final/synthetic_heldout_evidence
# LOEO folds
for env in desert european_urban european_rural; do
  python 08_tiled_eval.py --weights $(cat runs/loeo_$env/final_model.txt) --data_dir data/splits/loeo_$env/heldout \
      --conf <fold conf> --meta data/splits/loeo_$env/heldout/scene_index.jsonl --targets_dataset sim_to_sim_loeo \
      --out_dir eval/final/loeo_$env
done
# real test set: best model, zero-shot baseline (its own val-calibrated conf), evidence pass
python 08_tiled_eval.py --weights <best> --data_dir data/test --conf <conf> --meta data/test/altitudes.csv \
    --cache eval/final/real_test_best/detections.npz --targets_dataset sim_to_real --out_dir eval/final/real_test_best
python tools/zeroshot_baseline.py --data_dir data/splits/main/val --out_dir eval/zeroshot_val --dataset_yaml <winning yaml>
python 08_tiled_eval.py --data_dir data/splits/main/val --cache eval/zeroshot_val/detections.npz --calibrate --targets "" --out_dir eval/zeroshot_val
python tools/zeroshot_baseline.py --data_dir data/splits/main/heldout --out_dir eval/final/synthetic_heldout_zeroshot --dataset_yaml <winning yaml>
python 08_tiled_eval.py --data_dir data/splits/main/heldout --cache eval/final/synthetic_heldout_zeroshot/detections.npz \
    --conf <zero-shot conf> --meta data/splits/main/heldout/scene_index.jsonl --targets_dataset synthetic_heldout \
    --out_dir eval/final/synthetic_heldout_zeroshot
python tools/zeroshot_baseline.py --data_dir data/test --final_evaluation --out_dir eval/final/real_test_zeroshot --dataset_yaml <winning yaml>
python 08_tiled_eval.py --data_dir data/test --cache eval/final/real_test_zeroshot/detections.npz --conf <zero-shot conf> \
    --meta data/test/altitudes.csv --targets_dataset sim_to_real --out_dir eval/final/real_test_zeroshot
python 11_evidence_pipeline.py --weights <A> <B> --images data/test/images --tta flips \
    --stage_thresholds abl/val_seq/<run>/stage_thresholds.json --report_conf <multiview threshold> --out_dir abl/real_evidence
# camera ablation on the same held-out set
for arm in true mis; do
  python 08_tiled_eval.py --weights $(cat runs/abl_$arm/final_model.txt) --data_dir data/splits/main/heldout \
      --conf <arm conf> --targets "" --bootstrap 0 --out_dir eval/abl_${arm}_heldout
done
python tools/compare_evals.py --data_dir data/splits/main/heldout --a_cache eval/abl_true_heldout/detections.npz \
    --a_conf <true conf> --a_name true_camera --b_cache eval/abl_mis_heldout/detections.npz --b_conf <mis conf> \
    --b_name mismatch_50mm --meta data/splits/main/heldout/scene_index.jsonl --out results_bundle/ablation_camera.json
# H2, H3, dataset statistics
python tools/gsd_validation.py --photos data/h2/photos --heights data/h2/heights.csv --cvat_xml data/h2/annotations.xml
python tools/tamper_experiment.py --n_trials 200 --out_dir results_bundle
python tools/dataset_stats.py --extra_render_dirs data/raw/synthetic_valseq data/raw/synthetic_heldoutseq --tiled_dirs data/tiled/main
# Jetson files (you run them): copy engine_decision.json and benchmark_report.json to runs/jetson/
```

## 22. 2026-10-08 — results bundle, Task 9

- **`tools/export_results_bundle.py`** (new) + **`tools/results_bundle_config.json`** (input paths = §21's
  outputs). Copies every result into `results_bundle/` (targets + lock, freeze record, eval reports +
  per-class CSVs, decision records, H2/H3/dataset stats with their CSVs, split manifest + lists), writes the
  figures (3 synthetic held-out frames with ground truth, one per environment; the real-test frame with the
  best model's predictions at its val-calibrated threshold + the assessability statement from
  `ewads_survey.resolvability`; the first evidence review crop; the H2 scatter), `environment.txt`,
  `git_commit.txt` (+ `code_manifest.sha256` when not a git repo), `SUMMARY.md` (one table per hypothesis:
  target, measured, 95% CI, verdict, source file — values and verdicts copied, never recomputed; missing
  inputs = NOT RUN and listed; freeze status first) and `MANIFEST.sha256` over every file.
- `tools/make_synth_splits.py`: `--heldout_seq_dir` → `main/heldout_seq` (held-out sequences for scoring 11).
- Tests: new `tests/test_export_bundle.py` (a full fake project built with the real tools; every SUMMARY
  number/verdict equals its source; NOT RUN / missing / LOEO 'incomplete'; figures; manifest verifies);
  `tests/test_synth_splits.py` + heldout_seq.

## 23. 2026-10-16 — review round 3 (draft decisions)

- **`tools/reviews/round3.json`** (new): draft decisions from reading the 11 round-3 sheets. Rejects: `single_cable` tile 23 (a speck, no visible cable), `plastic_casing` tile 33 (broken fragment). Everything else kept. Not yet applied; the user confirms before `--apply`.
- Run the dry run and `--apply` **on the workstation**: the 40-model procedural classes and `assets/models/_review_v3/index.json` exist only there.
- Licence round 3a (`round3a_licence_nc.json`) is already applied: all 19 stems are in `_rejected/`.
- Not verified: the thumbnails do not show size. Faint single_cable tiles (4, 10, 11, 14, 22, 29, 30, 32, 35) and sparse cable_bundle tiles (3, 24) need a GLB bounding-box check before any decision.
- The sheets show only models NOT already in review v2 (`--skip_reviewed`), so the counts on the sheets are the new models only. Final class counts = laptop count + the sheet count (e.g. hard_drive 1 + 40, circuit_board 10 + 2).

## 24. 2026-11-05 — ATHAR completion layer (`athar/`, branch `claude/athar-completion`)

Status of the three "designed, not built" parts, and what was added: `../docs/ATHAR_STATUS.md`.
After an 11 run (sign the head first, as before):
```bash
python -m athar.verify <run_dir> --trusted_fingerprint sha256:... --require_signature [--media_root <flight media>]
python -m athar.dashboard <run_dir> --reviewer "<name>" --trusted_fingerprint sha256:...   # http://127.0.0.1:8765/
python -m athar.agent draft <run_dir> --reviewer "<name>" --backend ollama --model <local model>   # or --backend stub
python tools/sign_chain_head.py sign --log <run_dir>/review_decisions.jsonl --private_key <reviewer key, outside repo>
python -m athar.verify <run_dir> --trusted_fingerprint <evidence key> --review_fingerprint <reviewer key> --require_signature
```
Tests: `pytest -m "not gpu and not needs_data and not needs_torch"` (CI: ubuntu + windows, CPU).
No thresholds, targets or scientific code were changed.
