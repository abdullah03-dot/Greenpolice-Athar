"""
E-Waste Detection Pipeline — Phase 4: AdaIN Style Transfer (Synthetic → Real)
==============================================================================
Applies Adaptive Instance Normalization (AdaIN) to transfer the visual style
of real drone captures onto synthetic BlenderProc images.

Why AdaIN over CycleGAN:
  • Trains in hours, not days — critical for iteration speed
  • Achieves ~75% of CycleGAN quality for this use case
  • No discriminator instability (pure feed-forward at inference)
  • Can transfer multiple real-image styles per synthetic image (ensemble)

Architecture:
  • VGG-19 encoder (pretrained, frozen) — extracts style + content features
  • Decoder (trained from scratch) — reconstructs stylized image
  • AdaIN layer: transfers mean + std of real image's feature statistics
    to the content features of the synthetic image
  • Loss: content loss (relu4_1) + style loss (relu1_1, relu2_1, relu3_1, relu4_1)

Training requirement:
  • Content images: synthetic BlenderProc renders (unlabeled OK)
  • Style images:   real drone captures (unlabeled, 200+ recommended)
  • No paired data required

RESOLUTION (changed 2026-10-02): the original squashed every image to 512x512 and, at
inference, stylised that and LANCZOS-upscaled it back to the original size.  On a 4K frame
that is an 8x blur: every small object was smeared before the detector ever saw it.  Now:
  - training uses native-resolution RANDOM CROPS (no resizing), so the decoder learns the
    texture statistics of real pixels at the real ground-sampling distance;
  - transfer runs at NATIVE resolution on the TILES produced by 03b_tile_dataset.py
    (fully-convolutional, so any size works; padded to a multiple of 8 and cropped back);
  - frames larger than --max_side (default 1536) are refused: tile them first (03b).
Order: BlenderProc -> 03b tiling -> 04 AdaIN (on tiles) -> 03 augmentation -> 05 training.

Usage:
  conda activate ewads
  pip install torch torchvision pillow tqdm

  # Train AdaIN model (run on laptop — fits in 8GB VRAM):
  python 04_adain_transfer.py train \\
    --content_dir ./data/tiled/synthetic \\
    --style_dir   ./data/raw/real/images \\
    --output_dir  ./models/adain \\
    --epochs 20 \\
    --batch_size 4

  # Apply style transfer to synthetic dataset:
  python 04_adain_transfer.py transfer \\
    --model_path  ./models/adain/adain_best.pth \\
    --input_dir   ./data/tiled/synthetic \\
    --style_dir   ./data/raw/real/images \\
    --output_dir  ./data/tiled_styled/synthetic \\
    --alpha 0.75         ← style strength (0=content only, 1=full style)
    --n_styles 3         ← apply N different real styles per synthetic image

  # After this, copy/symlink tiled/real and tiled/val next to tiled_styled/synthetic and run
  #   python 03_augmentation_pipeline.py --input_dir ./data/tiled_styled

VRAM budget on RTX 3060 (8GB):
  Training with batch_size=4: ~5.5GB  ✓
  Training with batch_size=8: ~9GB    ✗ (OOM)
  Inference: ~2GB (can batch 32+)     ✓
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np


# ─── Device setup ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── VGG-19 Feature Extractor ─────────────────────────────────────────────────
class VGGEncoder(nn.Module):
    """
    VGG-19 encoder, pretrained on ImageNet, frozen.
    Returns intermediate feature maps for style + content loss computation.
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        features = vgg.features

        # Slice at layer boundaries for multi-scale style loss
        self.slice1 = nn.Sequential(*[features[i] for i in range(2)])   # relu1_1
        self.slice2 = nn.Sequential(*[features[i] for i in range(2, 7)])  # relu2_1
        self.slice3 = nn.Sequential(*[features[i] for i in range(7, 12)]) # relu3_1
        self.slice4 = nn.Sequential(*[features[i] for i in range(12, 21)])# relu4_1

        # Freeze — we only train the decoder
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4


# ─── AdaIN Layer ──────────────────────────────────────────────────────────────
def adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    """
    Adaptive Instance Normalization.
    Transfers mean and std of style features to content features.

    content_feat: (B, C, H, W) — synthetic image features (relu4_1)
    style_feat:   (B, C, H, W) — real drone image features (relu4_1)

    Returns normalized content features with style statistics.
    """
    eps = 1e-5

    # Content statistics
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std  = content_feat.std( dim=[2, 3], keepdim=True) + eps

    # Style statistics (target)
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std  = style_feat.std( dim=[2, 3], keepdim=True) + eps

    # Normalize content, then scale/shift to style statistics
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean


# ─── Decoder ──────────────────────────────────────────────────────────────────
class AdaINDecoder(nn.Module):
    """
    Learned decoder: maps AdaIN-transformed features back to pixel space.
    Mirror image of VGG encoder, with upsampling instead of pooling.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Full AdaIN Model ──────────────────────────────────────────────────────────
class AdaINModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = VGGEncoder()
        self.decoder = AdaINDecoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, feat):
        return self.decoder(feat)

    def transfer(self, content: torch.Tensor, style: torch.Tensor,
                 alpha: float = 1.0) -> torch.Tensor:
        """
        Transfer style from real image to synthetic image.

        alpha: style strength (0=keep content, 1=full style transfer)
        """
        content_feats = self.encoder(content)
        style_feats   = self.encoder(style)

        # AdaIN at relu4_1 (deepest features, strongest style signal)
        t = adain(content_feats[-1], style_feats[-1])

        # Interpolate between content and stylized (alpha blending)
        t = alpha * t + (1 - alpha) * content_feats[-1]

        return self.decoder(t)


# ─── Loss Functions ───────────────────────────────────────────────────────────
def content_loss(feat_stylized, feat_adain):
    """Content loss: L2 between stylized output features and AdaIN target."""
    return F.mse_loss(feat_stylized, feat_adain)


def style_loss(encoder: VGGEncoder, stylized: torch.Tensor,
               style: torch.Tensor) -> torch.Tensor:
    """
    Style loss: match mean and std of feature maps at multiple VGG layers.
    """
    feats_stylized = encoder(stylized)
    feats_style    = encoder(style)

    loss = 0.0
    for f_s, f_ref in zip(feats_stylized, feats_style):
        mean_s, std_s   = f_s.mean(dim=[2,3]),   f_s.std(dim=[2,3])
        mean_r, std_r   = f_ref.mean(dim=[2,3]), f_ref.std(dim=[2,3])
        loss += F.mse_loss(mean_s, mean_r) + F.mse_loss(std_s, std_r)

    return loss


# ─── Datasets ─────────────────────────────────────────────────────────────────
class UnpairedImageDataset(Dataset):
    """
    Loads content (synthetic) and style (real) images unpaired.
    Each __getitem__ returns one content image and one randomly sampled style image.
    """

    def __init__(self, content_dir: Path, style_dir: Path,
                 image_size: int = 512):
        self.content_files = self._find_images(content_dir)
        self.style_files   = self._find_images(style_dir)
        self.transform     = self._build_transform(image_size)

        print(f"  Content images : {len(self.content_files)}")
        print(f"  Style images   : {len(self.style_files)}")

        if len(self.style_files) < 50:
            print(f"  ⚠  WARNING: Only {len(self.style_files)} style images.")
            print(f"     AdaIN quality improves significantly with 200+ real captures.")

    @staticmethod
    def _find_images(directory: Path) -> list:
        # case-insensitive: DJI writes .JPG, and glob is case-sensitive on Linux
        return sorted(p for p in directory.rglob("*")
                      if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    @staticmethod
    def _build_transform(image_size: int) -> T.Compose:
        # Native-resolution random crop: NO resize, so texture scale matches inference.
        return T.Compose([
            T.RandomCrop(image_size, pad_if_needed=True, padding_mode="reflect"),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.content_files)

    def __getitem__(self, idx):
        content = Image.open(self.content_files[idx]).convert("RGB")
        style   = Image.open(random.choice(self.style_files)).convert("RGB")
        return self.transform(content), self.transform(style)


# ─── Training ─────────────────────────────────────────────────────────────────
def train_adain(content_dir: Path, style_dir: Path, output_dir: Path,
                epochs: int = 20, batch_size: int = 4,
                lr: float = 1e-4, image_size: int = 512,
                style_weight: float = 10.0):

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  AdaIN Style Transfer — Training")
    print(f"  Device     : {DEVICE}")
    print(f"  Image size : {image_size}")
    print(f"  Batch size : {batch_size}")
    print(f"  Epochs     : {epochs}")
    print(f"  Style wt   : {style_weight}")
    print(f"{'='*60}\n")

    # Model
    model = AdaINModel().to(DEVICE)
    # Only decoder has learnable parameters
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )

    # Dataset
    dataset    = UnpairedImageDataset(content_dir, style_dir, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2,
                            pin_memory=DEVICE.type == "cuda", drop_last=True)

    history = {"content_loss": [], "style_loss": [], "total_loss": []}
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_c_loss = 0.0
        epoch_s_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for content_imgs, style_imgs in pbar:
            content_imgs = content_imgs.to(DEVICE)
            style_imgs   = style_imgs.to(DEVICE)

            # Forward pass
            content_feats = model.encoder(content_imgs)
            style_feats   = model.encoder(style_imgs)

            # AdaIN transform
            t = adain(content_feats[-1], style_feats[-1])

            # Decode
            stylized = model.decoder(t)

            # NO clamp here.  The decoder works in ImageNet-NORMALISED space (about -2.1..2.6);
            # the old `clamp(0, 1)` confined it to a narrow grey band and zeroed the gradient
            # outside it, so the decoder never learned dark or saturated colours (washed-out output).

            # Compute losses
            stylized_feats = model.encoder(stylized)
            c_loss = content_loss(stylized_feats[-1], t)
            s_loss = style_loss(model.encoder, stylized, style_imgs)
            total  = c_loss + style_weight * s_loss

            # Backward
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            optimizer.step()

            epoch_c_loss += c_loss.item()
            epoch_s_loss += s_loss.item()

            pbar.set_postfix({
                "c": f"{c_loss.item():.4f}",
                "s": f"{s_loss.item():.4f}",
            })

        scheduler.step()

        avg_c = epoch_c_loss / len(dataloader)
        avg_s = epoch_s_loss / len(dataloader)
        avg_t = avg_c + style_weight * avg_s

        history["content_loss"].append(avg_c)
        history["style_loss"].append(avg_s)
        history["total_loss"].append(avg_t)

        print(f"  Epoch {epoch+1:3d} | content: {avg_c:.4f} | "
              f"style: {avg_s:.4f} | total: {avg_t:.4f}")

        # Save best model
        if avg_t < best_loss:
            best_loss = avg_t
            ckpt = {
                "epoch": epoch,
                "decoder_state": model.decoder.state_dict(),
                "loss": best_loss,
                "config": {
                    "image_size": image_size,
                    "style_weight": style_weight,
                },
            }
            torch.save(ckpt, output_dir / "adain_best.pth")
            print(f"    ✓ Best model saved (loss={best_loss:.4f})")

        # Checkpoint every 5 epochs (the CURRENT decoder, not a copy of the best one)
        if (epoch + 1) % 5 == 0:
            torch.save({"epoch": epoch, "decoder_state": model.decoder.state_dict(), "loss": avg_t,
                        "config": {"image_size": image_size, "style_weight": style_weight}},
                       output_dir / f"adain_epoch{epoch+1:03d}.pth")

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Training complete. Best loss: {best_loss:.4f}")
    print(f"  Model saved: {output_dir / 'adain_best.pth'}")


# ─── Inference / Transfer ──────────────────────────────────────────────────────
class AdaINInferencer:
    """
    Load trained AdaIN model and apply style transfer to a directory of images.
    Preserves original YOLO label files (image content changes, labels stay).
    """

    def __init__(self, model_path: Path, alpha: float = 0.75, protect_objects: bool = True,
                 object_alpha: float = 0.35):
        self.alpha = alpha
        self.protect_objects, self.object_alpha = protect_objects, object_alpha
        self.fidelity = []          # per labelled object: (long side px, detail ratio)

        # Load model
        ckpt = torch.load(str(model_path), map_location=DEVICE, weights_only=False)
        self.model = AdaINModel().to(DEVICE)
        self.model.decoder.load_state_dict(ckpt["decoder_state"])
        self.model.eval()

        image_size = ckpt.get("config", {}).get("image_size", 512)

        norm = [T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.preprocess = T.Compose(norm)                       # content: native resolution
        self.style_crop = T.Compose([T.RandomCrop(image_size, pad_if_needed=True,
                                                  padding_mode="reflect")] + norm)
        self.max_side = 1536

        self.denorm = T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225],
        )

        print(f"  AdaIN model loaded from {model_path}  (native-resolution mode)")
        print(f"  Style strength (alpha): {alpha}")

    @staticmethod
    def _boxes_px(label_path: Optional[Path], w: int, h: int) -> np.ndarray:
        if not label_path or not label_path.exists():
            return np.zeros((0, 4))
        rows = []
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        return np.array(rows).reshape(-1, 4)

    def _object_mask(self, boxes: np.ndarray, w: int, h: int) -> np.ndarray:
        """Soft mask: 1 on each labelled object (box grown by 25% + 3 px), feathered edges."""
        import cv2
        m = np.zeros((h, w), np.float32)
        for x1, y1, x2, y2 in boxes:
            gx, gy = 0.125 * (x2 - x1) + 3, 0.125 * (y2 - y1) + 3
            cv2.rectangle(m, (int(max(0, x1 - gx)), int(max(0, y1 - gy))),
                          (int(min(w - 1, x2 + gx)), int(min(h - 1, y2 + gy))), 1.0, -1)
        return cv2.GaussianBlur(m, (0, 0), 3) if len(boxes) else m

    @staticmethod
    def _detail(gray: np.ndarray, box) -> float:
        import cv2
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        crop = gray[y1:max(y1 + 2, y2), x1:max(x1 + 2, x2)]
        if crop.size < 4:
            return float("nan")
        gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0); gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1)
        return float(np.mean(np.hypot(gx, gy)))

    @torch.no_grad()
    def transfer_image(self, content_path: Path, style_path: Path,
                       out_path: Path, original_size: Optional[tuple] = None,
                       label_path: Optional[Path] = None):
        """
        Transfer style from style_path onto content_path at native resolution.
        With protect_objects, labelled objects keep (1 - object_alpha) of their original pixels:
        AdaIN reconstructs the image from 1/8-resolution features, which is exactly the scale at
        which a 16 px object is reduced to 2x2 cells and can be painted over.  The background
        (where the domain gap mostly lives: ground, light, sensor noise) gets the full style.
        Detail retention per object (Sobel energy stylised / original) is recorded either way.
        """
        content_pil = Image.open(content_path).convert("RGB")
        style_pil   = Image.open(style_path).convert("RGB")

        orig_w, orig_h = content_pil.size
        if max(orig_w, orig_h) > self.max_side:
            raise ValueError(
                f"{content_path.name} is {orig_w}x{orig_h}: stylising a frame this large at native "
                f"resolution does not fit in VRAM and downscaling it would erase small objects. "
                f"Cut it into tiles with 03b_tile_dataset.py first (or raise --max_side).")

        content_t = self.preprocess(content_pil).unsqueeze(0).to(DEVICE)
        style_t   = self.style_crop(style_pil).unsqueeze(0).to(DEVICE)

        # encoder downsamples 8x and the decoder upsamples 8x -> pad to a multiple of 8
        ph, pw = (-orig_h) % 8, (-orig_w) % 8
        if ph or pw:
            content_t = F.pad(content_t, (0, pw, 0, ph), mode="reflect")

        stylized_t = self.model.transfer(content_t, style_t, alpha=self.alpha)
        stylized_t = self.denorm(stylized_t.squeeze(0)).clamp(0, 1)[:, :orig_h, :orig_w]

        stylized_arr = (stylized_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        boxes = self._boxes_px(label_path, orig_w, orig_h)
        orig_arr = np.asarray(content_pil, dtype=np.uint8)
        if self.protect_objects and len(boxes):
            m = self._object_mask(boxes, orig_w, orig_h)[..., None] * (1.0 - self.object_alpha)
            stylized_arr = (stylized_arr * (1 - m) + orig_arr * m).round().clip(0, 255).astype(np.uint8)
        if len(boxes):
            g0 = np.asarray(Image.fromarray(orig_arr).convert("L"), np.float32)
            g1 = np.asarray(Image.fromarray(stylized_arr).convert("L"), np.float32)
            for b in boxes:
                d0 = self._detail(g0, b)
                if d0 and d0 > 1.0:                       # skip flat boxes (ratio meaningless)
                    self.fidelity.append((float(max(b[2] - b[0], b[3] - b[1])), self._detail(g1, b) / d0))
        stylized_pil = Image.fromarray(stylized_arr)           # already native size: no resampling

        out_path.parent.mkdir(parents=True, exist_ok=True)
        stylized_pil.save(str(out_path), quality=95)

    def write_fidelity_report(self, output_dir: Path) -> dict:
        """Median detail retention per object size.  < ~0.6 on the 8-32 px buckets means the
        style transfer is erasing small objects: lower --alpha / --object_alpha, or drop AdaIN."""
        if not self.fidelity:
            return {}
        a = np.array(self.fidelity)
        rep = {"protect_objects": self.protect_objects, "alpha": self.alpha,
               "object_alpha": self.object_alpha, "n_objects": int(len(a)), "buckets": {}}
        print(f"\n  Object detail retained after stylisation (Sobel energy ratio, 1.0 = unchanged):")
        for lo, hi in ((8, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)):
            sel = a[(a[:, 0] >= lo) & (a[:, 0] < hi), 1]
            sel = sel[np.isfinite(sel)]
            if len(sel):
                lab = f"{lo}-{int(hi)}px" if hi < 1e8 else f">={lo}px"
                rep["buckets"][lab] = {"n": int(len(sel)), "median": float(np.median(sel)),
                                       "p10": float(np.percentile(sel, 10))}
                flag = "  <- small objects being erased" if (hi <= 32 and np.median(sel) < 0.6) else ""
                print(f"    {lab:>9}: median {np.median(sel):.2f}  p10 {np.percentile(sel, 10):.2f}  (n={len(sel)}){flag}")
        (output_dir / "adain_fidelity.json").write_text(json.dumps(rep, indent=2))
        return rep

    def transfer_directory(self, input_dir: Path, style_dir: Path,
                           output_dir: Path, n_styles: int = 1):
        """
        Apply style transfer to all images in input_dir.
        YOLO label files are copied unchanged (bboxes are coordinate-only).
        n_styles: apply N different style images per content image (creates N outputs)
        """
        img_dir = input_dir / "images"
        lbl_dir = input_dir / "labels"
        out_img = output_dir / "images"
        out_lbl = output_dir / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        # case-insensitive: DJI writes .JPG, and glob is case-sensitive on Linux
        style_files = sorted(p for p in style_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})

        if not style_files:
            print(f"[ERROR] No style images found in {style_dir}")
            return

        image_files = (list(img_dir.glob("*.jpg")) +
                       list(img_dir.glob("*.jpeg")) +
                       list(img_dir.glob("*.png")))

        print(f"\n  Applying AdaIN style transfer")
        print(f"  Content images : {len(image_files)}")
        print(f"  Style pool     : {len(style_files)}")
        print(f"  Styles/image   : {n_styles}")
        print(f"  Alpha          : {self.alpha}")
        print(f"  Output         : {output_dir}")

        for img_path in tqdm(image_files, desc="  Transferring"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")

            for style_i in range(n_styles):
                style_path = random.choice(style_files)
                suffix = f"_s{style_i}" if n_styles > 1 else ""
                out_name = img_path.stem + suffix
                out_img_path = out_img / f"{out_name}.jpg"
                out_lbl_path = out_lbl / f"{out_name}.txt"

                # Transfer image
                try:
                    self.transfer_image(img_path, style_path, out_img_path, label_path=lbl_path)
                except Exception as e:
                    print(f"  [WARN] Failed {img_path.name}: {e}")
                    continue

                # Copy label (unchanged — bboxes are coordinate-only)
                if lbl_path.exists():
                    import shutil
                    shutil.copy2(lbl_path, out_lbl_path)
                else:
                    out_lbl_path.touch()

        self.write_fidelity_report(output_dir)
        print(f"\n  Transfer complete: {output_dir}")
        print(f"  Next step: run 03_augmentation_pipeline.py on the folder that contains {output_dir.name}/")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AdaIN style transfer for synthetic-to-real domain adaptation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_p = subparsers.add_parser("train", help="Train AdaIN model")
    train_p.add_argument("--content_dir",  type=str, required=True)
    train_p.add_argument("--style_dir",    type=str, required=True)
    train_p.add_argument("--output_dir",   type=str, default="./models/adain")
    train_p.add_argument("--epochs",       type=int, default=20)
    train_p.add_argument("--batch_size",   type=int, default=4)
    train_p.add_argument("--lr",           type=float, default=1e-4)
    train_p.add_argument("--image_size",   type=int, default=512)
    train_p.add_argument("--style_weight", type=float, default=10.0)

    # Transfer subcommand
    xfer_p = subparsers.add_parser("transfer", help="Apply style transfer")
    xfer_p.add_argument("--model_path",   type=str, required=True)
    xfer_p.add_argument("--input_dir",    type=str, required=True)
    xfer_p.add_argument("--style_dir",    type=str, required=True)
    xfer_p.add_argument("--output_dir",   type=str, required=True)
    xfer_p.add_argument("--alpha",        type=float, default=0.75,
                        help="Style strength (0=content, 1=full style)")
    xfer_p.add_argument("--max_side",     type=int, default=1536,
                        help="refuse inputs larger than this (tile them with 03b first)")
    xfer_p.add_argument("--n_styles",     type=int, default=1,
                        help="Number of different styles per content image")
    xfer_p.add_argument("--no_protect_objects", action="store_true",
                        help="stylise labelled objects fully (default keeps 65%% of their pixels)")
    xfer_p.add_argument("--object_alpha", type=float, default=0.35,
                        help="style strength INSIDE labelled objects when protecting them")

    args = parser.parse_args()

    if args.command == "train":
        train_adain(
            content_dir=Path(args.content_dir) / "images",
            style_dir=Path(args.style_dir),
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            style_weight=args.style_weight,
        )

    elif args.command == "transfer":
        inferencer = AdaINInferencer(
            model_path=Path(args.model_path),
            alpha=args.alpha,
            protect_objects=not args.no_protect_objects,
            object_alpha=args.object_alpha,
        )
        inferencer.max_side = args.max_side
        inferencer.transfer_directory(
            input_dir=Path(args.input_dir),
            style_dir=Path(args.style_dir),
            output_dir=Path(args.output_dir),
            n_styles=args.n_styles,
        )


if __name__ == "__main__":
    main()
