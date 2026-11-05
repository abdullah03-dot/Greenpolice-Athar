#!/usr/bin/env python3
"""
Vet real drone photos before they become AdaIN style images (04_adain_transfer.py --style_dir).

AdaIN copies the STATISTICS of whatever you feed it.  Blurry, over-exposed, oblique or duplicated
photos -- or photos from altitudes the renderer never produces -- teach the synthetic data the
wrong "real".  And a style photo that is also in your real TEST set leaks the test domain into
training.  This tool checks each photo and writes a clean, usable list.

Checks per photo
  size        >= the training tile (default 768) on both sides, else AdaIN crops are padded
  nadir       DJI XMP GimbalPitchDegree within 10 deg of -90 (oblique views have a different
              look AND a GSD that changes across the frame)
  altitude    DJI XMP RelativeAltitude -> GSD -> which renderer altitude profile it represents
  sharpness   variance of the Laplacian on the native-resolution centre crop (motion / focus blur)
  exposure    share of clipped highlights / crushed shadows
  duplicates  64-bit dHash; near-identical consecutive shots are kept once
  leakage     any photo whose dHash matches an image under --exclude_dir (your real val/test)

Usage
  python tools/check_style_photos.py --photos D:/drone/raw_flights --exclude_dir data/real_test/images \\
         --out_csv style_check.csv --copy_usable data/style_photos
Then:  python 04_adain_transfer.py train --style_dir data/style_photos ...
"""
import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ewads_survey as S  # noqa: E402

PROFILES = {"low_evidence": (3, 15), "mid_confirmation": (15, 40), "high_surveillance": (40, 80)}
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def dhash(gray: np.ndarray) -> int:
    import cv2
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def analyse(path: Path, tile: int, cam: S.Camera) -> dict:
    import cv2
    r = {"file": str(path)}
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        r["problems"] = "unreadable"
        return r
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r.update(width=w, height=h, dhash=dhash(gray))
    cy, cx, k = h // 2, w // 2, min(512, h // 2, w // 2)
    crop = gray[cy - k:cy + k, cx - k:cx + k]
    r["sharpness"] = round(float(cv2.Laplacian(crop, cv2.CV_64F).var()), 1)
    r["clipped_hi"] = round(float((gray >= 250).mean()), 4)
    r["clipped_lo"] = round(float((gray <= 5).mean()), 4)
    x = S.read_dji_xmp(path)
    r["rel_alt_m"] = x.get("rel_alt_m")
    r["gimbal_pitch"] = x.get("gimbal_pitch_deg")
    if r["rel_alt_m"]:
        g = S.gsd_m(r["rel_alt_m"], w, cam)
        r["gsd_cm"] = round(g * 100, 3)
        r["profile"] = next((k_ for k_, (lo, hi) in PROFILES.items() if lo <= r["rel_alt_m"] < hi), "outside")
    probs = []
    if min(w, h) < tile:
        probs.append(f"smaller than tile {tile}")
    if r["gimbal_pitch"] is not None and abs(r["gimbal_pitch"] + 90) > 10:
        probs.append(f"oblique ({r['gimbal_pitch']:.0f} deg)")
    if r["rel_alt_m"] is None:
        probs.append("no DJI altitude (cannot place in an altitude profile)")
    if r["clipped_hi"] > 0.05:
        probs.append(f"{r['clipped_hi']:.0%} blown highlights")
    if r["clipped_lo"] > 0.10:
        probs.append(f"{r['clipped_lo']:.0%} crushed shadows")
    r["problems"] = "; ".join(probs)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True, help="folder of real drone photos (searched recursively)")
    ap.add_argument("--exclude_dir", action="append", default=[],
                    help="real val/test images that must NOT be used as style (repeatable)")
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--camera", default="phantom4pro", choices=list(S.CAMERAS))
    ap.add_argument("--blur_rel", type=float, default=0.35,
                    help="flag photos sharper than less than this fraction of the median sharpness")
    ap.add_argument("--dup_bits", type=int, default=4, help="dHash Hamming distance counted as duplicate")
    ap.add_argument("--allow_missing_altitude", action="store_true")
    ap.add_argument("--out_csv", default="style_check.csv")
    ap.add_argument("--copy_usable", default=None, help="copy usable photos here")
    a = ap.parse_args()

    cam = S.CAMERAS[a.camera]
    files = sorted(p for p in Path(a.photos).rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"[ERROR] no images under {a.photos}")
    print(f"  analysing {len(files)} photos ...")
    rows = [analyse(p, a.tile, cam) for p in files]

    excl = []
    for d in a.exclude_dir:
        import cv2
        for p in Path(d).rglob("*"):
            if p.suffix.lower() in EXTS:
                g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if g is not None:
                    excl.append(dhash(g))

    sharp = np.array([r["sharpness"] for r in rows if "sharpness" in r])
    med = float(np.median(sharp)) if len(sharp) else 0.0
    kept_hashes = []
    for r in rows:
        if "dhash" not in r:
            r["usable"] = False
            continue
        probs = [r["problems"]] if r["problems"] else []
        if a.allow_missing_altitude:
            probs = [p.replace("no DJI altitude (cannot place in an altitude profile)", "").strip("; ")
                     for p in probs]
            probs = [p for p in probs if p]
        if r["sharpness"] < a.blur_rel * med:
            probs.append(f"blurry ({r['sharpness']:.0f} vs median {med:.0f})")
        if any(hamming(r["dhash"], e) <= a.dup_bits for e in excl):
            probs.append("LEAKAGE: matches an --exclude_dir image")
        if any(hamming(r["dhash"], k) <= a.dup_bits for k in kept_hashes):
            probs.append("near-duplicate")
        r["problems"] = "; ".join(probs)
        r["usable"] = not probs
        if r["usable"]:
            kept_hashes.append(r["dhash"])

    keys = ["file", "usable", "problems", "width", "height", "rel_alt_m", "gsd_cm", "profile",
            "gimbal_pitch", "sharpness", "clipped_hi", "clipped_lo"]
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    usable = [r for r in rows if r.get("usable")]
    reasons = Counter(p.split(" (")[0].split(":")[0] for r in rows if not r.get("usable")
                      for p in (r.get("problems") or "unreadable").split("; "))
    prof = Counter(r.get("profile", "unknown") for r in usable)
    print(f"\n  usable: {len(usable)} / {len(rows)}   (report: {a.out_csv})")
    for k, v in reasons.most_common():
        print(f"    rejected - {k}: {v}")
    print("  usable photos per renderer altitude profile:")
    for k in list(PROFILES) + ["outside", "unknown"]:
        if prof.get(k):
            print(f"    {k:<18} {prof[k]}")
    missing = [k for k in PROFILES if not prof.get(k)]
    if len(usable) < 200:
        print(f"  ⚠  {len(usable)} usable photos: AdaIN needs ~200+ for stable statistics")
    if missing:
        print(f"  ⚠  no usable photos for {', '.join(missing)}: synthetic scenes at those altitudes will be "
              f"styled with statistics from other altitudes")
    if any("LEAKAGE" in (r.get("problems") or "") for r in rows):
        print("  ⚠  some photos are (near-)copies of your val/test images -- they were rejected")
    if a.copy_usable:
        out = Path(a.copy_usable)
        out.mkdir(parents=True, exist_ok=True)
        for r in usable:
            src = Path(r["file"])
            shutil.copy2(src, out / f"{src.parent.name}__{src.name}")
        print(f"  copied {len(usable)} usable photos -> {out}")


if __name__ == "__main__":
    main()
