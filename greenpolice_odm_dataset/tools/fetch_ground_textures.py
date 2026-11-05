"""
Download a curated set of ambientCG ground materials (CC0) for the renderer.

Why not `blenderproc download cc_textures`: it downloads the ENTIRE library
(tens of GB: fabrics, tiles, wood, food...) and its folder names never matched
the renderer's old tag list (sand_01, desert_ground...). This script pages
through the ambientCG catalogue metadata (small JSON), picks ground surfaces per
environment by asset-ID prefix + tags, and downloads only those 2K-JPG zips
(~15-40 MB each).

Output (what 02_blenderproc_synthetic_v2.py reads):
  assets/textures/ground/<environment>/<AssetID>/
      <AssetID>_2K-JPG_Color.jpg  _Roughness.jpg  _NormalGL.jpg  _AmbientOcclusion.jpg ...
      asset.json      {"id", "environment", "tags", "real_size_m", "source_url", "license"}
  assets/textures/ground/manifest.json   (every asset used — cite this in the paper)

Usage (run on your machine; needs internet):
  python tools/fetch_ground_textures.py                       # 14 per environment
  python tools/fetch_ground_textures.py --per_env 20
  python tools/fetch_ground_textures.py --list_only           # show candidates, download nothing
  python tools/fetch_ground_textures.py --ids desert=Ground054,Ground080 european_rural=Grass004

API structure follows BlenderProc's own downloader
(blenderproc/scripts/download_cc_textures.py): api/v2/full_json, include=downloadData,
assets[].downloadFolders.default.downloadFiletypeCategories.zip.downloads[] with
attribute "2K-JPG" -> downloadLink, and a browser User-Agent header is required.
"""
import argparse
import io
import json
import random
import re
import sys
import time
import zipfile
from pathlib import Path

API = "https://ambientcg.com/api/v2/full_json"
HEADERS = {"User-Agent": "Mozilla/5.0 (EWADS research data pipeline)"}

# Asset-ID prefixes that are ground surfaces, per environment. An asset qualifies
# if its ID is <Prefix><digits>[letter] AND it passes the tag filters.
ENV_RULES = {
    "desert": {
        "prefixes": ["Ground", "Gravel", "Rocks"],
        "want_tags": ["sand", "desert", "dry", "dirt", "soil", "cracked", "dust", "arid",
                      "gravel", "pebbles", "stones", "rocks", "earth"],
        "reject_tags": ["grass", "moss", "snow", "ice", "leaves", "forest", "wet", "mud",
                        "water", "puddle", "flowers", "tiles", "brick", "wood"],
    },
    "european_urban": {
        "prefixes": ["Asphalt", "Concrete", "PavingStones", "Gravel", "Ground"],
        "want_tags": ["asphalt", "concrete", "gravel", "pavement", "paving", "road", "dirt",
                      "urban", "cobblestone", "street", "ground"],
        "reject_tags": ["snow", "ice", "water", "marking", "lines", "sign", "manhole",
                        "wall", "facade", "tiles", "roof", "painted", "graffiti",
                        "sand", "desert", "grass", "forest", "moss", "leaves"],
    },
    "european_rural": {
        "prefixes": ["Grass", "Ground", "Moss", "Gravel"],
        "want_tags": ["grass", "moss", "forest", "leaves", "mud", "soil", "dirt", "meadow",
                      "field", "lawn", "earth", "roots"],
        "reject_tags": ["snow", "ice", "water", "sand", "desert", "tiles", "brick", "wall",
                        "artificial", "astroturf"],
    },
}
# Anything with these in the ID is never a usable open-ground surface
GLOBAL_REJECT_ID = re.compile(r"^(RoadLines|Road\d|Snow|Ice|Decal|Manhole|Sign|Leaking|Foliage|3D)", re.I)


def _get(url, params=None, retries=3):
    import requests
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r
        except Exception as e:
            if k == retries - 1:
                raise
            print(f"    retry {k + 1}: {e}")
            time.sleep(2 * (k + 1))


def fetch_catalogue(cache: Path) -> list:
    """All material assets' metadata (cached to disk — the catalogue barely changes)."""
    if cache.exists():
        return json.loads(cache.read_text())
    assets, offset = [], 0
    while True:
        r = _get(API, {"type": "material", "include": "downloadData,tagData,dimensionsData",
                       "limit": 100, "offset": offset})
        found = r.json().get("foundAssets", [])
        if not found:
            break
        assets.extend(found)
        offset += 100
        print(f"  catalogue: {len(assets)} assets", end="\r")
    print()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(assets))
    return assets


def zip_2k_jpg(asset: dict):
    """(downloadLink, zipContent) for the 2K-JPG zip, or None."""
    try:
        downloads = asset["downloadFolders"]["default"]["downloadFiletypeCategories"]["zip"]["downloads"]
    except (KeyError, TypeError):
        return None
    for d in downloads:
        if d.get("attribute") == "2K-JPG":
            return d.get("downloadLink"), d.get("zipContent") or []
    return None


def tags_of(asset: dict) -> list:
    t = asset.get("tags") or []
    return [str(x).lower() for x in t]


def real_size_m(asset: dict):
    """
    Physical width of the texture in metres, if ambientCG provides it. Field
    names in dimensionsData are not documented here, so look for any numeric
    'dimension*' value and treat values > 20 as centimetres. None -> the
    renderer randomises the tile size (1.2-3 m).
    """
    for k, v in asset.items():
        if "dimension" in k.lower() and isinstance(v, (int, float)) and v > 0:
            return round(v / 100.0 if v > 20 else float(v), 3)
    return None


def classify(asset: dict, env: str) -> int:
    """Score >0 if the asset is a plausible ground surface for env (higher = better)."""
    aid = asset.get("assetId", "")
    if GLOBAL_REJECT_ID.match(aid):
        return 0
    rule = ENV_RULES[env]
    if not any(re.fullmatch(rf"{p}\d+[A-Z]?", aid) for p in rule["prefixes"]):
        return 0
    link = zip_2k_jpg(asset)
    if not link:
        return 0
    if any("opacity" in f.lower() for f in link[1]):      # cut-out decals, not ground
        return 0
    tags = tags_of(asset)
    if any(any(r in t for t in tags) for r in rule["reject_tags"]):
        return 0
    hits = sum(any(w in t for t in tags) for w in rule["want_tags"])
    return 1 + hits if hits else 0        # must carry at least one environment tag


def download(asset: dict, env: str, out_root: Path) -> bool:
    aid = asset["assetId"]
    link, content = zip_2k_jpg(asset)
    dst = out_root / env / aid
    if dst.exists() and list(dst.glob("*_Color.jpg")):
        return True
    dst.mkdir(parents=True, exist_ok=True)
    try:
        data = _get(link).content
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():                     # flat extract, maps only
                if name.lower().endswith((".jpg", ".png")) and "preview" not in name.lower():
                    (dst / Path(name).name).write_bytes(z.read(name))
    except Exception as e:
        print(f"    ✗ {aid}: {e}")
        return False
    if not list(dst.glob("*_Color.jpg")):
        print(f"    ✗ {aid}: zip had no *_Color.jpg ({sorted(p.name for p in dst.iterdir())[:4]})")
        return False
    (dst / "asset.json").write_text(json.dumps({
        "id": aid, "environment": env, "tags": tags_of(asset),
        "real_size_m": real_size_m(asset), "source_url": f"https://ambientcg.com/view?id={aid}",
        "download": link, "license": "CC0 1.0 (ambientCG)",
    }, indent=2))
    return True


def main():
    ap = argparse.ArgumentParser(description="Curated ambientCG ground textures for EWADS")
    ap.add_argument("--out_dir", default="./assets/textures/ground")
    ap.add_argument("--per_env", type=int, default=14)
    ap.add_argument("--envs", nargs="+", default=list(ENV_RULES))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--list_only", action="store_true")
    ap.add_argument("--ids", nargs="*", default=[],
                    help="force exact IDs: env=ID1,ID2 (skips automatic selection for that env)")
    args = ap.parse_args()
    try:
        import requests  # noqa: F401
    except ImportError:
        sys.exit("pip install requests")

    out = Path(args.out_dir)
    catalogue = fetch_catalogue(out / "_catalogue_cache.json")
    by_id = {a.get("assetId"): a for a in catalogue}
    forced = {k: v.split(",") for k, v in (s.split("=", 1) for s in args.ids)}
    rng = random.Random(args.seed)

    manifest = {}
    for env in args.envs:
        if env in forced:
            chosen = [by_id[i] for i in forced[env] if i in by_id]
            missing = [i for i in forced[env] if i not in by_id]
            if missing:
                print(f"  ⚠ {env}: unknown IDs {missing}")
        else:
            scored = [(classify(a, env), a) for a in catalogue]
            cands = [a for s, a in sorted(scored, key=lambda x: -x[0]) if s > 0]
            # take the best-tagged half deterministically, fill the rest at random for variety
            top = cands[: args.per_env // 2]
            rest = cands[args.per_env // 2:]
            rng.shuffle(rest)
            chosen = top + rest[: args.per_env - len(top)]
            print(f"\n{env}: {len(cands)} candidates")
        for a in chosen:
            print(f"   {a['assetId']:<18} {', '.join(tags_of(a)[:8])}")
        if args.list_only:
            continue
        ok = [a["assetId"] for a in chosen if download(a, env, out)]
        manifest[env] = ok
        print(f"  {env}: {len(ok)}/{len(chosen)} downloaded")

    if not args.list_only:
        (out / "manifest.json").write_text(json.dumps(
            {"source": "ambientCG (CC0)", "resolution": "2K-JPG", "environments": manifest}, indent=2))
        print(f"\nManifest: {out / 'manifest.json'}")
        print("Now eyeball a few folders: anything that isn't open ground (walls, tiles,"
              " painted road) should be deleted before the full render.")


if __name__ == "__main__":
    main()
