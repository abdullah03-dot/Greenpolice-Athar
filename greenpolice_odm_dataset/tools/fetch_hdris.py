"""
Download a curated set of outdoor Poly Haven HDRIs (CC0) per environment.

The renderer lights HDRI_PROBABILITY (60%) of scenes with one of these and the
rest with its procedural Nishita sky + sun, so lighting is not limited to one
analytic sky model. HDRI scenes use the HDRI alone — outdoor HDRIs already
contain the sun, and adding the script's sun lamp double-lit the frame.

Output (what 02_blenderproc_synthetic_v2.py reads):
  assets/haven/<environment>/<id>_2k.hdr
  assets/haven/manifest.json      (ids + licence — cite in the paper)

Usage (needs internet):
  python tools/fetch_hdris.py                 # 10 per environment, 2k
  python tools/fetch_hdris.py --per_env 16 --resolution 4k
  python tools/fetch_hdris.py --list_only

Endpoints follow BlenderProc's own downloader (blenderproc/scripts/download_haven.py):
  https://api.polyhaven.com/assets?t=hdris          -> {id: {name, categories, tags, ...}}
  https://api.polyhaven.com/files/<id>              -> ["hdri"][<res>]["hdr"]["url"]
"""
import argparse
import json
import random
import sys
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0 (EWADS research data pipeline)"}
# An HDRI qualifies for an environment if it is outdoor, daylight, and has one
# of the wanted words in its categories/tags. Scored by number of matches.
ENV_RULES = {
    "desert":         ["desert", "arid", "sand", "dry", "clear", "sunny", "midday", "mountain"],
    "european_urban": ["urban", "city", "street", "town", "industrial", "parking", "road", "overcast"],
    "european_rural": ["nature", "field", "countryside", "forest", "meadow", "farm", "rural",
                       "overcast", "partly cloudy", "grass"],
}
REJECT = ["indoor", "studio", "night", "interior", "sunset", "sunrise", "low contrast"]


def words(info: dict) -> list:
    return [str(x).lower() for x in (info.get("categories") or []) + (info.get("tags") or [])]


def score(info: dict, env: str) -> int:
    w = words(info)
    if not any("outdoor" in x for x in w) or any(r in x for x in w for r in REJECT):
        return 0
    return sum(any(k in x for x in w) for k in ENV_RULES[env])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./assets/haven")
    ap.add_argument("--per_env", type=int, default=10)
    ap.add_argument("--resolution", default="2k", choices=["1k", "2k", "4k"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--list_only", action="store_true")
    args = ap.parse_args()
    try:
        import requests
    except ImportError:
        sys.exit("pip install requests")

    out = Path(args.out_dir)
    catalogue = requests.get("https://api.polyhaven.com/assets?t=hdris", headers=HEADERS, timeout=60).json()
    rng = random.Random(args.seed)
    used, manifest = set(), {}
    for env in ENV_RULES:
        cands = sorted(((score(i, env), hid) for hid, i in catalogue.items() if hid not in used),
                       reverse=True)
        cands = [hid for s, hid in cands if s > 0]
        top, rest = cands[: args.per_env // 2], cands[args.per_env // 2:]
        rng.shuffle(rest)
        chosen = top + rest[: args.per_env - len(top)]
        used.update(chosen)
        print(f"\n{env}: {len(cands)} candidates -> {len(chosen)}")
        for hid in chosen:
            print(f"   {hid:<32} {', '.join(words(catalogue[hid])[:7])}")
        if args.list_only:
            continue
        d = out / env
        d.mkdir(parents=True, exist_ok=True)
        ok = []
        for hid in chosen:
            dst = d / f"{hid}_{args.resolution}.hdr"
            if not dst.exists():
                try:
                    files = requests.get(f"https://api.polyhaven.com/files/{hid}", headers=HEADERS,
                                         timeout=60).json()
                    url = files["hdri"][args.resolution]["hdr"]["url"]
                    dst.write_bytes(requests.get(url, headers=HEADERS, timeout=300).content)
                except Exception as e:
                    print(f"    ✗ {hid}: {e}")
                    continue
            ok.append(hid)
            print(f"    ✓ {dst.name}")
        manifest[env] = ok
    if not args.list_only:
        (out / "manifest.json").write_text(json.dumps(
            {"source": "Poly Haven (CC0)", "resolution": args.resolution, "environments": manifest},
            indent=2))
        print(f"\nManifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
