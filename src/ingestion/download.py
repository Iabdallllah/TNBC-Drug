"""Download GDSC2 + DepMap raw files for TNBC platform.
Usage:
    python src/ingestion/download.py --small-only   # fast: GDSC2 + Model.csv (~22MB)
    python src/ingestion/download.py --all           # + expression matrix (~1.1GB)
"""
import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
FILES = {
    "gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx": (
        "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/"
        "GDSC2_fitted_dose_response_27Oct23.xlsx"
    ),
    "depmap/Model.csv": (
        "https://zenodo.org/api/records/20355477/files/Model.csv/content"
    ),
    "depmap/OmicsProfiles.csv": (
        "https://zenodo.org/api/records/20355477/files/OmicsProfiles.csv/content"
    ),
    "depmap/OmicsExpressionTPMLogp1HumanAllGenesStranded.csv": (
        "https://zenodo.org/api/records/20355477/files/"
        "OmicsExpressionTPMLogp1HumanAllGenesStranded.csv/content"
    ),
}
SMALL = [k for k in FILES if "OmicsExpressionTPM" not in k]


def show_progress(block, block_size, total):
    done = block * block_size
    pct = min(100, done * 100 / total) if total > 0 else 0
    sys.stdout.write(f"\r  {pct:5.1f}%  {done//1024//1024}MB/{total//1024//1024}MB")
    sys.stdout.flush()


def fetch(rel, url):
    dest = RAW / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"SKIP (exists): {rel} [{dest.stat().st_size//1024//1024}MB]")
        return
    print(f"GET  {rel}")
    print(f"  <- {url}")
    urlretrieve(url, dest, reporthook=show_progress)
    print(f"\nOK -> {dest} [{dest.stat().st_size//1024//1024}MB]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include 1.1GB expression matrix")
    ap.add_argument("--small-only", action="store_true", help="only GDSC2 + Model.csv")
    args = ap.parse_args()
    keys = SMALL if (args.small_only or not args.all) else list(FILES)
    for k in keys:
        fetch(k, FILES[k])
    print("\nDone. Next: python src/ingestion/audit.py")


if __name__ == "__main__":
    main()
