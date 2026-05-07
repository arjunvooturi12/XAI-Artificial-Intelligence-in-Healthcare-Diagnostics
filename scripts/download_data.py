"""
scripts/download_data.py
========================
Downloads ChestX-ray14 and ISIC 2019 datasets via Kaggle API.

Usage:
    python scripts/download_data.py --dataset chestxray
    python scripts/download_data.py --dataset isic
    python scripts/download_data.py --dataset both   (default)

Requirements:
    - kaggle.json placed at ~/.kaggle/kaggle.json
    - pip install kaggle
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# ── Kaggle dataset identifiers ────────────────────────────────
DATASETS = {
    "chestxray": {
        "slug": "nih-chest-xrays/data",
        "dest": "data/chestxray",
        "description": "NIH ChestX-ray14 (112,000 X-rays, 14 disease labels)",
        "size": "~45 GB",
    },
    "isic": {
        "slug": "andrewmvd/isic-2019",
        "dest": "data/isic",
        "description": "ISIC 2019 Skin Lesion (25,331 dermoscopy images, 8 classes)",
        "size": "~10 GB",
    },
}


def check_kaggle():
    """Verify Kaggle API credentials exist."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n[ERROR] Kaggle API key not found.")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token'")
        print(f"  3. Place kaggle.json at: {kaggle_json}")
        print("  4. chmod 600 ~/.kaggle/kaggle.json\n")
        sys.exit(1)
    os.chmod(kaggle_json, 0o600)
    print("✓ Kaggle credentials found.")


def download_dataset(name: str):
    info = DATASETS[name]
    dest = Path(info["dest"])
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Downloading: {info['description']}")
    print(f"  Size:        {info['size']}")
    print(f"  Destination: {dest.resolve()}")
    print(f"{'='*55}")

    cmd = [
        "kaggle", "datasets", "download",
        "-d", info["slug"],
        "-p", str(dest),
        "--unzip",
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"✓ {name} downloaded successfully.")
    else:
        print(f"✗ Download failed for {name}. Check your Kaggle credentials.")


def main():
    parser = argparse.ArgumentParser(description="Download XAI project datasets.")
    parser.add_argument(
        "--dataset",
        choices=["chestxray", "isic", "both"],
        default="both",
        help="Which dataset to download (default: both)",
    )
    args = parser.parse_args()

    check_kaggle()

    targets = ["chestxray", "isic"] if args.dataset == "both" else [args.dataset]
    for name in targets:
        download_dataset(name)

    print("\n✓ All downloads complete. Next: run notebooks/01_preprocessing.ipynb")


if __name__ == "__main__":
    main()
