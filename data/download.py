import zipfile
from pathlib import Path

import kaggle

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR   = Path("data/raw")
TRAIN_DIR = RAW_DIR / "train"
TEST_DIR  = RAW_DIR / "test"

def download_dataset():
    """Download DR dataset using Kaggle CLI."""

    print("Creating directories...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Authenticating with Kaggle...")
    kaggle.api.authenticate()

    print("Downloading competition files...")
    kaggle.api.competition_download_files(
        competition="diabetic-retinopathy-detection",
        path=RAW_DIR,
        quiet=False
    )

    print("\nUnzipping files...")
    for zip_file in RAW_DIR.glob("*.zip"):
        print(f"  Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(RAW_DIR)
        zip_file.unlink()

    print("\nDone!")
    _print_summary()

def _print_summary():
    train_count = len(list(TRAIN_DIR.glob("*.jpeg"))) if TRAIN_DIR.exists() else 0
    print(f"  train/: {train_count:,} images")

if __name__ == "__main__":
    download_dataset()