import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path("/content/data")
SAMPLE_DIR    = BASE_DIR / "sample_train"
PROCESSED_DIR = BASE_DIR / "processed"
DRIVE_DIR     = Path("/content/drive/MyDrive/diabetic-retinopathy")
LABELS_CSV    = BASE_DIR / "trainLabels_sample.csv"
IMAGE_SIZE    = 512

# ── Grade labels ─────────────────────────────────────────────────────────────
GRADE_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhances visibility of retinal lesions — standard in clinical
    DR grading pipelines.
    """
    lab     = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    lab     = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def crop_black_border(image: np.ndarray, tolerance: int = 7) -> np.ndarray:
    """
    Remove black borders around fundus images.
    Focuses the model on actual retinal tissue only.
    """
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask   = gray > tolerance
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]

def preprocess_image(img_path: Path, output_path: Path, size: int = IMAGE_SIZE) -> bool:
    """Full preprocessing pipeline for a single fundus image."""
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"  Warning: could not read {img_path}")
        return False
    image = crop_black_border(image)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
    image = apply_clahe(image)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return True

def run_preprocessing():
    """
    Full preprocessing pipeline:
    1. Load sampled labels
    2. Stratified train/val/test split (70/15/15)
    3. Preprocess and save each image
    4. Save split CSVs to Google Drive
    """
    print("Loading labels...")
    df = pd.read_csv(LABELS_CSV)
    print(f"  Total samples: {len(df)}")
    print(f"  Grade distribution:\n{df['level'].value_counts().sort_index()}\n")

    # ── Stratified split ──────────────────────────────────────────────────────
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["level"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["level"], random_state=42
    )

    print(f"Split sizes:")
    print(f"  Train: {len(train_df)} images")
    print(f"  Val:   {len(val_df)} images")
    print(f"  Test:  {len(test_df)} images\n")

    # ── Process each split ────────────────────────────────────────────────────
    splits = {
        "train": train_df,
        "val":   val_df,
        "test":  test_df,
    }

    for split_name, split_df in splits.items():
        print(f"Processing {split_name} set...")
        success, failed = 0, 0

        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            img_name  = f"{row['image']}.jpeg"
            src_path  = SAMPLE_DIR / img_name
            dest_path = PROCESSED_DIR / split_name / img_name

            if preprocess_image(src_path, dest_path):
                success += 1
            else:
                failed += 1

        print(f"  {success} processed, {failed} failed\n")

    # ── Save label CSVs to Drive ──────────────────────────────────────────────
    drive_processed = DRIVE_DIR / "processed"
    drive_processed.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(drive_processed / "train_labels.csv", index=False)
    val_df.to_csv(drive_processed / "val_labels.csv",     index=False)
    test_df.to_csv(drive_processed / "test_labels.csv",   index=False)

    print("Preprocessing complete!")

if __name__ == "__main__":
    run_preprocessing()