import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)

from models.efficientnet import DRModel
from datasets.dr_dataset import build_dataloaders

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "/content/data/processed/"
LABELS_DIR    = "/content/drive/MyDrive/diabetic-retinopathy/processed/"
MODEL_PATH    = "/content/drive/MyDrive/diabetic-retinopathy/best_model.pth"
SAVE_DIR      = "/content/drive/MyDrive/diabetic-retinopathy/"
NUM_CLASSES   = 5
BATCH_SIZE    = 4
GRADE_NAMES   = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def load_model(model_path: str, device: torch.device) -> DRModel:
    """Load best model from checkpoint."""
    model      = DRModel(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model      = model.to(device)
    model.eval()
    print(f"Model loaded — best kappa was {checkpoint['kappa']:.4f}")
    return model


def run_evaluation():
    """Full evaluation on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}\n")

    # Load model and data
    model = load_model(MODEL_PATH, device)
    _, _, test_loader = build_dataloaders(
        processed_dir=PROCESSED_DIR,
        labels_dir=LABELS_DIR,
        batch_size=BATCH_SIZE
    )

    # Run inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # ── Metrics ───────────────────────────────────────────────────────────────
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    print(f"Test Quadratic Kappa: {kappa:.4f}\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=GRADE_NAMES))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=GRADE_NAMES,
        yticklabels=GRADE_NAMES
    )
    ax.set_title(f"Confusion Matrix — Test Set (Kappa: {kappa:.4f})")
    ax.set_ylabel("True Grade")
    ax.set_xlabel("Predicted Grade")
    plt.tight_layout()
    plt.savefig(SAVE_DIR + "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Confusion matrix saved!")

    return all_labels, all_preds, kappa


if __name__ == "__main__":
    run_evaluation()