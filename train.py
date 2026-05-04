import gc

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets.dr_dataset import build_dataloaders
from models.efficientnet import DRModel
from models.losses import OrdinalRegressionLoss

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "/content/data/processed/"
LABELS_DIR    = "/content/drive/MyDrive/diabetic-retinopathy/processed/"
SAVE_PATH     = "/content/drive/MyDrive/diabetic-retinopathy/best_model.pth"
NUM_EPOCHS    = 15
BATCH_SIZE    = 4
LR            = 1e-4
NUM_CLASSES   = 5


def train_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def val_epoch(model, loader, criterion, device):
    """Run one validation epoch."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels      = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return total_loss / len(loader), correct / total, kappa


def train(
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float       = LR
):
    """Full training loop with cosine annealing scheduler."""

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}\n")

    train_loader, val_loader, _ = build_dataloaders(
        processed_dir=PROCESSED_DIR,
        labels_dir=LABELS_DIR,
        batch_size=batch_size
    )

    model     = DRModel(num_classes=NUM_CLASSES, dropout=0.3, pretrained=True).to(device)
    criterion = OrdinalRegressionLoss(num_classes=NUM_CLASSES, smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_kappa = -1
    history    = []

    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} "
          f"{'Val Loss':>9} {'Val Acc':>8} {'Kappa':>7}")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc        = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc,  kappa  = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss":   val_loss,   "val_acc":   val_acc,
            "kappa":      kappa
        })

        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.4f} "
              f"{val_loss:>9.4f} {val_acc:>8.4f} {kappa:>7.4f}")

        if kappa > best_kappa:
            best_kappa = kappa
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "kappa":               kappa,
                "val_acc":             val_acc
            }, SAVE_PATH)
            print(f"         ★ New best kappa: {kappa:.4f} — model saved!")

    print(f"\nTraining complete! Best kappa: {best_kappa:.4f}")
    return model, history


if __name__ == "__main__":
    model, history = train()