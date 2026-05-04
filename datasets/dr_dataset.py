import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DRDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy grading.

    Handles:
    - Loading preprocessed fundus images
    - Applying augmentations (train) or just normalization (val/test)
    - Returning image + grade label pairs
    """

    GRADE_NAMES = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR"
    }

    def __init__(self, labels_df: pd.DataFrame, img_dir: str, split: str = "train"):
        self.df        = labels_df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.split     = split
        self.transform = self._build_transforms()

    def _build_transforms(self):
        """
        Training: aggressive augmentation to improve generalization.
        Val/Test: only normalize — no augmentation.
        """
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std  = [0.229, 0.224, 0.225]

        if self.split == "train":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std)
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std)
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_name = f"{row['image']}.jpeg"
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(row["level"], dtype=torch.long)

        return image, label


def build_dataloaders(
    processed_dir: str,
    labels_dir: str,
    batch_size: int = 16
):
    """Build train, val and test dataloaders."""

    train_df = pd.read_csv(os.path.join(labels_dir, "train_labels.csv"))
    val_df   = pd.read_csv(os.path.join(labels_dir, "val_labels.csv"))
    test_df  = pd.read_csv(os.path.join(labels_dir, "test_labels.csv"))

    train_ds = DRDataset(train_df, os.path.join(processed_dir, "train"), split="train")
    val_ds   = DRDataset(val_df,   os.path.join(processed_dir, "val"),   split="val")
    test_ds  = DRDataset(test_df,  os.path.join(processed_dir, "test"),  split="test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True,  num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader