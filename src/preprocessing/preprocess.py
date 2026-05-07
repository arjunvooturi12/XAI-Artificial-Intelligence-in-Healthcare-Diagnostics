"""
src/preprocessing/preprocess.py
=================================
Phase 1 — Data preprocessing pipeline for:
  - ChestX-ray14  (multi-label, 14 classes)
  - ISIC 2019     (multi-class, 8 classes)

Handles:
  - Image resizing & normalization
  - Class-balanced train/val/test splits
  - Data augmentation (albumentations)
  - Dataset statistics computation
  - PyTorch Dataset / DataLoader creation
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Reproducibility ───────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Constants ─────────────────────────────────────────────────
IMG_SIZE = 224  # Required by ResNet / EfficientNet / ViT

CHESTXRAY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

ISIC_LABELS = [
    "MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC",
]

# ImageNet normalization (used for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════
#  TRANSFORMS
# ═══════════════════════════════════════════════════════════════

def get_train_transforms() -> A.Compose:
    """Augmentation pipeline for training set."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
        A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(0.05, 0.1), hole_width_range=(0.05, 0.1), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """No augmentation for validation/test — only resize & normalize."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_pytorch_val_transform() -> T.Compose:
    """PyTorch-style val transform (for Grad-CAM compatibility)."""
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ═══════════════════════════════════════════════════════════════
#  CHESTX-RAY14 DATASET
# ═══════════════════════════════════════════════════════════════

class ChestXrayDataset(Dataset):
    """
    NIH ChestX-ray14 Dataset.

    Args:
        df:        DataFrame with columns [image_path, label_0 ... label_13]
        transform: albumentations Compose transform
    """

    def __init__(self, df: pd.DataFrame, transform: Optional[A.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.labels = CHESTXRAY_LABELS

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        label = torch.tensor(
            row[self.labels].values.astype(np.float32),
            dtype=torch.float32,
        )
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


def build_chestxray_df(data_root: str) -> pd.DataFrame:
    """
    Parse ChestX-ray14 metadata CSV and build a DataFrame
    with image paths and one-hot labels.

    Args:
        data_root: path to the folder containing Data_Entry_2017.csv
                   and the images/ subfolder.
    Returns:
        pd.DataFrame with columns: image_path, + one column per disease label.
    """
    data_root = Path(data_root)
    csv_path = data_root / "Data_Entry_2017.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found at {csv_path}.\n"
            "Run: python scripts/download_data.py --dataset chestxray"
        )

    df = pd.read_csv(csv_path)
    df = df[["Image Index", "Finding Labels"]].copy()
    df.columns = ["filename", "findings"]

    # Build image paths (images are spread across images_*/images/ folders)
    image_dirs = sorted(data_root.glob("images_*/images"))
    path_map: Dict[str, Path] = {}
    for d in image_dirs:
        for p in d.glob("*.png"):
            path_map[p.name] = p

    df["image_path"] = df["filename"].map(lambda f: str(path_map.get(f, "")))
    df = df[df["image_path"] != ""].copy()

    # One-hot encode labels
    for label in CHESTXRAY_LABELS:
        df[label] = df["findings"].apply(lambda x: int(label in x.split("|")))

    # Drop "No Finding" rows if you want only diseased images (optional)
    # df = df[df["findings"] != "No Finding"]

    return df[["image_path"] + CHESTXRAY_LABELS]


def split_chestxray(
    df: pd.DataFrame,
    val_size: float = 0.10,
    test_size: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split using the first label column for stratification."""
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=SEED,
        stratify=df[CHESTXRAY_LABELS[0]],
    )
    val_frac = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_frac, random_state=SEED,
        stratify=train_val[CHESTXRAY_LABELS[0]],
    )
    return train, val, test


# ═══════════════════════════════════════════════════════════════
#  ISIC 2019 DATASET
# ═══════════════════════════════════════════════════════════════

class ISICDataset(Dataset):
    """
    ISIC 2019 Skin Lesion Dataset.

    Args:
        df:        DataFrame with columns [image_path, label (int)]
        transform: albumentations Compose transform
    """

    def __init__(self, df: pd.DataFrame, transform: Optional[A.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_names = ISIC_LABELS

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        label = torch.tensor(row["label"], dtype=torch.long)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


def build_isic_df(data_root: str) -> pd.DataFrame:
    """
    Parse ISIC 2019 ground truth CSV and build a DataFrame
    with image paths and integer class labels.

    Args:
        data_root: path to the ISIC folder containing
                   ISIC_2019_Training_GroundTruth.csv and ISIC_2019_Training_Input/
    Returns:
        pd.DataFrame with columns: image_path, label, class_name
    """
    data_root = Path(data_root)
    csv_path = data_root / "ISIC_2019_Training_GroundTruth.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"ISIC ground truth CSV not found at {csv_path}.\n"
            "Run: python scripts/download_data.py --dataset isic"
        )

    gt = pd.read_csv(csv_path)
    img_dir = data_root / "ISIC_2019_Training_Input"

    records = []
    for _, row in gt.iterrows():
        img_id = row["image"]
        img_path = img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        label_idx = int(np.argmax(row[ISIC_LABELS].values))
        records.append({
            "image_path": str(img_path),
            "label": label_idx,
            "class_name": ISIC_LABELS[label_idx],
        })

    return pd.DataFrame(records)


def split_isic(
    df: pd.DataFrame,
    val_size: float = 0.10,
    test_size: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split by class label."""
    n_classes = df["label"].nunique()
    stratify_col = df["label"] if len(df) > n_classes * 10 else None
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=stratify_col,
    )
    val_frac = val_size / (1 - test_size)
    stratify_tv = train_val["label"] if len(train_val) > n_classes * 10 else None
    train, val = train_test_split(
        train_val, test_size=val_frac, random_state=SEED, stratify=stratify_tv,
    )
    return train, val, test


# ═══════════════════════════════════════════════════════════════
#  DATALOADERS
# ═══════════════════════════════════════════════════════════════

def get_chestxray_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for ChestX-ray14."""
    df = build_chestxray_df(data_root)
    train_df, val_df, test_df = split_chestxray(df)

    train_ds = ChestXrayDataset(train_df, transform=get_train_transforms())
    val_ds   = ChestXrayDataset(val_df,   transform=get_val_transforms())
    test_ds  = ChestXrayDataset(test_df,  transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"ChestX-ray14  |  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader


def get_isic_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for ISIC 2019."""
    df = build_isic_df(data_root)
    train_df, val_df, test_df = split_isic(df)

    train_ds = ISICDataset(train_df, transform=get_train_transforms())
    val_ds   = ISICDataset(val_df,   transform=get_val_transforms())
    test_ds  = ISICDataset(test_df,  transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"ISIC 2019     |  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════
#  DATASET STATISTICS
# ═══════════════════════════════════════════════════════════════

def compute_dataset_stats(dataset_name: str, df: pd.DataFrame) -> Dict:
    """Compute and print dataset statistics."""
    stats = {"dataset": dataset_name, "total": len(df)}

    if dataset_name == "chestxray":
        label_counts = {l: int(df[l].sum()) for l in CHESTXRAY_LABELS}
        stats["label_counts"] = label_counts
        stats["no_finding"] = int((df[CHESTXRAY_LABELS].sum(axis=1) == 0).sum())
        print(f"\n── ChestX-ray14 Statistics ──────────────────")
        print(f"  Total images : {stats['total']:,}")
        print(f"  No Finding   : {stats['no_finding']:,}")
        print(f"  Label counts :")
        for label, count in label_counts.items():
            bar = "█" * (count // 2000)
            print(f"    {label:<22} {count:>6,}  {bar}")

    elif dataset_name == "isic":
        label_counts = df["class_name"].value_counts().to_dict()
        stats["label_counts"] = label_counts
        print(f"\n── ISIC 2019 Statistics ─────────────────────")
        print(f"  Total images : {stats['total']:,}")
        print(f"  Class counts :")
        for cls, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            bar = "█" * (count // 200)
            print(f"    {cls:<6} {count:>6,}  {bar}")

    return stats


# ═══════════════════════════════════════════════════════════════
#  QUICK SMOKE TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("Running preprocessing smoke test with synthetic data...\n")

    # Create tiny synthetic DataFrames to test pipeline without real data
    n = 50
    tmp_dir = Path("C:/temp/xai_smoke_test")
    tmp_dir.mkdir(exist_ok=True)

    # Create dummy images
    for i in range(n):
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img.save(tmp_dir / f"img_{i:03d}.png")

    # ChestX-ray smoke test
    records = []
    for i in range(n):
        row = {"image_path": str(tmp_dir / f"img_{i:03d}.png")}
        for j, label in enumerate(CHESTXRAY_LABELS):
            row[label] = random.randint(0, 1)
        records.append(row)
    df_cx = pd.DataFrame(records)
    compute_dataset_stats("chestxray", df_cx)

    train, val, test = split_chestxray(df_cx)
    ds = ChestXrayDataset(train, transform=get_train_transforms())
    img, lbl = ds[0]
    print(f"\n  ChestXrayDataset sample — image: {img.shape}, label: {lbl.shape}")

    # ISIC smoke test
    records = []
    for i in range(n):
        records.append({
            "image_path": str(tmp_dir / f"img_{i:03d}.png"),
            "label": random.randint(0, 7),
            "class_name": ISIC_LABELS[random.randint(0, 7)],
        })
    df_is = pd.DataFrame(records)
    compute_dataset_stats("isic", df_is)

    train, val, test = split_isic(df_is)
    ds = ISICDataset(train, transform=get_train_transforms())
    img, lbl = ds[0]
    print(f"\n  ISICDataset sample      — image: {img.shape}, label: {lbl}")

    print("\n✓ Smoke test passed — preprocessing pipeline is ready.")