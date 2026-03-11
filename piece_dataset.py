"""
ChessReD Piece Classification Dataset

Loads individual piece crops from bounding box annotations for training
a 12-class piece classifier (6 white + 6 black, no 'empty' class).

Usage:
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PieceClassificationDataset(
        dataroot="path/to/end-to-end-chess-recognition",
        split="train",
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import json
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORIES = {
    0: "white-pawn",
    1: "white-rook",
    2: "white-knight",
    3: "white-bishop",
    4: "white-queen",
    5: "white-king",
    6: "black-pawn",
    7: "black-rook",
    8: "black-knight",
    9: "black-bishop",
    10: "black-queen",
    11: "black-king",
    # 12: "empty"  ← excluded, we only classify actual pieces
}

NUM_CLASSES = len(CATEGORIES)  # 12


# ── Dataset ────────────────────────────────────────────────────────────────────

class PieceClassificationDataset(Dataset):
    """Dataset for training a piece classifier on ChessReD bounding boxes.
    
    Each sample is a single piece crop with its category label (0-11).
    Only uses the ChessReD2K subset (2,078 images with bbox annotations).
    """

    def __init__(
        self,
        dataroot: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        use_chessred2k_split: bool = True,
    ) -> None:
        """
        Args:
            dataroot: Path to ChessReD root (contains annotations.json and images/)
            split: One of 'train', 'val', 'test'
            transform: Optional transform to apply to piece crops
            use_chessred2k_split: If True, use the chessred2k nested split
                                  (70/15/15 on the 2,078 annotated images).
                                  If False, use the main split (which includes
                                  unannotated images - not recommended for this task).
        """
        super().__init__()
        self.dataroot = Path(dataroot)
        self.split = split
        self.transform = transform
        self.dataset_mode = "annotations"

        ann_path = self.dataroot / "annotations.json"
        split_dir = self.dataroot / self.split

        if ann_path.exists():
            self._init_from_annotations(ann_path, use_chessred2k_split)
            return

        if split_dir.is_dir():
            self._init_from_piece_crops(split_dir)
            return

        raise FileNotFoundError(
            f"Could not find supported dataset structure under {self.dataroot}. "
            f"Expected either {ann_path} or split folder {split_dir}."
        )

    def _init_from_annotations(self, ann_path: Path, use_chessred2k_split: bool) -> None:
        self.dataset_mode = "annotations"

        with open(ann_path, "r") as f:
            raw = json.load(f)

        images_df = pd.DataFrame(raw["images"])
        pieces_df = pd.DataFrame(raw["annotations"]["pieces"])
        corners_df = pd.DataFrame(raw["annotations"]["corners"])
        annotated_ids = set(corners_df["image_id"])

        if use_chessred2k_split:
            if "chessred2k" not in raw["splits"]:
                raise KeyError("chessred2k split not found in annotations.json")

            split_info = raw["splits"]["chessred2k"][self.split]
            split_ids = set(split_info["image_ids"])
            valid_ids = annotated_ids & split_ids
        else:
            split_info = raw["splits"][self.split]
            split_ids = set(split_info["image_ids"])
            valid_ids = annotated_ids & split_ids

        self.pieces = pieces_df[pieces_df["image_id"].isin(valid_ids)].copy()
        self.pieces = self.pieces[self.pieces["category_id"] != 12].copy()
        self.images = images_df[images_df["id"].isin(valid_ids)].set_index("id")
        self.pieces = self.pieces.reset_index(drop=True)

        print(f"[PieceClassificationDataset] {self.split} split:")
        print(f"  {len(valid_ids)} images")
        print(f"  {len(self.pieces)} piece samples")
        print("  Class distribution:")
        for cat_id in sorted(self.pieces["category_id"].unique()):
            count = (self.pieces["category_id"] == cat_id).sum()
            name = CATEGORIES.get(cat_id, f"unknown-{cat_id}")
            print(f"    {cat_id:2d} {name:20s}: {count:5d} samples")

    def _init_from_piece_crops(self, split_dir: Path) -> None:
        self.dataset_mode = "piece_crops"

        name_to_idx = {name: idx for idx, name in CATEGORIES.items()}
        valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        self.samples = []
        class_counts = {}

        for class_name in sorted(name_to_idx.keys()):
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue

            label = name_to_idx[class_name]
            image_paths = [
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_suffixes
            ]
            class_counts[class_name] = len(image_paths)
            self.samples.extend((path, label) for path in image_paths)

        if not self.samples:
            raise RuntimeError(
                f"No image samples found in {split_dir}. "
                "Expected folders like train/white-pawn, train/black-king, etc."
            )

        print(f"[PieceClassificationDataset] {self.split} split (piece_crops mode):")
        print(f"  {len(self.samples)} piece samples")
        print("  Class distribution:")
        for class_name in sorted(class_counts.keys()):
            print(f"    {class_name:20s}: {class_counts[class_name]:5d} samples")

    def __len__(self) -> int:
        if self.dataset_mode == "piece_crops":
            return len(self.samples)
        return len(self.pieces)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: Tensor of shape (C, H, W) - the piece crop
            label: Integer category ID (0-11)
        """
        if self.dataset_mode == "piece_crops":
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, label

        row = self.pieces.iloc[idx]
        
        # ── Load full image ────────────────────────────────────────────────────
        img_id = row["image_id"]
        img_meta = self.images.loc[img_id]
        img_path = self.dataroot / img_meta["path"]
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        
        # ── Crop the piece ─────────────────────────────────────────────────────
        bbox = row["bbox"]  # [x, y, width, height]
        x, y, w, h = bbox
        
        # Convert to integer pixel coords and clamp to image bounds
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(image.width, int(x + w))
        y2 = min(image.height, int(y + h))
        
        piece_crop = image.crop((x1, y1, x2, y2))
        
        # ── Apply transform ────────────────────────────────────────────────────
        if self.transform is not None:
            piece_crop = self.transform(piece_crop)
        
        label = int(row["category_id"])
        
        return piece_crop, label


# ── Utility functions ──────────────────────────────────────────────────────────

def get_default_transforms(train: bool = True):
    """Get default transforms for piece classification.
    
    Args:
        train: If True, include data augmentation for training
    
    Returns:
        torchvision.transforms.Compose object
    """
    from torchvision import transforms
    
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(dataroot: Union[str, Path],
                      batch_size: int = 32,
                      num_workers: int = 4):
    """Convenience function to create train/val/test dataloaders.
    
    Args:
        dataroot: Path to ChessReD root directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
    
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing DataLoaders
    """
    from torch.utils.data import DataLoader
    
    loaders = {}
    for split in ["train", "val", "test"]:
        is_train = (split == "train")
        dataset = PieceClassificationDataset(
            dataroot=dataroot,
            split=split,
            transform=get_default_transforms(train=is_train),
            use_chessred2k_split=True
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True,
                       help="Path to ChessReD root directory")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Testing PieceClassificationDataset")
    print("="*80 + "\n")
    
    # Test all splits
    for split in ["train", "val", "test"]:
        print(f"\n{'─'*80}")
        dataset = PieceClassificationDataset(
            dataroot=args.dataroot,
            split=split,
            transform=get_default_transforms(train=(split=="train")),
            use_chessred2k_split=True
        )
        
        # Show a sample
        img, label = dataset[0]
        print(f"\nSample from {split}:")
        print(f"  Image shape: {img.shape}")
        print(f"  Label: {label} ({CATEGORIES[label]})")
