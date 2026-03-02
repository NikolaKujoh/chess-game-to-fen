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

        # ── Load annotations ───────────────────────────────────────────────────
        ann_path = self.dataroot / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"annotations.json not found at {ann_path}")

        with open(ann_path, "r") as f:
            raw = json.load(f)

        # Convert to pandas for easier filtering
        images_df = pd.DataFrame(raw["images"])
        pieces_df = pd.DataFrame(raw["annotations"]["pieces"])
        corners_df = pd.DataFrame(raw["annotations"]["corners"])

        # ── Filter to ChessReD2K subset ────────────────────────────────────────
        # Only images with corner annotations have bbox annotations (2,078 total)
        annotated_ids = set(corners_df["image_id"])
        
        if use_chessred2k_split:
            # Use the nested chessred2k split (70/15/15 on annotated subset)
            if "chessred2k" not in raw["splits"]:
                raise KeyError("chessred2k split not found in annotations.json")
            
            split_info = raw["splits"]["chessred2k"][split]
            split_ids = set(split_info["image_ids"])
            
            # Intersection: only annotated images in this split
            valid_ids = annotated_ids & split_ids
        else:
            # Use the main split but filter to annotated images
            split_info = raw["splits"][split]
            split_ids = set(split_info["image_ids"])
            valid_ids = annotated_ids & split_ids

        # ── Filter pieces to valid images ──────────────────────────────────────
        self.pieces = pieces_df[pieces_df["image_id"].isin(valid_ids)].copy()
        
        # Remove 'empty' class (category_id=12) - we only classify actual pieces
        self.pieces = self.pieces[self.pieces["category_id"] != 12].copy()
        
        # Build image metadata lookup
        self.images = images_df[images_df["id"].isin(valid_ids)].set_index("id")

        # Reset index for clean iteration
        self.pieces = self.pieces.reset_index(drop=True)

        print(f"[PieceClassificationDataset] {split} split:")
        print(f"  {len(valid_ids)} images")
        print(f"  {len(self.pieces)} piece samples")
        print(f"  Class distribution:")
        for cat_id in sorted(self.pieces["category_id"].unique()):
            count = (self.pieces["category_id"] == cat_id).sum()
            name = CATEGORIES.get(cat_id, f"unknown-{cat_id}")
            print(f"    {cat_id:2d} {name:20s}: {count:5d} samples")

    def __len__(self) -> int:
        return len(self.pieces)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: Tensor of shape (C, H, W) - the piece crop
            label: Integer category ID (0-11)
        """
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
