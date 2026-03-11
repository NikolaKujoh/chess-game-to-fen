from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(image_size: int = 224) -> transforms.Compose:
    """Standard transforms for piece classification."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_datasets(data_root: Path, image_size: int = 224):
    """Create train/val/test ImageFolder datasets.

    Expected layout:
      data_root/
        train/<class_name>/*.png
        val/<class_name>/*.png
        test/<class_name>/*.png
    """
    data_root = Path(data_root)
    transform = get_transforms(image_size=image_size)

    train_data = datasets.ImageFolder(root=data_root / "train", transform=transform)
    val_data = datasets.ImageFolder(root=data_root / "val", transform=transform)
    test_data = datasets.ImageFolder(root=data_root / "test", transform=transform)

    return train_data, val_data, test_data


def create_dataloaders(
    data_root: Path,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
) -> Tuple[Dict[str, DataLoader], list, dict]:
    """Create dataloaders and return class metadata."""
    train_data, val_data, test_data = create_datasets(data_root, image_size=image_size)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    return loaders, train_data.classes, train_data.class_to_idx
