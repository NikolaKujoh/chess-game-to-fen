import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .data import create_dataloaders
from .engine import evaluate, train_one_epoch
from .model import ChessModelV2


def parse_args():
    parser = argparse.ArgumentParser(description="Train piece classifier")
    parser.add_argument("--data-root", type=Path, required=True, help="Path containing train/val/test folders")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/piece_classifier"))
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    loaders, classes, class_to_idx = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    print(f"Classes ({len(classes)}): {classes}")

    model = ChessModelV2(
        input_shape=3,
        hidden_units=args.hidden_units,
        output_shape=len(classes),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device)
        val_metrics = evaluate(model, loaders["val"], loss_fn, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.2f}% | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.2f}%"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(model.state_dict(), args.output_dir / "best_model.pth")

    test_metrics = evaluate(model, loaders["test"], loss_fn, device)
    print(f"Test | loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.2f}%")

    with open(args.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(args.output_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    torch.save(model.state_dict(), args.output_dir / "last_model.pth")


if __name__ == "__main__":
    main()
