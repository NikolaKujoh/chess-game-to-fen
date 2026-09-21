"""
CNN koji prepoznaje koja je figura na (zauzetom) polju table (faza 3).

Slično kao occupancy_cnn.py, samo veća mreža i 12 klasa umesto 2
(6 tipova figura x 2 boje). Uči se na cropovima iz
data/processed/pieces/train|val|test preko ImageFolder-a. Na kraju
treninga ispisuje matricu konfuzije da se vidi koje figure mreža najviše
meša (npr. top/kraljica iz daljine).

Korišćenje:
    python piece_cnn.py --data ../data/processed/pieces --epochs 20
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class PieceCNN(nn.Module):
    def __init__(self, img_size: int = 96, num_classes: int = 12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            n_flat = self.features(dummy).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_loaders(data_root: Path, img_size, batch_size, num_workers):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_root / "val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(data_root / "test",  transform=eval_tf)

    classes = train_ds.classes
    print(f"Klasa ({len(classes)}): {classes}")

    use_persistent = num_workers > 0

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, 
                              persistent_workers=use_persistent)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=num_workers, pin_memory=True, 
                            persistent_workers=use_persistent)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=True, 
                             persistent_workers=use_persistent)
    return train_loader, val_loader, test_loader, classes


@torch.inference_mode()
def evaluate(model, loader, device, num_classes):
    model.eval()
    correct = total = 0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(1)
        correct += (preds == y).sum().item()
        total += y.numel()
        for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
            conf[t, p] += 1
    return correct / total, conf


def print_confusion(conf, classes):
    print("\nTačnost po klasama:")
    for i, name in enumerate(classes):
        row_sum = conf[i].sum()
        acc = conf[i, i] / row_sum if row_sum else 0.0
        print(f"  {name:14s}: {acc*100:5.1f}%  (n={row_sum})")

    short = [c.replace("white-", "w").replace("black-", "b")[:3] for c in classes]
    print("\nMatrica konfuzije (redovi=tačno, kolone=predviđeno):")
    header = "      " + " ".join(f"{s:>4s}" for s in short)
    print(header)
    for i, name in enumerate(short):
        row = " ".join(f"{conf[i, j]:>4d}" for j in range(len(classes)))
        print(f"  {name:>3s} {row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", type=str, default="../models/piece_cnn.pth")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Uređaj: {device}")

    data_root = Path(args.data)
    train_loader, val_loader, test_loader, classes = make_loaders(
        data_root, args.img_size, args.batch_size, args.num_workers)
    num_classes = len(classes)

    model = PieceCNN(img_size=args.img_size, num_classes=num_classes).to(device)
    print(f"Broj parametara modela: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = seen = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        
        scheduler.step()
        val_acc, _ = evaluate(model, val_loader, device, num_classes)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoha {epoch:02d}/{args.epochs} | lr {current_lr:.6f} | train loss {running/seen:.4f} | "
              f"val acc {val_acc*100:.2f}%")
        
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), out_path)

    model.load_state_dict(torch.load(out_path, weights_only=True))
    test_acc, conf = evaluate(model, test_loader, device, num_classes)
    print(f"\n── TESTIRANJE ── tačnost {test_acc*100:.2f}%  (najbolji val {best_val*100:.2f}%)")
    print_confusion(conf, classes)

    cm_path = out_path.parent / "piece_confusion.txt"
    with open(cm_path, "w") as f:
        f.write("Klase: " + ", ".join(classes) + "\n")
        np.savetxt(f, conf, fmt="%d")
    print(f"\nTežine sačuvane u {out_path.resolve()}")
    print(f"Matrica konfuzije sačuvana u {cm_path.resolve()}")


if __name__ == "__main__":
    main()