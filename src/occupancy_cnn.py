"""
CNN koji za jedno polje table kaže da li je prazno ili zauzeto (faza 2).

Mala konvolutivna mreža (3 conv+pool bloka) trenirana na cropovima koje
napravi export_cells.py (data/processed/occupancy/train|val|test, po
folderima empty/occupied preko ImageFolder-a). Binarni problem, čuva se
najbolji checkpoint po F1 meri (bitnije od gole tačnosti jer klase nisu
uvek izbalansirane).

Korišćenje:
    python occupancy_cnn.py --data ../data/processed/occupancy --epochs 10
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class OccupancyCNN(nn.Module):
    def __init__(self, img_size: int = 64, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            n_flat = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_loaders(data_root: Path, img_size: int, batch_size: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

    print("Klase i indeksi:", train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.class_to_idx


@torch.inference_mode()
def evaluate(model, loader, device, occupied_idx):
    model.eval()
    correct = total = 0
    tp = fp = fn = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(1)
        correct += (preds == y).sum().item()
        total += y.numel()
        tp += ((preds == occupied_idx) & (y == occupied_idx)).sum().item()
        fp += ((preds == occupied_idx) & (y != occupied_idx)).sum().item()
        fn += ((preds != occupied_idx) & (y == occupied_idx)).sum().item()

    acc = correct / total
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", type=str, default="../models/occupancy_cnn.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Uređaj: {device}")

    data_root = Path(args.data)
    train_loader, val_loader, test_loader, c2i = make_loaders(
        data_root, args.img_size, args.batch_size, args.num_workers)
    occupied_idx = c2i["occupied"]

    model = OccupancyCNN(img_size=args.img_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Broj parametara modela: {n_params:,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        train_loss = running / seen

        acc, prec, rec, f1 = evaluate(model, val_loader, device, occupied_idx)
        print(f"Epoha {epoch}/{args.epochs} | train loss {train_loss:.4f} | "
              f"val acc {acc*100:.2f}% prec {prec*100:.2f}% "
              f"rec {rec*100:.2f}% f1 {f1*100:.2f}%")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), out_path)

    # Učitavanje najboljeg modela pre testiranja
    if out_path.exists():
        model.load_state_dict(torch.load(out_path))

    acc, prec, rec, f1 = evaluate(model, test_loader, device, occupied_idx)
    print("\n── TESTIRANJE (Najbolji checkpoint) ──")
    print(f"Tačnost {acc*100:.2f}% | Preciznost {prec*100:.2f}% | "
          f"Odziv {rec*100:.2f}% | F1 {f1*100:.2f}%")
    print(f"\nTežine sačuvane u {out_path.resolve()}")


if __name__ == "__main__":
    main()