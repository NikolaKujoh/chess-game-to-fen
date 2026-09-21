"""
Samo evaluacija već istreniranog corner_cnn.py modela, bez treniranja.

Učita models/corner_cnn.pth i izračuna prosečnu grešku po uglu (u
pikselima) na val i test splitu. Sa --visualize flagom nacrta par slika
gde se vidi tačan ugao (zeleno) i predviđen (crveno x), korisno da se na
oko proveri koliko model greši.

Korišćenje:
    python corner_eval.py --dataroot "<dataset>"
    python corner_eval.py --dataroot "<dataset>" --visualize --num 4
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image


class CornerDataset(Dataset):
    def __init__(self, records, dataroot, split, img_size):
        self.items = [r for r in records if r.get("split") == split]
        self.dataroot = Path(dataroot)
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img = Image.open(self.dataroot / rec["path"]).convert("RGB")
        W, H = img.size
        target = []
        for (x, y) in rec["corners"]:
            target.append(x / W); target.append(y / H)
        return (self.tf(img), torch.tensor(target, dtype=torch.float32),
                torch.tensor([W, H], dtype=torch.float32), idx)


class CornerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(in_feat, 8), nn.Sigmoid())

    def forward(self, x):
        return self.backbone(x)


@torch.inference_mode()
def eval_split(model, loader, device):
    model.eval()
    total_px = n = 0
    for X, target, wh, _ in loader:
        pred = model(X.to(device)).cpu()
        B = pred.shape[0]
        pp = pred.view(B, 4, 2) * wh.view(B, 1, 2)
        tp = target.view(B, 4, 2) * wh.view(B, 1, 2)
        d = torch.sqrt(((pp - tp) ** 2).sum(dim=2))
        total_px += d.sum().item(); n += d.numel()
    return total_px / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--records", default="chesscog_parsed.json")
    ap.add_argument("--weights", default="../models/corner_cnn.pth")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--num", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(args.records) as f:
        records = json.load(f)

    model = CornerNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded weights from {args.weights}")

    for split in ("val", "test"):
        ds = CornerDataset(records, args.dataroot, split, args.img_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        px = eval_split(model, loader, device)
        print(f"{split}: mean corner error {px:.1f}px   ({len(ds)} images)")

    if args.visualize:
        import matplotlib.pyplot as plt
        import numpy as np
        ds = CornerDataset(records, args.dataroot, "test", args.img_size)
        if args.seed is not None:
            random.seed(args.seed)
        idxs = random.sample(range(len(ds)), min(args.num, len(ds)))

        cols = 2
        rows = (len(idxs) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
        axes = np.array(axes).reshape(-1)

        model.eval()
        for i, di in enumerate(idxs):
            X, target, wh, _ = ds[di]
            rec = ds.items[di]
            img = Image.open(Path(args.dataroot) / rec["path"]).convert("RGB")
            W, H = img.size
            with torch.inference_mode():
                pred = model(X.unsqueeze(0).to(device)).cpu().view(4, 2).numpy()
            pred_px = pred * np.array([W, H])
            true_px = target.view(4, 2).numpy() * np.array([W, H])

            ax = axes[i]
            ax.imshow(img)
            ax.plot(*true_px.T, "go", markersize=12, label="true", markeredgecolor="k")
            ax.plot(*pred_px.T, "rx", markersize=14, markeredgewidth=3, label="pred")
            ax.set_title(f"id={rec['id']} angle={rec['camera_angle']}", fontsize=9)
            ax.legend(fontsize=8); ax.axis("off")
        for j in range(len(idxs), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(args.save, dpi=110, bbox_inches="tight") if args.save else plt.show()


if __name__ == "__main__":
    main()
