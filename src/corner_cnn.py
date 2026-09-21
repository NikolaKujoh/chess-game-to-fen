"""
CNN za detekciju 4 ugla table, regresioni pristup (faza 1).

Na ulazu je slika table, na izlazu 8 brojeva - (x,y) za svaki od 4 ugla,
normalizovano na [0,1]. Backbone je ResNet18 sa ImageNet težinama, na
kraju regresiona glava (Linear -> 8, sigmoid). Trenira se u dve faze:
prvo samo glava dok je backbone zamrznut, pa onda fino podešavanje cele
mreže. Najbolje težine (po grešci na val skupu) se čuvaju u
models/corner_cnn.pth.

Vidi i corner_heatmap.py - alternativni pristup preko heatmapa, koji je
u testiranju ispao precizniji od ovog regresionog.

Korišćenje:
    python corner_cnn.py --dataroot "<dataset>" --batch-size 16 --img-size 224
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image


CORNER_ORDER_LEN = 4


class CornerDataset(Dataset):
    def __init__(self, records, dataroot: Path, split: str, img_size: int, train: bool):
        self.items = [r for r in records if r.get("split") == split]
        self.dataroot = dataroot
        self.img_size = img_size

        if train:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
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

        corners = rec["corners"]
        target = []
        for (x, y) in corners:
            target.append(x / W)
            target.append(y / H)
        target = torch.tensor(target, dtype=torch.float32)

        img_t = self.tf(img)
        return img_t, target, torch.tensor([W, H], dtype=torch.float32)


class CornerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_feat, 8),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.backbone(x)

    def set_backbone_trainable(self, trainable: bool):
        for name, p in self.backbone.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = trainable


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total_px = 0.0
    n_corners = 0
    mse_sum = 0.0
    n_vals = 0
    for X, target, wh in loader:
        X = X.to(device)
        pred = model(X).cpu()

        mse_sum += ((pred - target) ** 2).sum().item()
        n_vals += target.numel()

        B = pred.shape[0]
        pred_pts = pred.view(B, 4, 2)
        true_pts = target.view(B, 4, 2)
        wh_ = wh.view(B, 1, 2)
        pred_px = pred_pts * wh_
        true_px = true_pts * wh_
        d = torch.sqrt(((pred_px - true_px) ** 2).sum(dim=2))
        total_px += d.sum().item()
        n_corners += d.numel()

    return total_px / n_corners, mse_sum / n_vals


def run_phase(model, loader, val_loader, device, epochs, lr, phase_name, loss_fn, best, out_path):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        running = seen = 0
        for X, target, _ in loader:
            X, target = X.to(device), target.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X), target)
            loss.backward()
            opt.step()
            running += loss.item() * X.size(0)
            seen += X.size(0)
        px_err, val_mse = evaluate(model, val_loader, device)
        print(f"[{phase_name}] epoch {epoch}/{epochs} | train_loss {running/seen:.5f} "
              f"| val pixel-err {px_err:.1f}px | val mse {val_mse:.5f}")
        if px_err < best["px"]:
            best["px"] = px_err
            torch.save(model.state_dict(), out_path)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--records", default="chesscog_parsed.json")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--freeze-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default="../models/corner_cnn.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataroot = Path(args.dataroot)
    with open(args.records) as f:
        records = json.load(f)

    train_ds = CornerDataset(records, dataroot, "train", args.img_size, train=True)
    val_ds   = CornerDataset(records, dataroot, "val",   args.img_size, train=False)
    test_ds  = CornerDataset(records, dataroot, "test",  args.img_size, train=False)
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = CornerNet().to(device)
    loss_fn = nn.SmoothL1Loss()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = {"px": float("inf")}

    print("\n=== Phase 1: train head (backbone frozen) ===")
    model.set_backbone_trainable(False)
    best = run_phase(model, train_loader, val_loader, device,
                     args.freeze_epochs, args.head_lr, "head", loss_fn, best, out_path)

    print("\n=== Phase 2: fine-tune whole network ===")
    model.set_backbone_trainable(True)
    best = run_phase(model, train_loader, val_loader, device,
                     args.finetune_epochs, args.finetune_lr, "ft", loss_fn, best, out_path)

    model.load_state_dict(torch.load(out_path))
    px_err, mse = evaluate(model, test_loader, device)
    print(f"\n── TEST ── mean corner error {px_err:.1f}px | mse {mse:.5f} "
          f"(best val {best['px']:.1f}px)")
    print(f"Saved best weights to {out_path.resolve()}")


if __name__ == "__main__":
    main()
