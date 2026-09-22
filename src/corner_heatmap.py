"""
Alternativa za corner_cnn.py - uglove predviđa preko heatmapa.

Umesto da regresijom direktno predviđa 8 brojeva (x,y za 4 ugla), ova
mreža za svaki ugao izbaci svoju heatmapu (toplotnu mapu) preko slike, a
tačka sa najvišom vrednošću u toj mapi je predviđeni ugao. Backbone je
opet ResNet18, samo se na kraju dodaju dekonvolucije da se vrati na veću
rezoluciju. U testiranju (pipeline.py --corner-mode heatmap) ovaj pristup
je davao bolje rezultate od regresionog.

Korišćenje:
    python corner_heatmap.py --dataroot "<dataset>"
    python corner_heatmap.py --dataroot "<dataset>" --visualize --num 4
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image


N_CORNERS = 4
IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def gaussian_heatmaps(corners_hm, hm_size, sigma=2.0):
    H = np.zeros((N_CORNERS, hm_size, hm_size), dtype=np.float32)
    xs = np.arange(hm_size)
    ys = np.arange(hm_size)[:, None]
    for k, (cx, cy) in enumerate(corners_hm):
        H[k] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return H


class HeatmapDataset(Dataset):
    def __init__(self, records, dataroot, split, img_size, hm_size, train, sigma=2.0):
        self.items = [r for r in records if r.get("split") == split]
        self.dataroot = Path(dataroot)
        self.img_size = img_size
        self.hm_size = hm_size
        self.sigma = sigma
        aug = [transforms.ColorJitter(0.2, 0.2, 0.2)] if train else []
        self.tf = transforms.Compose(
            [transforms.Resize((img_size, img_size))] + aug +
            [transforms.ToTensor(), transforms.Normalize(*IMNET)]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img = Image.open(self.dataroot / rec["path"]).convert("RGB")
        W, Hh = img.size
        corners_hm = []
        for (x, y) in rec["corners"]:
            corners_hm.append([x * self.hm_size / W, y * self.hm_size / Hh])
        target = gaussian_heatmaps(np.array(corners_hm), self.hm_size, self.sigma)
        return (self.tf(img), torch.from_numpy(target),
                torch.tensor([W, Hh], dtype=torch.float32))


BACKBONES = {
    "resnet18": (models.resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
}


class HeatmapNet(nn.Module):
    def __init__(self, n_deconv=4, backbone_name="resnet18"):
        super().__init__()
        ctor, weights = BACKBONES[backbone_name]
        backbone = ctor(weights=weights)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        layers = []
        in_ch = 512
        chans = [256, 128, 64, 32][:n_deconv]
        for out_ch in chans:
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.decoder = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_ch, N_CORNERS, 1)

    def forward(self, x):
        return self.head(self.decoder(self.encoder(x)))

    def set_encoder_trainable(self, flag):
        for p in self.encoder.parameters():
            p.requires_grad = flag


def decode_heatmaps(hm, wh):
    B, K, h, w = hm.shape
    hm_np = hm.detach().cpu().numpy()
    out = np.zeros((B, K, 2), dtype=np.float32)
    for b in range(B):
        W, Hh = wh[b].tolist()
        for k in range(K):
            m = hm_np[b, k]
            py, px = np.unravel_index(np.argmax(m), m.shape)
            dx = dy = 0.0
            if 0 < px < w - 1:
                dx = 0.25 * np.sign(m[py, px + 1] - m[py, px - 1])
            if 0 < py < h - 1:
                dy = 0.25 * np.sign(m[py + 1, px] - m[py - 1, px])
            fx, fy = px + dx, py + dy
            out[b, k, 0] = fx * W / w
            out[b, k, 1] = fy * Hh / h
    return out


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total = n = 0
    for X, target, wh in loader:
        pred_hm = model(X.to(device))
        pred_px = decode_heatmaps(pred_hm, wh)
        B, K, h, w = target.shape
        tnp = target.numpy()
        for b in range(B):
            W, Hh = wh[b].tolist()
            for k in range(K):
                py, px = np.unravel_index(np.argmax(tnp[b, k]), (h, w))
                tx, ty = px * W / w, py * Hh / h
                d = np.hypot(pred_px[b, k, 0] - tx, pred_px[b, k, 1] - ty)
                total += d
                n += 1
    return total / n


def run_phase(model, loader, val_loader, device, epochs, lr, name, best, out_path):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()
    for e in range(1, epochs + 1):
        model.train()
        run = seen = 0
        for X, target, _ in loader:
            X, target = X.to(device), target.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X), target)
            loss.backward()
            opt.step()
            run += loss.item() * X.size(0)
            seen += X.size(0)
        px = evaluate(model, val_loader, device)
        print(f"[{name}] epoha {e}/{epochs} | train loss {run/seen:.6f} | val px greška {px:.1f}")
        if px < best["px"]:
            best["px"] = px
            torch.save(model.state_dict(), out_path)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--records", default="chesscog_parsed.json")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--n-deconv", type=int, default=4)
    ap.add_argument("--backbone", choices=list(BACKBONES), default="resnet18",
                    help="resnet34 = more capacity, usually a bit lower px error")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--freeze-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=12)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default="../models/corner_heatmap.pth")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--num", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Uređaj: {device}")
    hm_size = args.img_size // (2 ** (5 - args.n_deconv))
    print(f"Slika {args.img_size} -> topološka mapa {hm_size}x{hm_size}")

    with open(args.records) as f:
        records = json.load(f)

    def mk(split, train):
        return HeatmapDataset(records, args.dataroot, split, args.img_size,
                              hm_size, train, args.sigma)

    train_loader = DataLoader(mk("train", True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(mk("val", False), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(mk("test", False), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = HeatmapNet(n_deconv=args.n_deconv, backbone_name=args.backbone).to(device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        import matplotlib.pyplot as plt
        model.load_state_dict(torch.load(out_path, map_location=device))
        model.eval()
        ds = mk("test", False)
        if args.seed is not None:
            random.seed(args.seed)
        idxs = random.sample(range(len(ds)), min(args.num, len(ds)))
        cols = 2
        rows = (len(idxs) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 7*rows))
        axes = np.array(axes).reshape(-1)
        for i, di in enumerate(idxs):
            X, target, wh = ds[di]
            rec = ds.items[di]
            img = Image.open(Path(args.dataroot)/rec["path"]).convert("RGB")
            with torch.inference_mode():
                hm = model(X.unsqueeze(0).to(device))
            pred = decode_heatmaps(hm, wh.unsqueeze(0))[0]
            true = np.array(rec["corners"], dtype=np.float32)
            ax = axes[i]
            ax.imshow(img)
            ax.plot(*true.T, "go", ms=12, mec="k", label="tačno")
            ax.plot(*pred.T, "rx", ms=14, mew=3, label="predviđeno")
            ax.set_title(f"id={rec['id']} ugao={rec['camera_angle']}", fontsize=9)
            ax.legend(fontsize=8)
            ax.axis("off")
        for j in range(len(idxs), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.show()
        return

    best = {"px": float("inf")}
    print("\n=== Faza 1: glava + dekoder (zamrznut enkoder) ===")
    model.set_encoder_trainable(False)
    best = run_phase(model, train_loader, val_loader, device,
                     args.freeze_epochs, args.head_lr, "glava", best, out_path)

    print("\n=== Faza 2: fino podešavanje (celo stablo) ===")
    model.set_encoder_trainable(True)
    best = run_phase(model, train_loader, val_loader, device,
                     args.finetune_epochs, args.finetune_lr, "ft", best, out_path)

    model.load_state_dict(torch.load(out_path))
    px = evaluate(model, test_loader, device)
    print(f"\n── TESTIRANJE ── prosečna greška po uglu {px:.1f}px  (najbolji val {best['px']:.1f}px)")
    print(f"Težine sačuvane u {out_path.resolve()}")


if __name__ == "__main__":
    main()