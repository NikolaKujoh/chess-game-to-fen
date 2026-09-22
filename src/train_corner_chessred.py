"""
Trening heatmap detektora uglova table na ChessReD (realne fotografije),
od nule, sa game-level splitom identičnim onom za YOLO piece detektor
(resplit_chessred_by_game.py).

Arhitektura je ista kao u corner_heatmap.py (ResNet18 enkoder + transponovane
konvolucije), samo se uči na stvarnim fotografijama umesto na sintetičkom
chesscog skupu - taj sintetički model (models/corner_heatmap.pth) loše
prenosi na stvarne slike (domain gap), pa je potreban poseban model za
ChessReD/telefonske fotografije.

KRITIČNO: split se računa istom logikom (seed, n_folds, fold) kao u
resplit_chessred_by_game.py, pa iste partije završavaju u istim
splitovima. Bez toga bi corner model video test slike piece modela i
obrnuto, što bi kontaminiralo end-to-end evaluaciju.

Korišćenje:
    python train_corner_chessred.py \
        --annotations /content/annotations.json \
        --dataroot /content/chessred \
        --out /content/models/corner_heatmap_chessred.pth \
        --n-folds 5 --fold 0 --seed 42 \
        --img-size 256 --batch-size 16 --finetune-epochs 15
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

Image.MAX_IMAGE_PIXELS = None  # ChessReD slike su 3072x3072

N_CORNERS = 4
IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# isti redosled kao CORNER_SQUARES u ostatku projekta: a1, a8, h8, h1
CORNER_ORDER = ["bottom_left", "top_left", "top_right", "bottom_right"]


# ---------------------------------------------------------------- split

def split_games_kfold(game_ids, n_folds, fold, seed):
    games = sorted(game_ids)
    random.Random(seed).shuffle(games)
    folds = [[] for _ in range(n_folds)]
    for i, g in enumerate(games):
        folds[i % n_folds].append(g)
    test_games = set(folds[fold])
    val_games = set(folds[(fold + 1) % n_folds])
    train_games = set(games) - test_games - val_games
    return train_games, val_games, test_games


def split_games_single(game_ids, ratio_str, seed):
    games = sorted(game_ids)
    random.Random(seed).shuffle(games)
    parts = [float(x) for x in ratio_str.split("/")]
    total = sum(parts)
    n = len(games)
    n_train = round(n * parts[0] / total)
    n_val = round(n * parts[1] / total)
    return (set(games[:n_train]), set(games[n_train:n_train + n_val]),
            set(games[n_train + n_val:]))


def build_records(annotations_path, dataroot, n_folds, fold, ratio, seed):
    with open(annotations_path) as f:
        raw = json.load(f)

    images = {im["id"]: im for im in raw["images"]}
    corners_by_img = {c["image_id"]: c["corners"] for c in raw["annotations"]["corners"]}

    annotated_ids = sorted(corners_by_img)
    game_ids = set(images[i]["game_id"] for i in annotated_ids)

    if n_folds:
        train_g, val_g, test_g = split_games_kfold(game_ids, n_folds, fold, seed)
        print(f"K-fold split (n_folds={n_folds}, fold={fold}, seed={seed})")
    else:
        train_g, val_g, test_g = split_games_single(game_ids, ratio, seed)
        print(f"Single split (ratio={ratio}, seed={seed})")

    print(f"  partije: train={len(train_g)} val={len(val_g)} test={len(test_g)}")
    print(f"  test partije: {sorted(test_g)}")

    game_to_split = {}
    for g in train_g:
        game_to_split[g] = "train"
    for g in val_g:
        game_to_split[g] = "val"
    for g in test_g:
        game_to_split[g] = "test"

    dataroot = Path(dataroot)
    records = []
    missing = 0
    for img_id in annotated_ids:
        meta = images[img_id]
        rel = meta["path"].replace("\\", "/")
        if rel.startswith("images/"):
            rel = rel[len("images/"):]
        full = dataroot / rel
        if not full.exists():
            missing += 1
            continue
        cd = corners_by_img[img_id]
        records.append({
            "id": img_id,
            "path": str(full),
            "split": game_to_split[meta["game_id"]],
            "corners": [cd[name] for name in CORNER_ORDER],
            "w": meta["width"],
            "h": meta["height"],
        })

    counts = {s: sum(1 for r in records if r["split"] == s) for s in ("train", "val", "test")}
    print(f"  slika: {counts}  (nedostaje fajlova: {missing})")
    return records


# ---------------------------------------------------------------- data

def gaussian_heatmaps(corners_hm, hm_size, sigma=2.0):
    H = np.zeros((N_CORNERS, hm_size, hm_size), dtype=np.float32)
    xs = np.arange(hm_size)
    ys = np.arange(hm_size)[:, None]
    for k, (cx, cy) in enumerate(corners_hm):
        H[k] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return H


class ChessReDCornerDataset(Dataset):
    def __init__(self, records, split, img_size, hm_size, train, sigma=2.0):
        self.items = [r for r in records if r["split"] == split]
        self.img_size = img_size
        self.hm_size = hm_size
        self.sigma = sigma
        aug = [transforms.ColorJitter(0.3, 0.3, 0.3)] if train else []
        self.tf = transforms.Compose(
            [transforms.Resize((img_size, img_size))] + aug +
            [transforms.ToTensor(), transforms.Normalize(*IMNET)]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img = Image.open(rec["path"]).convert("RGB")
        W, H = img.size
        corners_hm = [[x * self.hm_size / W, y * self.hm_size / H]
                      for (x, y) in rec["corners"]]
        target = gaussian_heatmaps(np.array(corners_hm), self.hm_size, self.sigma)
        return (self.tf(img), torch.from_numpy(target),
                torch.tensor([W, H], dtype=torch.float32))


# --------------------------------------------------------------- model

class HeatmapNet(nn.Module):
    def __init__(self, n_deconv=4):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        layers = []
        in_ch = 512
        for out_ch in [256, 128, 64, 32][:n_deconv]:
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
            out[b, k, 0] = (px + dx) * W / w
            out[b, k, 1] = (py + dy) * Hh / h
    return out


@torch.inference_mode()
def evaluate(model, loader, device):
    """Srednja greška po uglu u pikselima ORIGINALNE slike."""
    model.eval()
    total = n = 0.0
    for X, target, wh in loader:
        pred_px = decode_heatmaps(model(X.to(device)), wh)
        B, K, h, w = target.shape
        tnp = target.numpy()
        for b in range(B):
            W, Hh = wh[b].tolist()
            for k in range(K):
                py, px = np.unravel_index(np.argmax(tnp[b, k]), (h, w))
                tx, ty = px * W / w, py * Hh / h
                total += np.hypot(pred_px[b, k, 0] - tx, pred_px[b, k, 1] - ty)
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
        print(f"[{name}] epoha {e}/{epochs} | train loss {run/seen:.6f} "
              f"| val greška {px:.1f}px")
        if px < best["px"]:
            best["px"] = px
            torch.save(model.state_dict(), out_path)
            print(f"    -> novi najbolji, sačuvano")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--out", default="/content/models/corner_heatmap_chessred.pth")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ratio", default="70/15/15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--n-deconv", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--freeze-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=15)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Uređaj: {device}")
    hm_size = args.img_size // (2 ** (5 - args.n_deconv))
    print(f"Ulaz {args.img_size}x{args.img_size} -> heatmap {hm_size}x{hm_size}")

    records = build_records(args.annotations, args.dataroot,
                            args.n_folds, args.fold, args.ratio, args.seed)

    def mk(split, train):
        return ChessReDCornerDataset(records, split, args.img_size, hm_size,
                                     train, args.sigma)

    train_loader = DataLoader(mk("train", True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(mk("val", False), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(mk("test", False), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    model = HeatmapNet(n_deconv=args.n_deconv).to(device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best = {"px": float("inf")}
    print("\n=== Faza 1: dekoder + glava (zamrznut enkoder) ===")
    model.set_encoder_trainable(False)
    best = run_phase(model, train_loader, val_loader, device,
                     args.freeze_epochs, args.head_lr, "glava", best, out_path)

    print("\n=== Faza 2: fino podešavanje cele mreže ===")
    model.set_encoder_trainable(True)
    best = run_phase(model, train_loader, val_loader, device,
                     args.finetune_epochs, args.finetune_lr, "ft", best, out_path)

    model.load_state_dict(torch.load(out_path))
    px = evaluate(model, test_loader, device)

    # greška relativno na dimenziju slike i na veličinu polja
    mean_dim = np.mean([r["w"] for r in records if r["split"] == "test"])
    print(f"\n── TEST (held-out partije) ──")
    print(f"  srednja greška po uglu: {px:.1f} px")
    print(f"  relativno na širinu slike ({mean_dim:.0f}px): {px/mean_dim*100:.3f}%")
    print(f"  ~ {px/(mean_dim/8):.3f} veličine jednog polja")
    print(f"  najbolji val: {best['px']:.1f} px")
    print(f"\nTežine: {out_path.resolve()}")


if __name__ == "__main__":
    main()
