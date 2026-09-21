"""
Faza 4: spaja sve 3 mreže u jedan pipeline slika -> FEN, plus evaluacija.

Redosled: corner mreža nađe 4 ugla -> homografski warp table (sa
headroom-om iznad) -> isečenih 64 polja -> occupancy mreža kaže koje je
polje prazno/zauzeto -> piece mreža prepozna figuru na zauzetim poljima
-> sve se sklopi u FEN.

Sa --eval prolazi kroz ceo test split i računa per-square (svako od 64
polja posebno) i full-board (koliko tabli je 100% tačno) tačnost, i to
dvaput - jednom sa uglovima koje predvidi corner mreža (pravi end-to-end)
i jednom sa ground-truth uglovima (gornja granica, izoluje grešku
occupancy/piece dela od greške u detekciji uglova).

Sa --image-id se umesto celog eval-a testira samo jedna slika iz dataseta
(--show je iscrta rezultat).

Korišćenje:
    python pipeline.py --dataroot "<dataset>" --eval
    python pipeline.py --dataroot "<dataset>" --image-id 0024 --show
    python pipeline.py --dataroot "<dataset>" --eval --corner-mode heatmap
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

CORNER_SQUARES = ["a1", "a8", "h8", "h1"]

PIECE_CLASSES = ['black-bishop', 'black-king', 'black-knight', 'black-pawn',
                 'black-queen', 'black-rook', 'white-bishop', 'white-king',
                 'white-knight', 'white-pawn', 'white-queen', 'white-rook']
NAME_TO_FEN = {
    "white-pawn": "P", "white-rook": "R", "white-knight": "N",
    "white-bishop": "B", "white-queen": "Q", "white-king": "K",
    "black-pawn": "p", "black-rook": "r", "black-knight": "n",
    "black-bishop": "b", "black-queen": "q", "black-king": "k",
}
OCC_CLASSES = ["empty", "occupied"]

class CornerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(in_feat, 8), nn.Sigmoid())
    def forward(self, x): return self.backbone(x)


class HeatmapNet(nn.Module):
    def __init__(self, n_deconv=4):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
        layers = []; in_ch = 512
        for out_ch in [256, 128, 64, 32][:n_deconv]:
            layers += [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.decoder = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_ch, 4, 1)
    def forward(self, x):
        return self.head(self.decoder(self.encoder(x)))


def decode_heatmaps_single(hm, W, H):
    import numpy as _np
    K, h, w = hm.shape
    out = _np.zeros((4, 2), dtype=_np.float32)
    for k in range(K):
        m = hm[k]
        py, px = _np.unravel_index(_np.argmax(m), m.shape)
        dx = dy = 0.0
        if 0 < px < w - 1:
            dx = 0.25 * _np.sign(m[py, px+1] - m[py, px-1])
        if 0 < py < h - 1:
            dy = 0.25 * _np.sign(m[py+1, px] - m[py-1, px])
        out[k, 0] = (px + dx) * W / w
        out[k, 1] = (py + dy) * H / h
    return out


class OccupancyCNN(nn.Module):
    def __init__(self, img_size=100, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        with torch.no_grad():
            n = self.features(torch.zeros(1, 3, img_size, img_size)).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(n, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes))
    def forward(self, x): return self.classifier(self.features(x))


class PieceCNN(nn.Module):
    def __init__(self, img_size=96, num_classes=12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        with torch.no_grad():
            n = self.features(torch.zeros(1, 3, img_size, img_size)).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(n, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes))
    def forward(self, x): return self.classifier(self.features(x))

def corner_destinations(size, headroom):
    return {"a1": [0, headroom + size], "a8": [0, headroom],
            "h8": [size, headroom], "h1": [size, headroom + size]}

def warp_from_corners(image_bgr, corners, size, headroom):
    dst_map = corner_destinations(size, headroom)
    src = np.array(corners, dtype=np.float32)
    dst = np.array([dst_map[s] for s in CORNER_SQUARES], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, H, (size, size + headroom))

def occ_crop(warped, r, c, cell, headroom, occ_pad):
    h, w = warped.shape[:2]
    pad = occ_pad * cell
    y1 = max(0, int(headroom + r*cell - pad)); y2 = min(h, int(headroom + (r+1)*cell + pad))
    x1 = max(0, int(c*cell - pad));           x2 = min(w, int((c+1)*cell + pad))
    return warped[y1:y2, x1:x2]

def piece_crop_fixed(warped, r, c, cell, headroom, top_pad, side_pad, bottom_pad):
    h, w = warped.shape[:2]
    x1 = max(0, int(c*cell - side_pad)); x2 = min(w, int((c+1)*cell + side_pad))
    y1 = max(0, int(headroom + r*cell - top_pad)); y2 = min(h, int(headroom + (r+1)*cell + bottom_pad))
    return warped[y1:y2, x1:x2]

IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def occ_tf(img_size):
    return transforms.Compose([transforms.Resize((img_size, img_size)),
                               transforms.ToTensor(), transforms.Normalize(*IMNET)])
def piece_tf(img_size):
    return transforms.Compose([transforms.Resize((img_size, img_size)),
                               transforms.ToTensor(), transforms.Normalize(*IMNET)])
def corner_tf(img_size):
    return transforms.Compose([transforms.Resize((img_size, img_size)),
                               transforms.ToTensor(), transforms.Normalize(*IMNET)])

def board_to_fen(board):
    rows = []
    for r in range(8):
        s = ""; empt = 0
        for c in range(8):
            ch = board[r][c]
            if ch == ".":
                empt += 1
            else:
                if empt: s += str(empt); empt = 0
                s += ch
        if empt: s += str(empt)
        rows.append(s)
    return "/".join(rows)

class Pipeline:
    def __init__(self, models_dir, device, size=512, headroom_cells=2,
                 occ_pad=0.25, top_pad=2.0, side_pad=0.1, bottom_pad=0.1,
                 corner_img=224, occ_img=64, piece_img=96,
                 corner_mode='regression'):
        self.device = device
        self.size = size
        self.cell = size // 8
        self.headroom = int(headroom_cells * self.cell)
        self.occ_pad = occ_pad
        self.top_pad = top_pad * self.cell
        self.side_pad = side_pad * self.cell
        self.bottom_pad = bottom_pad * self.cell
        self.corner_img = corner_img
        self.occ_img = occ_img
        self.piece_img = piece_img

        md = Path(models_dir)
        self.corner_mode = corner_mode
        if corner_mode == "heatmap":
            self.corner_img = 256
            self.corner = HeatmapNet().to(device)
            self.corner.load_state_dict(torch.load(md / "corner_heatmap.pth", map_location=device))
        else:
            self.corner = CornerNet().to(device)
            self.corner.load_state_dict(torch.load(md / "corner_cnn.pth", map_location=device))
        self.corner.eval()
        self.occ = OccupancyCNN(img_size=occ_img).to(device)
        self.occ.load_state_dict(torch.load(md / "occupancy_cnn.pth", map_location=device))
        self.occ.eval()
        self.piece = PieceCNN(img_size=piece_img).to(device)
        self.piece.load_state_dict(torch.load(md / "piece_cnn.pth", map_location=device))
        self.piece.eval()

        self.ctf = corner_tf(self.corner_img if corner_mode=='heatmap' else corner_img)
        self.otf = occ_tf(occ_img)
        self.ptf = piece_tf(piece_img)

    @torch.inference_mode()
    def predict_corners(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        W, H = pil.size
        x = self.ctf(pil).unsqueeze(0).to(self.device)
        if self.corner_mode == "heatmap":
            hm = self.corner(x)[0].cpu().numpy()
            return decode_heatmaps_single(hm, W, H)
        out = self.corner(x).cpu().view(4, 2).numpy()
        out[:, 0] *= W; out[:, 1] *= H
        return out

    @torch.inference_mode()
    def board_from_image(self, image_bgr, corners):
        warped = warp_from_corners(image_bgr, corners, self.size, self.headroom)

        occ_batch = []
        for r in range(8):
            for c in range(8):
                crop = occ_crop(warped, r, c, self.cell, self.headroom, self.occ_pad)
                occ_batch.append(self.otf(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
        occ_batch = torch.stack(occ_batch).to(self.device)
        occ_pred = self.occ(occ_batch).argmax(1).cpu().numpy()

        board = [["." for _ in range(8)] for _ in range(8)]
        occ_idx = [i for i in range(64) if occ_pred[i] == OCC_CLASSES.index("occupied")]
        if occ_idx:
            pbatch = []
            for i in occ_idx:
                r, c = divmod(i, 8)
                crop = piece_crop_fixed(warped, r, c, self.cell, self.headroom,
                                        self.top_pad, self.side_pad, self.bottom_pad)
                pbatch.append(self.ptf(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
            pbatch = torch.stack(pbatch).to(self.device)
            ppred = self.piece(pbatch).argmax(1).cpu().numpy()
            for k, i in enumerate(occ_idx):
                r, c = divmod(i, 8)
                board[r][c] = NAME_TO_FEN[PIECE_CLASSES[ppred[k]]]
        return board


def board_equal_counts(pred, true):
    correct = sum(1 for r in range(8) for c in range(8) if pred[r][c] == true[r][c])
    return correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--records", default="chesscog_parsed.json")
    ap.add_argument("--models", default="../models")
    ap.add_argument("--corner-mode", choices=["regression", "heatmap"],
                    default="regression")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--image-id", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--use-true-corners", action="store_true",
                    help="single-image: use ground-truth corners instead of predicted")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    dataroot = Path(args.dataroot)
    with open(args.records) as f:
        records = json.load(f)

    pipe = Pipeline(args.models, device, corner_mode=args.corner_mode)

    # ---- single image ----
    if args.image_id is not None:
        rec = next(r for r in records if r["id"] == args.image_id)
        img = cv2.imread(str(dataroot / rec["path"]))
        true_corners = np.array(rec["corners"], dtype=np.float32)
        corners = true_corners if args.use_true_corners else pipe.predict_corners(img)
        board = pipe.board_from_image(img, corners)
        pred_fen = board_to_fen(board)
        true_fen = rec["fen"].split()[0]
        correct = board_equal_counts(board, rec["board"])
        print(f"\nimage {args.image_id}")
        print(f"  pred FEN: {pred_fen}")
        print(f"  true FEN: {true_fen}")
        print(f"  squares correct: {correct}/64")
        if args.show:
            import matplotlib.pyplot as plt
            warped = warp_from_corners(img, corners, pipe.size, pipe.headroom)
            fig, ax = plt.subplots(figsize=(7, 8))
            ax.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            cell = pipe.cell
            for k in range(9):
                ax.axhline(pipe.headroom + k*cell, color="yellow", lw=1, alpha=0.5)
                ax.axvline(k*cell, color="yellow", lw=1, alpha=0.5)
            for r in range(8):
                for c in range(8):
                    ch = board[r][c]
                    if ch == ".": continue
                    ax.text(c*cell + cell/2, pipe.headroom + r*cell + cell/2, ch,
                            color="white" if ch.isupper() else "black",
                            fontsize=12, fontweight="bold", ha="center", va="center",
                            bbox=dict(boxstyle="circle", facecolor="gray", alpha=0.4))
            ax.set_title(f"{args.image_id}  {correct}/64 correct\n{pred_fen}", fontsize=9)
            ax.axis("off"); plt.tight_layout(); plt.show()
        return

    # ---- full test eval ----
    if args.eval:
        test = [r for r in records if r.get("split") == "test"]
        print(f"Evaluating {len(test)} test boards...\n")

        for mode in ("predicted", "true"):
            sq_correct = sq_total = 0
            board_correct = 0
            for n, rec in enumerate(test):
                img = cv2.imread(str(dataroot / rec["path"]))
                if mode == "true":
                    corners = np.array(rec["corners"], dtype=np.float32)
                else:
                    corners = pipe.predict_corners(img)
                board = pipe.board_from_image(img, corners)
                c = board_equal_counts(board, rec["board"])
                sq_correct += c; sq_total += 64
                if c == 64:
                    board_correct += 1
                if (n + 1) % 100 == 0:
                    print(f"  [{mode} corners] {n+1}/{len(test)}...")
            print(f"\n=== {mode.upper()} CORNERS ===")
            print(f"  per-square accuracy : {sq_correct/sq_total*100:.2f}%")
            print(f"  full-board accuracy : {board_correct/len(test)*100:.2f}% "
                  f"({board_correct}/{len(test)})\n")


if __name__ == "__main__":
    main()
