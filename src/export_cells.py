"""
Pravi dataset za occupancy i piece CNN.

Za svaku sliku iz parsed_dataset.json radi warp table (pomoću uglova) i
iseca svih 64 polja. Svako polje se snima kao posebna sličica u
data/processed/occupancy/<split>/<empty ili occupied>/ i, ako polje nije
prazno, i u data/processed/pieces/<split>/<naziv figure>/.

Ima i --preview, --preview-pieces, --preview-warp flagove za proveru da
li su crop-ovi dobri pre nego što se pokrene export.

Korišćenje:
    python export_cells.py --dataroot "<dataset>" --out ../data/processed
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


CORNER_SQUARES = ["a1", "a8", "h8", "h1"]

FEN_TO_NAME = {
    "P": "white-pawn",   "R": "white-rook",   "N": "white-knight",
    "B": "white-bishop", "Q": "white-queen",  "K": "white-king",
    "p": "black-pawn",   "r": "black-rook",   "n": "black-knight",
    "b": "black-bishop", "q": "black-queen",  "k": "black-king",
}
ALL_CLASSES = list(dict.fromkeys(FEN_TO_NAME.values()))


def corner_destinations(size, headroom):
    return {
        "a1": [0,    headroom + size],
        "a8": [0,    headroom],
        "h8": [size, headroom],
        "h1": [size, headroom + size],
    }


def warp_board(image_bgr, corners, size, headroom):
    dst_map = corner_destinations(size, headroom)
    src = np.array(corners, dtype=np.float32)
    dst = np.array([dst_map[sq] for sq in CORNER_SQUARES], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, H, (size, size + headroom))


def occupancy_crop(warped, r, c, cell, headroom, occ_pad):
    h, w = warped.shape[:2]
    pad = occ_pad * cell
    y1 = max(0, int(headroom + r * cell - pad))
    y2 = min(h, int(headroom + (r + 1) * cell + pad))
    x1 = max(0, int(c * cell - pad))
    x2 = min(w, int((c + 1) * cell + pad))
    return warped[y1:y2, x1:x2]


def piece_crop(warped, r, c, cell, headroom, top_pad, side_pad, bottom_pad):
    h, w = warped.shape[:2]
    x1 = max(0, int(c * cell - side_pad))
    x2 = min(w, int((c + 1) * cell + side_pad))
    y1 = max(0, int(headroom + r * cell - top_pad))
    y2 = min(h, int(headroom + (r + 1) * cell + bottom_pad))
    return warped[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--records", type=str, default="parsed_dataset.json")
    parser.add_argument("--out", type=str, default="../data/processed")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--headroom", type=float, default=2.0,
                        help="Extra space above board, in cells (default 2.0)")
    parser.add_argument("--occ-size", type=int, default=64)
    parser.add_argument("--piece-size", type=int, default=96)
    parser.add_argument("--top-pad", type=float, default=2.0,
                        help="Upward crop padding for piece, in cells (default 2.0)")
    parser.add_argument("--side-pad", type=float, default=0.1)
    parser.add_argument("--bottom-pad", type=float, default=0.1)
    parser.add_argument("--occ-pad", type=float, default=0.25,
                        help="Occupancy crop expansion per side, in cells (default 0.25 = 1.5x)")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-pieces", action="store_true")
    parser.add_argument("--preview-warp", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    with open(args.records, "r") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records.")

    size = args.size
    cell = size // 8
    headroom = int(args.headroom * cell)
    top_pad = args.top_pad * cell
    side_pad = args.side_pad * cell
    bottom_pad = args.bottom_pad * cell
    occ_pad = args.occ_pad

    if args.seed is not None:
        random.seed(args.seed)

    if args.preview_warp:
        import matplotlib.pyplot as plt
        rec = random.choice(records)
        warped = warp_board(cv2.imread(str(dataroot / rec["path"])),
                            rec["corners"], size, headroom)
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        
        for k in range(9):
            ax.axhline(y=headroom + k * cell, color="yellow", linewidth=1, alpha=0.6)
            ax.axvline(x=k * cell, color="yellow", linewidth=1, alpha=0.6)
            
        ax.axhline(y=headroom, color="red", linewidth=2)
        ax.text(5, headroom - 5, "board top edge", color="red", fontsize=10)
        ax.set_title(f"WARP + headroom={args.headroom} cells  id={rec['id']}", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        if args.save:
            plt.savefig(args.save, dpi=110, bbox_inches="tight")
            print(f"Saved {args.save}")
        else:
            plt.show()
        return

    if args.preview:
        import matplotlib.pyplot as plt
        rec = random.choice(records)
        warped = warp_board(cv2.imread(str(dataroot / rec["path"])),
                            rec["corners"], size, headroom)
        board = rec["board"]
        fig, axes = plt.subplots(8, 8, figsize=(14, 14))
        for r in range(8):
            for c in range(8):
                sq = occupancy_crop(warped, r, c, cell, headroom, occ_pad)
                ax = axes[r, c]
                ax.imshow(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
                ax.set_title("occ" if board[r][c] != "." else "empty", fontsize=7)
                ax.axis("off")
        fig.suptitle(f"OCCUPANCY SQUARES  occ_pad={args.occ_pad}  id={rec['id']}", fontsize=12)
        plt.tight_layout()
        if args.save:
            plt.savefig(args.save, dpi=90, bbox_inches="tight")
            print(f"Saved {args.save}")
        else:
            plt.show()
        return

    if args.preview_pieces:
        import matplotlib.pyplot as plt
        rec = random.choice(records)
        warped = warp_board(cv2.imread(str(dataroot / rec["path"])),
                            rec["corners"], size, headroom)
        board = rec["board"]
        fig, axes = plt.subplots(8, 8, figsize=(14, 14))
        for r in range(8):
            for c in range(8):
                ax = axes[r, c]
                ch = board[r][c]
                if ch == ".":
                    ax.axis("off")
                    continue
                crop = piece_crop(warped, r, c, cell, headroom, top_pad, side_pad, bottom_pad)
                ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                ax.set_title(FEN_TO_NAME[ch], fontsize=7)
                ax.axis("off")
        fig.suptitle(f"PIECE CROPS  top_pad={args.top_pad} headroom={args.headroom}  id={rec['id']}", fontsize=12)
        plt.tight_layout()
        if args.save:
            plt.savefig(args.save, dpi=90, bbox_inches="tight")
            print(f"Saved {args.save}")
        else:
            plt.show()
        return

    out_root = Path(args.out)
    for split in ("train", "val", "test"):
        (out_root / "occupancy" / split / "empty").mkdir(parents=True, exist_ok=True)
        (out_root / "occupancy" / split / "occupied").mkdir(parents=True, exist_ok=True)
        for cls in ALL_CLASSES:
            (out_root / "pieces" / split / cls).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    occ_counts = {"empty": 0, "occupied": 0}
    piece_total = 0

    for n, rec in enumerate(records):
        split = rec.get("split")
        if split not in counts:
            continue

        image_bgr = cv2.imread(str(dataroot / rec["path"]))
        if image_bgr is None:
            print(f"  [WARN] could not read {rec['path']}, skipping")
            continue

        warped = warp_board(image_bgr, rec["corners"], size, headroom)
        board = rec["board"]
        rid = rec["id"]

        for r in range(8):
            for c in range(8):
                ch = board[r][c]
                fname = f"{rid}_{r}{c}.png"

                sq = cv2.resize(occupancy_crop(warped, r, c, cell, headroom, occ_pad),
                                (args.occ_size, args.occ_size))
                occ = "empty" if ch == "." else "occupied"
                cv2.imwrite(str(out_root / "occupancy" / split / occ / fname), sq)
                occ_counts[occ] += 1
                counts[split] += 1

                if ch != ".":
                    pc = cv2.resize(
                        piece_crop(warped, r, c, cell, headroom, top_pad, side_pad, bottom_pad),
                        (args.piece_size, args.piece_size))
                    cv2.imwrite(str(out_root / "pieces" / split / FEN_TO_NAME[ch] / fname), pc)
                    piece_total += 1

        if (n + 1) % 200 == 0:
            print(f"  processed {n+1}/{len(records)} boards...")

    print("\nDone.")
    print("Occupancy cells per split:", counts)
    print("Occupancy balance:", occ_counts)
    print("Piece crops total:", piece_total)
    print(f"Output root: {out_root.resolve()}")


if __name__ == "__main__":
    main()