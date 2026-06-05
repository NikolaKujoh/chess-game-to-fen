"""
Slice warped ChessReD2K boards into 64 cell crops (with upward padding for
tall pieces) and export them for the occupancy + piece classifiers.

Two modes:
  --preview : show one board's 64 cell crops in an 8x8 grid (verify visually)
  (default) : mass-export all cells into folder layouts:

      <out>/occupancy/<split>/empty/*.png
      <out>/occupancy/<split>/occupied/*.png
      <out>/pieces/<split>/<class_name>/*.png

Cropping geometry for cell (r, c) on a `size`x`size` warped board:
    cell = size // 8
    x1 = c*cell - side_pad
    x2 = (c+1)*cell + side_pad
    y1 = r*cell - top_pad      <- big, captures tall pieces leaning up
    y2 = (r+1)*cell + bottom_pad
  then clamp to image bounds and resize to out_size.

Reads chessred2k_parsed.json (from parse_chessred2k.py).

Usage (from src):
    python export_cells.py --dataroot "." --images-root ".." --preview --seed 3
    python export_cells.py --dataroot "." --images-root ".." --out ../data/processed
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]

CATEGORIES = {
    0: "white-pawn",   1: "white-rook",   2: "white-knight",
    3: "white-bishop", 4: "white-queen",  5: "white-king",
    6: "black-pawn",   7: "black-rook",   8: "black-knight",
    9: "black-bishop", 10: "black-queen", 11: "black-king",
    12: "empty",
}
EMPTY = 12


def warp_board(image_bgr, corners_dict, size):
    src = np.array([corners_dict[name] for name in CORNER_ORDER], dtype=np.float32)
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, H, (size, size))


def crop_cell(warped, r, c, cell, top_pad, side_pad, bottom_pad):
    """Crop one padded cell from a warped board, clamped to image bounds."""
    h, w = warped.shape[:2]
    x1 = max(0, int(c * cell - side_pad))
    x2 = min(w, int((c + 1) * cell + side_pad))
    y1 = max(0, int(r * cell - top_pad))
    y2 = min(h, int((r + 1) * cell + bottom_pad))
    return warped[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True,
                        help="Folder containing chessred2k_parsed.json")
    parser.add_argument("--images-root", type=str, default=None,
                        help="Folder containing images/ (defaults to --dataroot)")
    parser.add_argument("--out", type=str, default="../data/processed",
                        help="Output root for exported cells")
    parser.add_argument("--size", type=int, default=512,
                        help="Warped board size in px (default 512)")
    parser.add_argument("--out-size", type=int, default=96,
                        help="Output cell size in px (default 96)")
    # Padding as fractions of one cell
    parser.add_argument("--top-pad", type=float, default=1.0,
                        help="Upward padding as fraction of a cell (default 1.0)")
    parser.add_argument("--side-pad", type=float, default=0.0,
                        help="Left/right padding as fraction of a cell (default 0)")
    parser.add_argument("--bottom-pad", type=float, default=0.0,
                        help="Downward padding as fraction of a cell (default 0)")
    parser.add_argument("--preview", action="store_true",
                        help="Show one board's 64 cells instead of exporting")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", type=str, default=None,
                        help="In preview mode, save the figure to this path")
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    images_root = Path(args.images_root) if args.images_root else dataroot

    with open(dataroot / "chessred2k_parsed.json", "r") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} parsed records.")

    size = args.size
    cell = size // 8
    top_pad = args.top_pad * cell
    side_pad = args.side_pad * cell
    bottom_pad = args.bottom_pad * cell

    if args.seed is not None:
        random.seed(args.seed)

    # ── Preview mode ─────────────────────────────────────────────────────────
    if args.preview:
        import matplotlib.pyplot as plt

        rec = random.choice(records)
        img_path = images_root / rec["path"]
        image_bgr = cv2.imread(str(img_path))
        warped = warp_board(image_bgr, rec["corners"], size)
        board = rec["board"]

        fig, axes = plt.subplots(8, 8, figsize=(16, 16))
        for r in range(8):
            for c in range(8):
                crop = crop_cell(warped, r, c, cell, top_pad, side_pad, bottom_pad)
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                ax = axes[r, c]
                ax.imshow(crop_rgb)
                cat = board[r][c]
                label = CATEGORIES[cat] if cat != EMPTY else "empty"
                ax.set_title(label, fontsize=7)
                ax.axis("off")

        fig.suptitle(f"id={rec['image_id']}  {rec['file_name']}\n"
                     f"top_pad={args.top_pad} side_pad={args.side_pad} "
                     f"bottom_pad={args.bottom_pad}", fontsize=11)
        plt.tight_layout()
        if args.save:
            plt.savefig(args.save, dpi=90, bbox_inches="tight")
            print(f"Saved preview to {args.save}")
        else:
            plt.show()
        return

    # ── Export mode ──────────────────────────────────────────────────────────
    out_root = Path(args.out)

    # Pre-make folders
    for split in ("train", "val", "test"):
        (out_root / "occupancy" / split / "empty").mkdir(parents=True, exist_ok=True)
        (out_root / "occupancy" / split / "occupied").mkdir(parents=True, exist_ok=True)
        for cat in range(12):
            (out_root / "pieces" / split / CATEGORIES[cat]).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    occ_counts = {"empty": 0, "occupied": 0}

    for n, rec in enumerate(records):
        split = rec.get("split", "unknown")
        if split not in counts:
            continue

        img_path = images_root / rec["path"]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [WARN] could not read {img_path}, skipping")
            continue

        warped = warp_board(image_bgr, rec["corners"], size)
        board = rec["board"]
        img_id = rec["image_id"]

        for r in range(8):
            for c in range(8):
                crop = crop_cell(warped, r, c, cell, top_pad, side_pad, bottom_pad)
                crop = cv2.resize(crop, (args.out_size, args.out_size))

                cat = board[r][c]
                fname = f"{img_id}_{r}{c}.png"

                # Occupancy layout
                occ = "empty" if cat == EMPTY else "occupied"
                cv2.imwrite(str(out_root / "occupancy" / split / occ / fname), crop)
                occ_counts[occ] += 1

                # Piece layout (occupied only)
                if cat != EMPTY:
                    cls = CATEGORIES[cat]
                    cv2.imwrite(str(out_root / "pieces" / split / cls / fname), crop)

                counts[split] += 1

        if (n + 1) % 100 == 0:
            print(f"  processed {n+1}/{len(records)} boards...")

    print("\nDone.")
    print("Cells per split:", counts)
    print("Occupancy balance:", occ_counts)
    print(f"Output root: {out_root.resolve()}")


if __name__ == "__main__":
    main()
