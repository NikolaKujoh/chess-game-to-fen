"""
Warp ChessReD2K boards to a top-down view and verify orientation.

Takes the ground-truth corners, computes a homography to a square, warps the
board, overlays an 8x8 grid, and stamps each square with its ground-truth piece
label (from the parsed board grid). This confirms BOTH:
  - the warp geometry is correct (board fills the square, grid lines on edges)
  - the orientation is correct (a8 top-left, h1 bottom-right)

Reads chessred2k_parsed.json (produced by parse_chessred2k.py).

Usage (run from src):
    python warp_check.py --dataroot ".."
    python warp_check.py --dataroot ".." --num 4 --seed 3 --size 512
    python warp_check.py --dataroot ".." --save warp_check.png
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]

PIECE_TO_FEN = {
    0: "P", 1: "R", 2: "N", 3: "B", 4: "Q", 5: "K",
    6: "p", 7: "r", 8: "n", 9: "b", 10: "q", 11: "k",
    12: ".",
}
EMPTY = 12


def warp_board(image_bgr, corners_dict, size=512):
    """Warp the board to a top-down `size`x`size` image.

    Mapping (white's perspective):
        top_left     -> (0, 0)
        top_right    -> (size, 0)
        bottom_right -> (size, size)
        bottom_left  -> (0, size)
    => square (row r, col c) in the output corresponds to board[r][c],
       with row 0 = rank 8 (top) and col 0 = file a (left).
    """
    src = np.array([corners_dict[name] for name in CORNER_ORDER], dtype=np.float32)
    dst = np.array([
        [0, 0],
        [size, 0],
        [size, size],
        [0, size],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, H, (size, size))
    return warped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True,
                        help="Folder containing chessred2k_parsed.json")
    parser.add_argument("--images-root", type=str, default=None,
                        help="Folder containing the images/ tree "
                             "(defaults to --dataroot if not given)")
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--size", type=int, default=512,
                        help="Output square size in pixels (default 512)")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    images_root = Path(args.images_root) if args.images_root else dataroot

    with open(dataroot / "chessred2k_parsed.json", "r") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} parsed records.")

    if args.seed is not None:
        random.seed(args.seed)
    sample = random.sample(records, min(args.num, len(records)))

    size = args.size
    cell = size // 8

    n = len(sample)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    axes = np.array(axes).reshape(-1)

    for i, rec in enumerate(sample):
        ax = axes[i]
        img_path = images_root / rec["path"]

        if not img_path.exists():
            ax.set_title(f"MISSING: {img_path}")
            ax.axis("off")
            continue

        # OpenCV loads BGR; convert to RGB for display
        image_bgr = cv2.imread(str(img_path))
        warped_bgr = warp_board(image_bgr, rec["corners"], size=size)
        warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)

        ax.imshow(warped_rgb)

        # 8x8 grid lines
        for k in range(9):
            ax.axhline(y=k * cell, color="yellow", linewidth=1, alpha=0.6)
            ax.axvline(x=k * cell, color="yellow", linewidth=1, alpha=0.6)

        # Stamp ground-truth piece labels in each square
        board = rec["board"]
        for r in range(8):
            for c in range(8):
                cat = board[r][c]
                if cat == EMPTY:
                    continue
                ch = PIECE_TO_FEN[cat]
                color = "white" if cat <= 5 else "black"
                ax.text(c * cell + cell / 2, r * cell + cell / 2, ch,
                        color=color, fontsize=13, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="circle", facecolor="gray", alpha=0.4,
                                  edgecolor="none"))

        # Mark a8 (top-left) and h1 (bottom-right) for orientation sanity
        ax.text(cell * 0.5, -cell * 0.25, "a8", color="red", fontsize=10,
                ha="center", fontweight="bold")
        ax.text(size - cell * 0.5, size + cell * 0.35, "h1", color="red",
                fontsize=10, ha="center", fontweight="bold")

        ax.set_title(f"id={rec['image_id']}  {rec['file_name']}\nFEN: {rec['fen']}",
                     fontsize=8)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=110, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()