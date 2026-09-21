"""
Vizuelna provera warp-a i uglova table.

Uzme par nasumičnih zapisa iz parsed_dataset.json, uradi homografski warp
na osnovu ground-truth uglova i nacrta preko toga 8x8 mrežu sa figurama
(iz FEN-a). Ne pravi nikakav dataset, služi samo da se vizuelno vidi da
li su uglovi tačni i da li warp lepo poravna tablu.

Korišćenje:
    python warp_board.py --dataroot "<dataset>" --num 4
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


CORNER_SQUARES = ["a1", "a8", "h8", "h1"]


def warp_board(image_bgr, corners, size):
    dst_map = {
        "a1": [0, size],
        "a8": [0, 0],
        "h8": [size, 0],
        "h1": [size, size],
    }
    src = np.array(corners, dtype=np.float32)
    dst = np.array([dst_map[sq] for sq in CORNER_SQUARES], dtype=np.float32)
    
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, H, (size, size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--records", default="parsed_dataset.json")
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    with open(args.records) as f:
        records = json.load(f)
        
    print(f"Učitano {len(records)} zapisa.")

    if args.seed is not None:
        random.seed(args.seed)
    sample = random.sample(records, min(args.num, len(records)))

    size = args.size
    cell = size // 8

    cols = 2
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    axes = np.array(axes).reshape(-1)

    for i, rec in enumerate(sample):
        ax = axes[i]
        img_path = dataroot / rec["path"]
        image_bgr = cv2.imread(str(img_path))
        
        if image_bgr is None:
            ax.set_title(f"Fali slika: {img_path}")
            ax.axis("off")
            continue

        warped = warp_board(image_bgr, rec["corners"], size)
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        ax.imshow(warped_rgb)

        # crtanje mreže
        for k in range(9):
            ax.axhline(y=k * cell, color="yellow", linewidth=2, alpha=0.5)
            ax.axvline(x=k * cell, color="yellow", linewidth=2, alpha=0.5)

        # ispis figura iz FEN-a
        board = rec["board"]
        for r in range(8):
            for c in range(8):
                ch = board[r][c]
                if ch == ".":
                    continue
                color = "white" if ch.isupper() else "black"
                ax.text(c * cell + cell / 2, r * cell + cell / 2, ch,
                        color=color, fontsize=12, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="circle", facecolor="gray",
                                  alpha=0.4, edgecolor="none"))

        ax.text(cell * 0.5, -cell * 0.25, "a8", color="red", fontsize=10,
                ha="center", fontweight="bold")
        ax.text(size - cell * 0.5, size + cell * 0.35, "h1", color="red",
                fontsize=10, ha="center", fontweight="bold")

        ax.set_title(f"id={rec['id']} angle={rec['camera_angle']}\n{rec['fen']}", fontsize=8)
        ax.axis("off")

    for j in range(len(sample), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=110, bbox_inches="tight")
        print(f"Sačuvana slika u {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()