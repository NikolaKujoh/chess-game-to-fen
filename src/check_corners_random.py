"""
Visually check corner annotations on a few RANDOM ChessReD2K images.

Reads the parsed records (chessred2k_parsed.json) if present, otherwise falls
back to annotations.json. Draws each board's 4 corners as colored dots with a
red outline tracing the border, so you can confirm the corners land on the
physical board corners.

Colors:
    red    = top_left
    lime   = top_right
    yellow = bottom_right
    cyan   = bottom_left

Usage (run from the src folder):
    python check_corners_random.py --dataroot ".."
    python check_corners_random.py --dataroot ".." --num 4 --seed 7
    python check_corners_random.py --dataroot ".." --save check.png
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]
CORNER_COLORS = {
    "top_left": "red",
    "top_right": "lime",
    "bottom_right": "yellow",
    "bottom_left": "cyan",
}


def load_records(dataroot: Path):
    """Prefer the parsed JSON; fall back to annotations.json."""
    parsed = dataroot / "chessred2k_parsed.json"
    if parsed.exists():
        with open(parsed, "r") as f:
            records = json.load(f)
        return records

    # Fallback: build minimal records straight from annotations.json
    with open(dataroot / "annotations.json", "r") as f:
        raw = json.load(f)
    images = {img["id"]: img for img in raw["images"]}
    records = []
    for c in raw["annotations"]["corners"]:
        meta = images[c["image_id"]]
        records.append({
            "image_id": c["image_id"],
            "file_name": meta["file_name"],
            "path": meta["path"],
            "corners": c["corners"],
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True,
                        help="Folder containing annotations.json / images/ "
                             "(and ideally chessred2k_parsed.json)")
    parser.add_argument("--num", type=int, default=4,
                        help="How many random images to show (default 4)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional path to save the figure instead of showing it")
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    records = load_records(dataroot)
    print(f"Loaded {len(records)} records with corners.")

    if args.seed is not None:
        random.seed(args.seed)

    sample = random.sample(records, min(args.num, len(records)))

    n = len(sample)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    axes = np.array(axes).reshape(-1)

    for i, rec in enumerate(sample):
        ax = axes[i]
        img_path = dataroot / rec["path"]

        if not img_path.exists():
            ax.set_title(f"MISSING: {img_path}")
            ax.axis("off")
            continue

        image = Image.open(img_path).convert("RGB")
        ax.imshow(image)

        corner_dict = rec["corners"]
        pts = np.array([corner_dict[name] for name in CORNER_ORDER])

        # Border outline (closed loop)
        loop = np.vstack([pts, pts[0]])
        ax.plot(loop[:, 0], loop[:, 1], "-", color="red", linewidth=2, alpha=0.7)

        # Colored corner dots + labels
        for name in CORNER_ORDER:
            x, y = corner_dict[name]
            ax.plot(x, y, "o", color=CORNER_COLORS[name], markersize=14,
                    markeredgecolor="black", markeredgewidth=1.5)
            ax.text(x, y, f" {name}", color=CORNER_COLORS[name],
                    fontsize=9, fontweight="bold")

        ax.set_title(f"id={rec['image_id']}  {rec['file_name']}", fontsize=10)
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
