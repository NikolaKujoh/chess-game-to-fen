"""
Glavna demo/inference skripta - slika sa diska u FEN.

Za razliku od pipeline.py (koji radi samo sa slikama iz dataseta, preko
--image-id), ova skripta prima bilo koju sliku sa diska (npr. sopstvenu
fotografiju table) i vraća kompletan FEN zapis.

Ne izvlači boju na potezu, pravo na rokadu, en passant niti brojače
poteza iz same slike (to nije moguće iz jedne fotografije) - te vrednosti
se zadaju kao argumenti ili ostaju na podrazumevanim (standardna početna
pozicija: beli na potezu, sve rokade dozvoljene, bez en passant).

Zahteva da su svi modeli već istrenirani u ../models/.

Pokretanje:
    python predict_fen.py "putanja/do/slike.jpg" --show
    python predict_fen.py "putanja/do/slike.jpg" --corner-mode heatmap --save-vis rezultat.png
"""

import argparse
from pathlib import Path

import cv2
import torch

from pipeline import Pipeline, board_to_fen, warp_from_corners


def predict(image_path, models_dir="../models", corner_mode="regression", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"ne mogu da ucitam sliku: {image_path}")

    pipe = Pipeline(models_dir, device, corner_mode=corner_mode)

    corners = pipe.predict_corners(img)
    board = pipe.board_from_image(img, corners)
    fen_placement = board_to_fen(board)
    warped = warp_from_corners(img, corners, pipe.size, pipe.headroom)

    return fen_placement, board, corners, warped, pipe


def full_fen(fen_placement, active_color="w", castling="KQkq", ep="-",
             halfmove=0, fullmove=1):
    # boja na potezu, rokade, en passant i brojaci poteza ne mogu da se
    # izvuku iz same slike, pa se uzimaju podrazumevane vrednosti
    return f"{fen_placement} {active_color} {castling} {ep} {halfmove} {fullmove}"


def visualize(warped, board, pipe, fen_placement, out_path=None, show=False):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    cell = pipe.cell
    for k in range(9):
        ax.axhline(pipe.headroom + k * cell, color="yellow", lw=1, alpha=0.5)
        ax.axvline(k * cell, color="yellow", lw=1, alpha=0.5)

    for r in range(8):
        for c in range(8):
            ch = board[r][c]
            if ch == ".":
                continue
            ax.text(c * cell + cell / 2, pipe.headroom + r * cell + cell / 2, ch,
                    color="white" if ch.isupper() else "black",
                    fontsize=12, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle", facecolor="gray", alpha=0.4))

    ax.set_title(fen_placement, fontsize=9)
    ax.axis("off")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"vizuelizacija sacuvana u {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="putanja do slike sahovske table")
    ap.add_argument("--models", default="../models")
    ap.add_argument("--corner-mode", choices=["regression", "heatmap"], default="regression")
    ap.add_argument("--active-color", default="w", choices=["w", "b"])
    ap.add_argument("--castling", default="KQkq")
    ap.add_argument("--ep", default="-")
    ap.add_argument("--halfmove", type=int, default=0)
    ap.add_argument("--fullmove", type=int, default=1)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-vis", default=None)
    args = ap.parse_args()

    fen_placement, board, corners, warped, pipe = predict(
        args.image, models_dir=args.models, corner_mode=args.corner_mode)

    fen = full_fen(fen_placement, args.active_color, args.castling,
                    args.ep, args.halfmove, args.fullmove)

    print("\nTabla:")
    for row in board:
        print(" ".join(row))
    print(f"\nFEN: {fen}")

    if args.show or args.save_vis:
        visualize(warped, board, pipe, fen_placement, out_path=args.save_vis, show=args.show)


if __name__ == "__main__":
    main()
