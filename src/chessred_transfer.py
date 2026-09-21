"""
Test sintetika -> stvarnost: da li occupancy/piece mreže rade i na pravim slikama.

Naši occupancy i piece CNN su trenirani samo na renderovanim (sintetičkim)
slikama iz chesscog dataseta. Ovde ih testiramo na pravim fotografijama iz
ChessReD dataseta, uzimajući ChessReD ground-truth uglove direktno (da bi
se izolovala greška klasifikatora od eventualne greške u detekciji
uglova - corner mreža se ovde uopšte ne koristi).

Za svaku ChessReD sliku: warp preko ground-truth uglova -> isečenih 64
polja -> occupancy CNN -> piece CNN -> uporedi sa pravom pločom iz
ChessReD anotacija. Očekivano je da rezultat bude dosta lošiji nego na
sintetičkom test skupu - ta razlika (domain gap) je i poenta ovog testa.

Treba ChessReD folder (annotations.json + images/) i istrenirani modeli u
../models.

Korišćenje:
    python chessred_transfer.py --chessred "<putanja do ChessReD>"
    python chessred_transfer.py --chessred "<...>" --limit 200   (brzi test)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from pipeline import Pipeline


CAT_TO_FEN = {
    0: "P", 1: "R", 2: "N", 3: "B", 4: "Q", 5: "K",
    6: "p", 7: "r", 8: "n", 9: "b", 10: "q", 11: "k",
}
ROWS = "87654321"
COLS = "abcdefgh"


def build_chessred_records(chessred_root: Path):
    with open(chessred_root / "annotations.json") as f:
        raw = json.load(f)

    images = {im["id"]: im for im in raw["images"]}
    corners = {c["image_id"]: c["corners"] for c in raw["annotations"]["corners"]}

    pieces_by_img = {}
    for p in raw["annotations"]["pieces"]:
        pieces_by_img.setdefault(p["image_id"], []).append(p)

    records = []
    for img_id, corner_dict in corners.items():
        meta = images[img_id]
        board = [["." for _ in range(8)] for _ in range(8)]
        for p in pieces_by_img.get(img_id, []):
            cat = p["category_id"]
            if cat == 12:
                continue
            pos = p["chessboard_position"]
            r = ROWS.index(pos[1]); c = COLS.index(pos[0])
            board[r][c] = CAT_TO_FEN[cat]
        records.append({
            "id": img_id,
            "path": meta["path"],
            "corners": corner_dict,
            "board": board,
        })
    return records


def named_to_order(corner_dict):
    return np.array([
        corner_dict["bottom_left"],
        corner_dict["top_left"],
        corner_dict["top_right"],
        corner_dict["bottom_right"],
    ], dtype=np.float32)


def board_correct_count(pred, true):
    return sum(1 for r in range(8) for c in range(8) if pred[r][c] == true[r][c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chessred", required=True,
                    help="ChessReD root (annotations.json + images/)")
    ap.add_argument("--models", default="../models")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N images (quick test)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    chessred_root = Path(args.chessred)
    records = build_chessred_records(chessred_root)
    print(f"ChessReD annotated images: {len(records)}")
    if args.limit:
        records = records[:args.limit]
        print(f"Limiting to {len(records)}")

    pipe = Pipeline(args.models, device, corner_mode="regression")

    sq_correct = sq_total = 0
    board_correct = 0
    missing = 0

    for n, rec in enumerate(records):
        img_path = chessred_root / rec["path"]
        img = cv2.imread(str(img_path))
        if img is None:
            missing += 1
            continue
        corners = named_to_order(rec["corners"])
        pred_board = pipe.board_from_image(img, corners)
        c = board_correct_count(pred_board, rec["board"])
        sq_correct += c; sq_total += 64
        if c == 64:
            board_correct += 1
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(records)} ... running per-square "
                  f"{sq_correct/sq_total*100:.1f}%")

    n_eval = sq_total // 64
    print("\n=== SYNTHETIC->REAL TRANSFER (ChessReD, ground-truth corners) ===")
    print(f"  images evaluated   : {n_eval}  (missing files: {missing})")
    print(f"  per-square accuracy: {sq_correct/sq_total*100:.2f}%")
    print(f"  full-board accuracy: {board_correct/n_eval*100:.2f}% "
          f"({board_correct}/{n_eval})")
    print("\nCompare to in-domain chesscog (true corners): "
          "96.96% per-square / 18.28% full-board.")


if __name__ == "__main__":
    main()
