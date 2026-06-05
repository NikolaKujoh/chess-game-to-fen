"""
Extract structured ChessReD2K data from annotations.json.

For every annotated image (the ~2078 with corner + bbox annotations) this builds
a clean record containing:
  - image metadata (id, path, size)
  - the 4 board corners (top_left, top_right, bottom_right, bottom_left)
  - an 8x8 board grid of piece labels, derived from chessboard_position
  - the raw piece list (position, category_id, category_name, bbox)
  - a FEN piece-placement string (handy for sanity checks)

Everything is written to a single JSON file you can load in later stages.

Board orientation (white's perspective, matching the dataset convention):
    rows top->bottom = ranks 8..1   ("87654321")
    cols left->right = files a..h    ("abcdefgh")
    => board[0][0] is a8, board[7][7] is h1

Usage:
    python parse_chessred2k.py --dataroot "path/to/folder_with_annotations.json"
    python parse_chessred2k.py --dataroot . --out chessred2k_parsed.json
"""

import argparse
import json
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────

CATEGORIES = {
    0: "white-pawn",   1: "white-rook",   2: "white-knight",
    3: "white-bishop", 4: "white-queen",  5: "white-king",
    6: "black-pawn",   7: "black-rook",   8: "black-knight",
    9: "black-bishop", 10: "black-queen", 11: "black-king",
    12: "empty",
}

# FEN letters for each category (empty handled separately as a run-length digit)
PIECE_TO_FEN = {
    0: "P", 1: "R", 2: "N", 3: "B", 4: "Q", 5: "K",
    6: "p", 7: "r", 8: "n", 9: "b", 10: "q", 11: "k",
}

EMPTY = 12

ROWS = "87654321"   # rank for each board row, top to bottom
COLS = "abcdefgh"   # file for each board column, left to right

CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def position_to_rc(pos: str):
    """Convert algebraic position like 'e2' to (row, col) on the 8x8 grid."""
    file_char = pos[0]   # 'a'..'h'
    rank_char = pos[1]   # '1'..'8'
    row = ROWS.index(rank_char)
    col = COLS.index(file_char)
    return row, col


def board_to_fen(board):
    """Convert an 8x8 grid of category ids to a FEN piece-placement string."""
    fen_rows = []
    for row in board:
        fen_row = ""
        empty_run = 0
        for cat in row:
            if cat == EMPTY:
                empty_run += 1
            else:
                if empty_run:
                    fen_row += str(empty_run)
                    empty_run = 0
                fen_row += PIECE_TO_FEN[cat]
        if empty_run:
            fen_row += str(empty_run)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)


# ── Main parsing ───────────────────────────────────────────────────────────────

def parse(dataroot: Path, use_chessred2k_split: bool = True):
    ann_path = dataroot / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"annotations.json not found at {ann_path}")

    with open(ann_path, "r") as f:
        raw = json.load(f)

    # image_id -> metadata
    images = {img["id"]: img for img in raw["images"]}

    # image_id -> corners dict
    corners_by_id = {c["image_id"]: c["corners"] for c in raw["annotations"]["corners"]}

    # image_id -> list of piece annotations
    pieces_by_id = {}
    for p in raw["annotations"]["pieces"]:
        pieces_by_id.setdefault(p["image_id"], []).append(p)

    annotated_ids = set(corners_by_id.keys())

    # Which split each image belongs to (within chessred2k)
    split_of = {}
    if use_chessred2k_split and "chessred2k" in raw["splits"]:
        for split in ("train", "val", "test"):
            for iid in raw["splits"]["chessred2k"][split]["image_ids"]:
                split_of[iid] = split

    records = []
    skipped = 0

    for img_id in sorted(annotated_ids):
        img_meta = images[img_id]

        # Corners, in a stable order
        corner_dict = corners_by_id[img_id]
        corners = {name: corner_dict[name] for name in CORNER_ORDER}

        # Build 8x8 board (default empty) + raw piece list
        board = [[EMPTY for _ in range(8)] for _ in range(8)]
        piece_list = []

        for p in pieces_by_id.get(img_id, []):
            cat = p["category_id"]
            if cat == EMPTY:
                continue
            pos = p["chessboard_position"]
            row, col = position_to_rc(pos)
            board[row][col] = cat
            piece_list.append({
                "position": pos,
                "row": row,
                "col": col,
                "category_id": cat,
                "category_name": CATEGORIES[cat],
                "bbox": p["bbox"],            # [x, y, w, h]
            })

        record = {
            "image_id": img_id,
            "file_name": img_meta["file_name"],
            "path": img_meta["path"],
            "width": img_meta["width"],
            "height": img_meta["height"],
            "split": split_of.get(img_id, "unknown"),
            "corners": corners,
            "board": board,                   # 8x8 grid of category ids
            "fen": board_to_fen(board),
            "pieces": piece_list,
            "num_pieces": len(piece_list),
        }
        records.append(record)

    return records, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True,
                        help="Folder containing annotations.json and images/")
    parser.add_argument("--out", type=str, default="chessred2k_parsed.json",
                        help="Output JSON path (default: chessred2k_parsed.json)")
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    records, skipped = parse(dataroot)

    # Split counts
    counts = {}
    for r in records:
        counts[r["split"]] = counts.get(r["split"], 0) + 1

    print(f"Parsed {len(records)} annotated images.")
    print("Per split:", counts)

    # Show one example so you can eyeball it
    if records:
        ex = records[0]
        print("\n── Example record ──")
        print(f"  image_id : {ex['image_id']}")
        print(f"  file     : {ex['file_name']}  ({ex['width']}x{ex['height']})")
        print(f"  split    : {ex['split']}")
        print(f"  pieces   : {ex['num_pieces']}")
        print(f"  corners  :")
        for name in CORNER_ORDER:
            x, y = ex["corners"][name]
            print(f"      {name:13s} = ({x:.1f}, {y:.1f})")
        print(f"  FEN      : {ex['fen']}")
        print("  board    :")
        for row in ex["board"]:
            print("     " + " ".join(
                PIECE_TO_FEN.get(c, ".") if c != EMPTY else "." for c in row
            ))

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
