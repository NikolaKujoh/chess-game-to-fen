"""
Spaja sirov dataset u jedan json fajl.

U dataset/train, dataset/val, dataset/test svaka slika ima svoj .json sa
figurama, uglovima i FEN-om. Ovaj skript prođe kroz sve njih i napravi
jedan parsed_dataset.json sa svim zapisima (dodaje i 8x8 board matricu
izvučenu iz FEN-a), da ne bi ostali skriptovi morali da čitaju stotine
pojedinačnih json fajlova.

Korišćenje:
    python parse_json.py --dataroot "<dataset>" --out parsed_dataset.json
"""

import argparse
import json
from pathlib import Path

FEN_TO_NAME = {
    "P": "white-pawn", "R": "white-rook", "N": "white-knight",
    "B": "white-bishop", "Q": "white-queen", "K": "white-king",
    "p": "black-pawn", "r": "black-rook", "n": "black-knight",
    "b": "black-bishop", "q": "black-queen", "k": "black-king",
}


def fen_to_board(fen):
    board = [["." for _ in range(8)] for _ in range(8)]
    ranks = fen.split()[0].split("/")
    
    for r, rank in enumerate(ranks):
        c = 0
        for ch in rank:
            if ch.isdigit():
                c += int(ch)
            else:
                board[r][c] = ch
                c += 1
    return board


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--out", default="parsed_dataset.json")
    args = parser.parse_args()

    dataset_root = Path(args.dataroot)
    all_records = []

    for split in ["train", "val", "test"]:
        split_path = dataset_root / split
        if not split_path.exists():
            continue
            
        json_files = sorted(split_path.glob("*.json"))
        
        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)

            pieces = []
            for p in data.get("pieces", []):
                sq = p["square"]
                col = ord(sq[0]) - ord('a')
                row = 8 - int(sq[1])
                
                pieces.append({
                    "piece": p["piece"],
                    "name": FEN_TO_NAME.get(p["piece"], ""),
                    "square": sq,
                    "row": row,
                    "col": col,
                    "box": p["box"]
                })

            img_path = jf.with_suffix(".png")
            
            all_records.append({
                "id": jf.stem,
                "split": split,
                "path": str(img_path.relative_to(dataset_root)),
                "fen": data["fen"],
                "white_turn": data.get("white_turn"),
                "camera_angle": data.get("camera", {}).get("angle"),
                "corners": data.get("corners"),
                "board": fen_to_board(data["fen"]),
                "pieces": pieces,
                "num_pieces": len(pieces)
            })

        print(f"Učitano {len(json_files)} iz {split}")

    with open(args.out, "w") as f:
        json.dump(all_records, f, indent=2)
        
    print(f"Ukupno sačuvano {len(all_records)} u {args.out}")


if __name__ == "__main__":
    main()