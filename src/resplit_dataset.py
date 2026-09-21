"""
Ponovo deli dataset na train/val/test.

Pokupi sve png+json parove (iz train/val/test foldera ili direktno iz
dataroot-a), izmeša ih i podeli u nove foldere po zadatom odnosu
(podrazumevano 60/20/20). Fajlovi se fizički premeštaju, ne kopiraju.
Korisno kad postojeći split ne odgovara ili kad se doda još slika.

Korišćenje:
    python resplit_dataset.py --dataroot "<dataset>" --out "<novi dataset>"
"""

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ratio", default="60/20/20")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    out_root = Path(args.out)

    # prikupljanje parova slika i json-a
    pairs = []
    search_dirs = [dataroot / s for s in ["train", "val", "test"] if (dataroot / s).is_dir()]
    if not search_dirs:
        search_dirs = [dataroot]

    for d in search_dirs:
        for json_file in d.glob("*.json"):
            png_file = json_file.with_suffix(".png")
            if png_file.exists():
                pairs.append((png_file, json_file))

    print(f"Pronađeno ukupno {len(pairs)} parova.")

    if not pairs:
        print("Nema fajlova za split.")
        return

    random.seed(args.seed)
    random.shuffle(pairs)

    parts = [float(x) for x in args.ratio.split("/")]
    total = sum(parts)
    tr, va = parts[0] / total, parts[1] / total

    n = len(pairs)
    n_train = int(n * tr)
    n_val = int(n * va)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:]
    }

    for split, split_pairs in splits.items():
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for png, js in split_pairs:
            shutil.move(str(png), str(split_dir / png.name))
            shutil.move(str(js), str(split_dir / js.name))
            
        print(f"{split}: {len(split_pairs)} parova premešteno")

    print(f"Završeno splitovanje u {out_root}")


if __name__ == "__main__":
    main()