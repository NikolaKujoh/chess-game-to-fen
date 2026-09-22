"""
Spaja vise YOLO detekcionih dataset-a (svaki vec u images/<split>+labels/<split>
formatu) u jedan, sa jedinstvenim data.yaml (12 klasa, PIECE_CLASSES redosled
iz pipeline.py). Fajlovi se linkuju/kopiraju sa prefiksom izvora u imenu
(chesscog_, chessred_, roboflow_) da ne bi doslo do kolizije imena.

VAZNO - redosled klasa NIJE isti u sva tri izvora, pa se svaki label
preslikava u kanonski PIECE_CLASSES redosled pre kopiranja:
  - chesscog (export_yolo.py)      -> vec pise data.yaml u kanonskom redosledu
  - chessred (resplit_chessred_by_game.py) -> nema data.yaml, koristi se
    poznati fiksni ChessReD redosled (bela, pa crna, P R N B Q K)
  - roboflow (spoljni export)      -> cita se iz njegovog data.yaml, klase
    se poklapaju po imenu (case/razmak/crta-insensitive)

Korišćenje:
    python merge_yolo_datasets.py --chesscog .../chesscog --chessred .../chessred_v2 \
        --roboflow .../Roboflow --out .../combined_v2
"""

import argparse
import shutil
from pathlib import Path

import yaml

from pipeline import PIECE_CLASSES

CANON_IDX = {name: i for i, name in enumerate(PIECE_CLASSES)}

# redosled category_id 0-11 kako ga pise resplit_chessred_by_game.py
CHESSRED_ORDER = ["white-pawn", "white-rook", "white-knight", "white-bishop",
                   "white-queen", "white-king", "black-pawn", "black-rook",
                   "black-knight", "black-bishop", "black-queen", "black-king"]


def norm(name):
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def map_from_names(names):
    mapping = {}
    for i, name in enumerate(names):
        canon = CANON_IDX.get(norm(name))
        if canon is None:
            print(f"  [WARN] klasa '{name}' (idx {i}) nije prepoznata, preskace se")
            continue
        mapping[i] = canon
    return mapping


def class_map(prefix, source_dir):
    if prefix == "chessred":
        return map_from_names(CHESSRED_ORDER)

    yml = source_dir / "data.yaml"
    if not yml.exists():
        return None  # pretpostavka: vec u kanonskom redosledu (npr. chesscog)

    names = yaml.safe_load(yml.read_text()).get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return map_from_names(names)


def copy_split(source_dir, split, prefix, mapping, out_root):
    img_split = source_dir / "images" / split
    lbl_split = source_dir / "labels" / split
    if not img_split.exists():
        return 0

    n = 0
    for img_path in img_split.iterdir():
        dst_img = out_root / "images" / split / f"{prefix}_{img_path.name}"
        dst_lbl = out_root / "labels" / split / f"{prefix}_{img_path.stem}.txt"
        if dst_img.exists():
            continue
        try:
            dst_img.symlink_to(img_path.resolve())
        except OSError:
            shutil.copy(img_path, dst_img)

        lbl_path = lbl_split / (img_path.stem + ".txt")
        lines = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cls = int(parts[0])
                if mapping is not None:
                    if cls not in mapping:
                        continue
                    cls = mapping[cls]
                lines.append(" ".join([str(cls)] + parts[1:]))
        dst_lbl.write_text("\n".join(lines))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chesscog", default=None)
    ap.add_argument("--chessred", default=None)
    ap.add_argument("--roboflow", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_root = Path(args.out)
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    sources = {"chesscog": args.chesscog, "chessred": args.chessred, "roboflow": args.roboflow}
    counts = {}
    for prefix, path in sources.items():
        if not path:
            continue
        source_dir = Path(path)
        mapping = class_map(prefix, source_dir)
        total = sum(copy_split(source_dir, split, prefix, mapping, out_root)
                    for split in ("train", "val", "test"))
        counts[prefix] = total

    yaml_lines = [f"path: {out_root.resolve()}", "train: images/train",
                  "val: images/val", "test: images/test", "names:"]
    yaml_lines += [f"  {i}: {name}" for i, name in enumerate(PIECE_CLASSES)]
    (out_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    print("Slika po izvoru:", counts)
    print(f"data.yaml -> {(out_root / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
