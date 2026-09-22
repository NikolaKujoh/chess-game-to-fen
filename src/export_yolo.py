"""
Priprema YOLO dataset iz parsed_dataset.json - detekcija figura direktno na
originalnoj fotografiji (bez warp-a), kao alternativa occupancy+piece CNN
paru. Svaka figura vec ima "box" (x, y, w, h u pikselima originalne slike)
iz parse_json.py, samo se normalizuje u YOLO format i upisuje label fajl.
Klase su iste kao PIECE_CLASSES u pipeline.py (isti redosled/indeksi).

Korišćenje:
    python export_yolo.py --dataroot "<dataset>" --out ../data/yolo
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from pipeline import PIECE_CLASSES

NAME_TO_IDX = {name: i for i, name in enumerate(PIECE_CLASSES)}


def write_label(rec, dataroot, out_root, split):
    img_path = dataroot / rec["path"]
    if not img_path.exists():
        return False

    with Image.open(img_path) as im:
        W, H = im.size

    dst_img = out_root / "images" / split / f"{rec['id']}.png"
    if not dst_img.exists():
        try:
            dst_img.symlink_to(img_path.resolve())
        except OSError:
            shutil.copy(img_path, dst_img)

    lines = []
    for p in rec["pieces"]:
        idx = NAME_TO_IDX.get(p["name"])
        if idx is None:
            continue
        x, y, w, h = p["box"]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        nw, nh = (x2 - x1) / W, (y2 - y1) / H
        lines.append(f"{idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    (out_root / "labels" / split / f"{rec['id']}.txt").write_text("\n".join(lines))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--records", default="parsed_dataset.json")
    ap.add_argument("--out", default="../data/yolo")
    args = ap.parse_args()

    dataroot = Path(args.dataroot)
    out_root = Path(args.out)
    with open(args.records) as f:
        records = json.load(f)

    counts = {"train": 0, "val": 0, "test": 0}
    for split in counts:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for rec in records:
        split = rec.get("split")
        if split not in counts:
            continue
        if write_label(rec, dataroot, out_root, split):
            counts[split] += 1

    yaml_lines = [f"path: {out_root.resolve()}", "train: images/train",
                  "val: images/val", "test: images/test", "names:"]
    yaml_lines += [f"  {i}: {name}" for i, name in enumerate(PIECE_CLASSES)]
    (out_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    print("Images per split:", counts)
    print(f"data.yaml -> {(out_root / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
