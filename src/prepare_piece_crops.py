import json
from pathlib import Path
import pandas as pd
from PIL import Image

CATEGORIES = {
    0: "white-pawn",
    1: "white-rook",
    2: "white-knight",
    3: "white-bishop",
    4: "white-queen",
    5: "white-king",
    6: "black-pawn",
    7: "black-rook",
    8: "black-knight",
    9: "black-bishop",
    10: "black-queen",
    11: "black-king",
}

def export_piece_crops(
    source_dataroot,
    output_root,
    use_chessred2k_split=True,
    image_format="png",
):
    source_dataroot = Path(source_dataroot)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ann_path = source_dataroot / "annotations.json"
    with open(ann_path, "r") as f:
        raw = json.load(f)

    images_df = pd.DataFrame(raw["images"])
    pieces_df = pd.DataFrame(raw["annotations"]["pieces"])
    corners_df = pd.DataFrame(raw["annotations"]["corners"])

    annotated_ids = set(corners_df["image_id"])

    for split in ["train", "val", "test"]:
        if use_chessred2k_split:
            split_ids = set(raw["splits"]["chessred2k"][split]["image_ids"])
        else:
            split_ids = set(raw["splits"][split]["image_ids"])

        valid_ids = annotated_ids & split_ids

        split_pieces = pieces_df[pieces_df["image_id"].isin(valid_ids)].copy()
        split_pieces = split_pieces[split_pieces["category_id"] != 12].copy()
        split_images = images_df[images_df["id"].isin(valid_ids)].set_index("id")

        print(f"\n[{split}] exporting {len(split_pieces)} crops...")

        for idx, row in split_pieces.iterrows():
            img_id = row["image_id"]
            label = int(row["category_id"])
            class_name = CATEGORIES[label]

            img_meta = split_images.loc[img_id]
            img_path = source_dataroot / img_meta["path"]

            image = Image.open(img_path).convert("RGB")

            x, y, w, h = row["bbox"]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(image.width, int(x + w))
            y2 = min(image.height, int(y + h))

            crop = image.crop((x1, y1, x2, y2))

            class_dir = output_root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            piece_id = row["id"] if "id" in row else idx
            out_path = class_dir / f"{img_id}_{piece_id}.{image_format}"
            crop.save(out_path)

    print("\nDone.")

if __name__ == "__main__":
    source_dataroot = r"data/raw/chessred2k"
    output_root = r"data/processed/piece_crops"
    export_piece_crops(source_dataroot, output_root)