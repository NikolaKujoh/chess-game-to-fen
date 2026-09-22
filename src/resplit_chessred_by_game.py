"""
Deli ChessReD dataset na train/val/test PO PARTIJI (game_id), ne po
pojedinačnoj slici, i direktno generiše YOLO detekcione labele za figure.

ZAŠTO: ChessReD ima samo 20 partija sa ~104 slike po partiji (uzastopni
potezi iste partije, isti sto, ista kamera). Split po pojedinačnoj slici
curi podatke - model vidi "brata" test slike (istu scenu, par poteza
ranije) tokom treninga, što daje lažno visoke rezultate. Ispravan split
mora da drži SVE slike jedne partije u istom splitu.

Podržava i K-fold (--n-folds/--fold) da bi se izmerio RASPON rezultata
preko različitih test partija, što je važno kad je broj partija mali
(N=20) - jedan nasumičan split može "pogoditi" lakše ili teže partije.

Izlazna struktura (kompatibilna sa merge_yolo_datasets.py):
    <out>/images/train/*.jpg (ili .png)
    <out>/images/val/*.jpg
    <out>/images/test/*.jpg
    <out>/labels/train/*.txt
    <out>/labels/val/*.txt
    <out>/labels/test/*.txt

Korišćenje (jedan split, 70/15/15 po broju partija):
    python resplit_chessred_by_game.py \
        --annotations /content/annotations.json \
        --dataroot /content/chessred \
        --out /content/yolo_data/chessred_v2 \
        --seed 42

Korišćenje (K-fold, za merenje raspona rezultata - ponoviti za fold 0..4):
    python resplit_chessred_by_game.py \
        --annotations /content/annotations.json \
        --dataroot /content/chessred \
        --out /content/yolo_data/chessred_fold0 \
        --n-folds 5 --fold 0 --seed 42
"""

import argparse
import json
import random
import shutil
from pathlib import Path

# ChessReD category_id -> (class_index, FEN slovo)
# category_id se poklapa 1:1 sa redosledom klasa u kombinovanom data.yaml:
# ['white-pawn','white-rook','white-knight','white-bishop','white-queen','white-king',
#  'black-pawn','black-rook','black-knight','black-bishop','black-queen','black-king']
CAT_TO_CLASS_IDX = {i: i for i in range(12)}  # category_id 0-11 -> class index 0-11 (1:1)
OCCLUDED_CAT = 12  # kategorija za figure van vidnog polja - preskače se


def get_bbox_xywh(piece, img_w, img_h):
    """Vraća (x, y, w, h) u apsolutnim pikselima iz piece anotacije.
    Proba nekoliko mogućih ključeva jer format nije bio potvrđen unapred."""
    if "bbox" in piece:
        x, y, w, h = piece["bbox"]
        return float(x), float(y), float(w), float(h)
    if "box" in piece:
        x, y, w, h = piece["box"]
        return float(x), float(y), float(w), float(h)
    raise KeyError(
        f"Ne mogu da nađem bbox u piece zapisu. Dostupni ključevi: {list(piece.keys())}. "
        f"Prilagodi get_bbox_xywh() ovoj strukturi."
    )


def split_games_single(game_ids, ratio_str, seed):
    games = sorted(game_ids)
    random.Random(seed).shuffle(games)

    parts = [float(x) for x in ratio_str.split("/")]
    total = sum(parts)
    n = len(games)
    n_train = round(n * parts[0] / total)
    n_val = round(n * parts[1] / total)
    # ostatak ide u test, da se svih N partija sigurno rasporedi
    n_test = n - n_train - n_val

    train_games = set(games[:n_train])
    val_games = set(games[n_train:n_train + n_val])
    test_games = set(games[n_train + n_val:])

    print(f"Split po partijama (seed={seed}): "
          f"train={len(train_games)} val={len(val_games)} test={len(test_games)}")
    print(f"  train partije: {sorted(train_games)}")
    print(f"  val partije  : {sorted(val_games)}")
    print(f"  test partije : {sorted(test_games)}")
    return train_games, val_games, test_games


def split_games_kfold(game_ids, n_folds, fold, seed):
    games = sorted(game_ids)
    random.Random(seed).shuffle(games)
    n = len(games)

    # deli na n_folds skoro jednakih delova
    folds = [[] for _ in range(n_folds)]
    for i, g in enumerate(games):
        folds[i % n_folds].append(g)

    test_games = set(folds[fold])
    val_fold_idx = (fold + 1) % n_folds
    val_games = set(folds[val_fold_idx])
    train_games = set(games) - test_games - val_games

    print(f"K-fold split (n_folds={n_folds}, fold={fold}, seed={seed}): "
          f"train={len(train_games)} val={len(val_games)} test={len(test_games)}")
    print(f"  test partije (fold {fold}): {sorted(test_games)}")
    print(f"  val partije (fold {val_fold_idx}): {sorted(val_games)}")
    return train_games, val_games, test_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--dataroot", required=True,
                    help="folder sa slikama (sadrži <game_id>/<fajl> podfoldere)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", default="70/15/15",
                    help="train/val/test odnos PO BROJU PARTIJA (koristi se ako --n-folds nije zadat)")
    ap.add_argument("--n-folds", type=int, default=None,
                    help="ako je zadato, radi K-fold split po partijama umesto --ratio")
    ap.add_argument("--fold", type=int, default=0,
                    help="koji fold materijalizovati (0-indeksirano), samo uz --n-folds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true",
                    help="koristi symlink umesto kopiranja slika (brže, štedi disk)")
    args = ap.parse_args()

    dataroot = Path(args.dataroot)
    out_root = Path(args.out)

    with open(args.annotations) as f:
        raw = json.load(f)

    images = {im["id"]: im for im in raw["images"]}
    corners = {c["image_id"] for c in raw["annotations"]["corners"]}  # samo id-jevi

    pieces_by_img = {}
    for p in raw["annotations"]["pieces"]:
        pieces_by_img.setdefault(p["image_id"], []).append(p)

    # samo potpuno anotirane slike (imaju i uglove) - isti skup kao ranije (2078)
    annotated_ids = sorted(corners)
    game_ids = set(images[i]["game_id"] for i in annotated_ids)
    print(f"Ukupno anotiranih slika: {len(annotated_ids)}, partija: {len(game_ids)}")

    if args.n_folds:
        train_games, val_games, test_games = split_games_kfold(
            game_ids, args.n_folds, args.fold, args.seed)
    else:
        train_games, val_games, test_games = split_games_single(
            game_ids, args.ratio, args.seed)

    game_to_split = {}
    for g in train_games:
        game_to_split[g] = "train"
    for g in val_games:
        game_to_split[g] = "val"
    for g in test_games:
        game_to_split[g] = "test"

    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    skipped_no_pieces = 0
    skipped_no_image = 0

    for img_id in annotated_ids:
        meta = images[img_id]
        split = game_to_split[meta["game_id"]]

        img_w, img_h = meta["width"], meta["height"]
        rel_path = meta["path"].replace("\\", "/")
        if rel_path.startswith("images/"):
            rel_path = rel_path[len("images/"):]
        src_img = dataroot / rel_path

        if not src_img.exists():
            skipped_no_image += 1
            continue

        pieces = pieces_by_img.get(img_id, [])
        if not pieces:
            skipped_no_pieces += 1
            # ipak nastavljamo - prazna tabla je validan slučaj, samo bez YOLO labela

        # jedinstveno ime fajla (game_id + image_id), da izbegnemo kolizije
        stem = f"g{meta['game_id']}_{img_id}_{Path(meta['file_name']).stem}"
        ext = Path(meta["file_name"]).suffix
        dst_img = out_root / "images" / split / f"{stem}{ext}"
        dst_lbl = out_root / "labels" / split / f"{stem}.txt"

        if args.link:
            if not dst_img.exists():
                dst_img.symlink_to(src_img.resolve())
        else:
            shutil.copy2(src_img, dst_img)

        lines = []
        for p in pieces:
            cat = p["category_id"]
            if cat == OCCLUDED_CAT:
                continue
            cls_idx = CAT_TO_CLASS_IDX[cat]
            x, y, w, h = get_bbox_xywh(p, img_w, img_h)

            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)

            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        dst_lbl.write_text("\n".join(lines))
        counts[split] += 1

    print(f"\nGotovo. Slika po splitu: {counts}")
    print(f"Preskočeno (nema fajl slike): {skipped_no_image}")
    print(f"Slika bez figura (prazna tabla, i dalje uključene): {skipped_no_pieces}")
    print(f"Izlaz: {out_root.resolve()}")


if __name__ == "__main__":
    main()
