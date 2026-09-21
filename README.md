# chess-game-to-fen

Deep learning pipeline to extract FEN notation from chess board images.

Nikola Nenadović

## Approaches in this repo

- **`src/` (base pipeline)** — a 4-stage CNN pipeline: a heatmap-based corner
  detector finds the 4 board corners, a homography warps the board into 64
  cells, an occupancy CNN flags empty/occupied cells, and a piece CNN
  classifies the occupied ones (12 classes: 6 piece types x 2 colors). All
  three models are already trained and included in `models/`.
- **`YOLO_chess.ipynb`** — a YOLO-based upgrade of the same problem, built on
  top of the pipeline above.

## Installation

```bash
pip install -r requirements.txt
```

## Quick test on your own image

Models are already trained and live in `models/`.

```bash
cd src
python predict_fen.py "path/to/image.jpg" --corner-mode heatmap --show
```

Prints the FEN for the position and shows a visualization of the prediction.

## Reproducing results (on the chesscog test set)

```bash
cd src
python parse_json.py --dataroot "<path to chesscog dataset>" --out parsed_dataset.json
python pipeline.py --dataroot "<path to chesscog dataset>" --eval --corner-mode heatmap
```

## Training from scratch (optional)

```bash
cd src
python parse_json.py --dataroot "<dataset>" --out parsed_dataset.json
python export_cells.py --dataroot "<dataset>" --out ../data/processed
python occupancy_cnn.py --data ../data/processed/occupancy
python piece_cnn.py --data ../data/processed/pieces
python corner_heatmap.py --dataroot "<dataset>"
python pipeline.py --dataroot "<dataset>" --eval --corner-mode heatmap
```

## File structure

- `src/` — pipeline code (each script documents itself at the top)
- `models/` — trained models (`.pth`) and the piece-classifier confusion matrix
- `YOLO_chess.ipynb` — YOLO-based upgrade built on this pipeline
