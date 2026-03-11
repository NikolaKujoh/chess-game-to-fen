# chess-game-to-fen

Deep learning pipeline to extract FEN notation from chess board images using CNNs and computer vision.

## Piece classifier (minimal training pipeline)

This repo includes a simple, clone-friendly piece-classifier module based on pre-cropped piece folders.

### Expected dataset layout

```text
/path/to/piece_crops/
  train/<class_name>/*.png
  val/<class_name>/*.png
  test/<class_name>/*.png
```

### Train

```bash
python -m src.piece_classifier.train \
  --data-root /path/to/piece_crops \
  --epochs 5 \
  --batch-size 32
```

Training artifacts are saved to `artifacts/piece_classifier/` by default:
- `best_model.pth`
- `last_model.pth`
- `history.json`
- `class_to_idx.json`

### Notes

- Use `src/prepare_piece_crops.py` to export crops from raw ChessReD annotations.
- `Neuralne_projekat.ipynb` can still be used for interactive experimentation/visualization.
