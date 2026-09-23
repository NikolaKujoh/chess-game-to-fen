"""
Mali server - telefon uploaduje fotografiju table (preko browsera), PC
pokrene postojeći pipeline (corner heatmap -> occupancy CNN -> piece CNN)
i vrati stranicu sa originalnom fotografijom i dijagramom pozicije (SVG,
iz python-chess biblioteke) jedno pored drugog.

Nije potreban hosting - telefon i PC treba samo da budu na istoj WiFi
mreži. Pokreni na PC-u, pa na telefonu otvori http://<IP-adresa-PC-a>:8000/
(IP adresu PC-a nađeš sa `ipconfig` na Windows-u, polje "IPv4 Address").

Korišćenje:
    cd src
    python server.py --models ../models --corner-mode heatmap
"""

import argparse
import base64

import chess
import chess.svg
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from pipeline import Pipeline, board_to_fen

app = FastAPI()
pipe: Pipeline = None  # postavljeno u main()

INDEX_PAGE = """<!doctype html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family:sans-serif;text-align:center;padding:16px">
<h2>Sahovska pozicija iz fotografije</h2>
<form action="/predict" method="post" enctype="multipart/form-data">
  <input type="file" name="file" accept="image/*" capture="environment" required
         style="font-size:1.1em">
  <br><br>
  <button type="submit" style="font-size:1.2em;padding:8px 24px">Prepoznaj</button>
</form>
</body></html>"""

RESULT_PAGE = """<!doctype html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family:sans-serif;text-align:center;padding:16px">
<h3 style="word-break:break-all">FEN: {fen}</h3>
<div style="display:flex;justify-content:center;gap:24px;flex-wrap:wrap">
  <img src="data:image/jpeg;base64,{photo_b64}" style="max-width:45%;height:auto">
  <div>{svg}</div>
</div>
<p><a href="/">&larr; nova slika</a></p>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_PAGE


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    corners = pipe.predict_corners(img)
    board = pipe.board_from_image(img, corners)
    fen_placement = board_to_fen(board)

    svg = chess.svg.board(chess.Board(f"{fen_placement} w - - 0 1"), size=380)
    photo_b64 = base64.b64encode(data).decode()

    return RESULT_PAGE.format(fen=fen_placement, photo_b64=photo_b64, svg=svg)


def main():
    global pipe
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="../models")
    ap.add_argument("--corner-mode", choices=["regression", "heatmap"], default="heatmap")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Uredjaj: {device}")
    pipe = Pipeline(args.models, device, corner_mode=args.corner_mode)
    print(f"Server na http://{args.host}:{args.port}  (na telefonu otvori http://<IP-PC-a>:{args.port}/)")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
