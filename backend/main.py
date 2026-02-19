"""
main.py
-------
Serveur FastAPI — point d'entrée de l'application.

Routes :
  GET  /              → sert frontend/index.html
  GET  /api/stats     → stats de l'index (nb identités, nb embeddings)
  WS   /ws            → WebSocket : reçoit des frames JPEG, retourne du JSON

Le WebSocket est le cœur du système temps réel :
  Client → envoie frame JPEG (bytes) toutes les ~100ms
  Serveur → répond JSON avec les visages détectés + identités

Démarrage :
  uvicorn backend.main:app --reload --port 8000
"""

import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.recognizer import FaceRecognizer

app = FastAPI(title="FaceTrace", description="Real-time face recognition — Academic project")

# Charger le recognizer une seule fois au démarrage (lourd à initialiser)
recognizer: FaceRecognizer = None


@app.on_event("startup")
async def startup():
    global recognizer
    index_dir = "./data/index"
    if Path(index_dir + "/index.faiss").exists():
        recognizer = FaceRecognizer(index_dir=index_dir)
    else:
        print("⚠️  No index found. Run scripts/build_index.py first.")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Sert l'interface web."""
    html_path = Path("frontend/index.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/stats")
async def get_stats():
    """Retourne les stats de l'index courant."""
    if recognizer is None:
        return JSONResponse({"status": "no_index", "embeddings": 0, "identities": 0})
    n_ids = len(set(m["name"] for m in recognizer.metadata))
    return {
        "status":      "ready",
        "embeddings":  recognizer.index.ntotal,
        "identities":  n_ids,
        "threshold":   0.40,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket temps réel.
    Reçoit : bytes (JPEG frame depuis la webcam du navigateur)
    Envoie : JSON (liste de visages détectés avec identités)
    """
    await websocket.accept()

    if recognizer is None:
        await websocket.send_text(json.dumps({"error": "Index not loaded"}))
        await websocket.close()
        return

    try:
        while True:
            # Recevoir le frame (bytes)
            jpeg_bytes = await websocket.receive_bytes()

            # Reconnaissance (tout en mémoire, aucun stockage)
            faces = recognizer.recognize_frame(jpeg_bytes)

            # Retourner le JSON au frontend
            await websocket.send_text(json.dumps({"faces": faces}))

    except WebSocketDisconnect:
        pass  # Connexion fermée proprement par le client