"""
recognizer.py
-------------
Pipeline de reconnaissance en temps réel.

Appelé à chaque frame WebSocket :
  1. Décoder le JPEG reçu en array numpy
  2. InsightFace → détecter tous les visages du frame
  3. Pour chaque visage : embedding → FAISS.search(k=3) → top-3 matches
  4. Appliquer le threshold de confiance
  5. Retourner les résultats en JSON (bounding boxes + noms + scores)

La séparation recognizer / indexer est intentionnelle :
  - L'indexer est lourd (construction) → lancé une seule fois offline
  - Le recognizer est léger (lecture) → instancié au démarrage du serveur

Note vie privée :
  - Aucun frame n'est stocké, tout est traité en mémoire
  - Les embeddings temporaires sont garbage-collectés après chaque frame
"""

import numpy as np
import cv2
import faiss
import json
from pathlib import Path
from typing import List, Dict, Any
from insightface.app import FaceAnalysis


# Seuil de confiance : en dessous, on retourne "Unknown"
# Cosine similarity ∈ [-1, 1] — 0.4 = seuil raisonnable pour buffalo_l
CONFIDENCE_THRESHOLD = 0.25


class FaceRecognizer:
    """
    Moteur de reconnaissance : charge l'index FAISS + InsightFace,
    expose une méthode recognize_frame() appelée en temps réel.
    """

    def __init__(self, index_dir: str = "./data/index"):
        index_dir = Path(index_dir)

        # Charger l'index FAISS
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        with open(index_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        # Modèle InsightFace (même modèle que l'indexer → cohérence des embeddings)
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        n_ids = len(set(m["name"] for m in self.metadata))
        print(f"✅ Recognizer ready — {self.index.ntotal} embeddings | {n_ids} identities")

    def recognize_frame(self, jpeg_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Reconnaît tous les visages dans un frame JPEG.

        Args:
            jpeg_bytes : frame brut reçu du WebSocket (bytes)

        Returns:
            Liste de dicts, un par visage détecté :
            {
              "bbox":       [x1, y1, x2, y2],
              "name":       "Elon Musk" | "Unknown",
              "confidence": 0.82,
              "top3": [{"name": ..., "score": ...}, ...]
            }
        """
        # Décoder JPEG → numpy BGR (traitement en mémoire uniquement)
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return []

        faces = self.face_app.get(frame)
        results = []

        for face in faces:
            # Embedding L2-normalisé
            emb  = face.embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            emb = (emb / norm).reshape(1, -1)

            # Recherche FAISS : top-3 voisins les plus proches
            scores, indices = self.index.search(emb, k=3)

            top3 = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS retourne -1 si pas assez de vecteurs
                    continue
                top3.append({
                    "name":  self.metadata[idx]["name"],
                    "score": round(float(score), 3),
                })

            # Décision : le meilleur match dépasse-t-il le threshold ?
            best = top3[0] if top3 else None
            name       = best["name"]  if best and best["score"] >= CONFIDENCE_THRESHOLD else "Unknown"
            confidence = best["score"] if best else 0.0

            bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
            results.append({
                "bbox":       bbox,
                "name":       name,
                "confidence": round(confidence, 3),
                "top3":       top3,
            })

        return results