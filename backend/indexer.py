"""
indexer.py
----------
Construit l'index FAISS à partir des photos collectées.

Pipeline par image :
  1. Lecture image → BGR (OpenCV)
  2. InsightFace buffalo_l → détection visage + embedding ArcFace 512-dim
  3. L2-normalisation → vecteur unitaire
  4. Ajout dans FAISS IndexFlatIP
  5. Métadonnée associée dans metadata.json

Pourquoi IndexFlatIP (Inner Product) ?
  - Sur des vecteurs L2-normalisés : produit intérieur = cosine similarity
  - Recherche exacte (pas approximée) — suffisant jusqu'à ~1M vecteurs
  - Au-delà : basculer sur IndexIVFFlat (approximé, beaucoup plus rapide)

Pourquoi InsightFace buffalo_l ?
  - Modèle ArcFace SOTA, entraîné sur MS1MV3 (5M images, 93k identités)
  - Accuracy LFW : 99.77% (meilleur open-source disponible)
  - Clé en main : détection + alignment + embedding en une seule API
"""

import json
import numpy as np
import faiss
import cv2
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from insightface.app import FaceAnalysis


class FaceIndexer:
    """
    Gère la construction et la sauvegarde de l'index FAISS.

    Fichiers produits :
      index_dir/index.faiss    — vecteurs (binaire FAISS)
      index_dir/metadata.json  — [{name, source_image}] indexé pareil que FAISS
    """

    EMBEDDING_DIM = 512  # Dimension fixe du modèle buffalo_l

    def __init__(self, index_dir: str = "./data/index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Index FAISS : produit intérieur exact
        self.index    = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self.metadata = []  # Liste parallèle à l'index FAISS

        # Modèle InsightFace (télécharge buffalo_l automatiquement au premier lancement)
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Détecte le visage principal et retourne son embedding L2-normalisé.

        On prend le visage le plus grand dans l'image (= le sujet principal).
        Retourne None si aucun visage n'est détecté.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        faces = self.face_app.get(img)
        if not faces:
            return None

        # Visage le plus grand = sujet principal
        main_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = main_face.embedding.astype(np.float32)

        # L2-normalisation → norme unitaire pour que IP = cosine similarity
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else None

    def add_person(self, name: str, images_dir: Path) -> int:
        """
        Indexe toutes les images d'une personne.
        Retourne le nombre d'embeddings ajoutés.
        """
        image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        added = 0

        for img_path in image_paths:
            emb = self.get_embedding(img_path)
            if emb is None:
                continue
            self.index.add(emb.reshape(1, -1))
            self.metadata.append({"name": name, "source": str(img_path)})
            added += 1

        return added

    def build_from_directory(self, faces_dir: Path) -> None:
        """
        Construit l'index complet depuis un dossier organisé par personne :
            faces_dir/
            ├── Elon_Musk/
            │   ├── 000.jpg
            │   └── 001.jpg
            └── Barack_Obama/
                └── 000.jpg
        """
        person_dirs = [d for d in sorted(faces_dir.iterdir()) if d.is_dir()]
        print(f"\n→ Indexing {len(person_dirs)} identities from {faces_dir}\n")

        for person_dir in tqdm(person_dirs, desc="Indexing"):
            name  = person_dir.name.replace("_", " ")
            added = self.add_person(name, person_dir)
            tqdm.write(f"  {'✓' if added else '✗'} {name}: {added} embeddings")

        n_identities = len(set(m["name"] for m in self.metadata))
        print(f"\n✅ Index built: {self.index.ntotal} embeddings | {n_identities} identities")
        self.save()

    def save(self) -> None:
        """Sauvegarde l'index FAISS et les métadonnées sur disque."""
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"💾 Saved to {self.index_dir}")

    def load(self) -> None:
        """Charge un index existant depuis le disque."""
        index_path = self.index_dir / "index.faiss"
        meta_path  = self.index_dir / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"No index found at {index_path}. Run build_index.py first.")

        self.index = faiss.read_index(str(index_path))
        with open(meta_path) as f:
            self.metadata = json.load(f)

        n_ids = len(set(m["name"] for m in self.metadata))
        print(f"✅ Index loaded: {self.index.ntotal} embeddings | {n_ids} identities")