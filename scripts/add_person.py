"""
add_person.py
-------------
Ajoute une personne à l'index FAISS.

Usage :
  python scripts/add_person.py --name "Elon Musk" --max 25
  python scripts/add_person.py --name "Barack Obama" --images_dir ./mes_photos/
"""

import argparse
import sys
from pathlib import Path

# Ajouter la racine du projet au path Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.scraper import collect_person_images
from backend.indexer import FaceIndexer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",       type=str, required=True)
    parser.add_argument("--index_dir",  type=str, default="./data/index")
    parser.add_argument("--faces_dir",  type=str, default="./data/faces")
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--max",        type=int, default=25)
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    faces_dir = Path(args.faces_dir)
    faces_dir.mkdir(parents=True, exist_ok=True)

    # --- Étape 1 : récupérer les images ---
    if args.images_dir:
        images_dir = Path(args.images_dir)
        print(f"Using existing images from {images_dir}")
    else:
        print(f"\n📥 Downloading images for '{args.name}'...")
        n_downloaded = collect_person_images(args.name, faces_dir, max_total=args.max)
        print(f"✅ {n_downloaded} images downloaded")
        images_dir = faces_dir / args.name.replace(" ", "_")

    if not images_dir.exists() or not list(images_dir.glob("*.jpg")):
        print(f"❌ No images found in {images_dir}")
        print("   → Télécharge manuellement des photos dans ce dossier")
        sys.exit(1)

    n_imgs = len(list(images_dir.glob("*.jpg")))
    print(f"📂 Found {n_imgs} images in {images_dir}")

    # --- Étape 2 : indexer ---
    print(f"\n🔍 Extracting face embeddings...")
    indexer = FaceIndexer(index_dir=str(index_dir))

    # Charger l'index existant si présent
    if (index_dir / "index.faiss").exists():
        indexer.load()
        print(f"   Loaded existing index ({indexer.index.ntotal} embeddings)")

    added = indexer.add_person(args.name, images_dir)
    print(f"✅ Added {added} embeddings for '{args.name}'")

    if added == 0:
        print("⚠️  0 embeddings — aucun visage détecté dans les images")
        print("   → Vérifie que les photos contiennent bien des visages nets")

    indexer.save()
    print(f"\n💾 Index saved — total: {indexer.index.ntotal} embeddings")


if __name__ == "__main__":
    main()