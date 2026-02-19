"""
build_index.py
--------------
Script CLI pour construire l'index FAISS.

Deux modes :
  1. --faces_dir : indexer un dossier d'images déjà téléchargées
  2. --names_file : télécharger + indexer une liste de noms

Usage :
  # Mode 1 : depuis un dossier existant (ex: VGGFace2)
  python scripts/build_index.py --faces_dir ./data/faces

  # Mode 2 : depuis une liste de noms (scraping auto)
  python scripts/build_index.py --names_file ./data/celebrities.txt

  # Vider l'index
  python scripts/build_index.py --clear
"""

import argparse
import shutil
from pathlib import Path
from backend.indexer import FaceIndexer
from backend.scraper import collect_person_images


def main():
    parser = argparse.ArgumentParser(description="Build FaceTrace FAISS index")
    parser.add_argument("--faces_dir",  type=str, default="./data/faces")
    parser.add_argument("--index_dir",  type=str, default="./data/index")
    parser.add_argument("--names_file", type=str, default=None,
                        help="Text file with one celebrity name per line")
    parser.add_argument("--max_images", type=int, default=25,
                        help="Max images to download per person (scraping mode)")
    parser.add_argument("--clear",      action="store_true",
                        help="Delete existing index and start fresh")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    faces_dir = Path(args.faces_dir)

    if args.clear:
        if index_dir.exists():
            shutil.rmtree(index_dir)
            print(f"🗑️  Index cleared: {index_dir}")
        return

    # Mode scraping : télécharger depuis une liste de noms
    if args.names_file:
        names = Path(args.names_file).read_text().strip().splitlines()
        names = [n.strip() for n in names if n.strip()]
        print(f"📥 Downloading images for {len(names)} people...")
        for name in names:
            n = collect_person_images(name, faces_dir, max_total=args.max_images)
            print(f"  {'✓' if n else '✗'} {name}: {n} images")

    # Construire l'index
    indexer = FaceIndexer(index_dir=str(index_dir))
    indexer.build_from_directory(faces_dir)


if __name__ == "__main__":
    main()