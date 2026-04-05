# FaceTrace 🎯
> Système de reconnaissance faciale temps réel avec deep learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-red)

---

## 📖 Description

**FaceTrace** est un système de reconnaissance faciale en temps réel qui utilise :
- **InsightFace (ArcFace)** pour l'extraction d'embeddings faciaux 512-dim
- **FAISS** pour la recherche vectorielle ultra-rapide
- **FastAPI + WebSocket** pour le streaming vidéo temps réel

Le système détecte les visages via webcam et les compare ensuite à une base de données locale d'identités pré-enregistrées.

---

## 🛠️ Technologies utilisées

**Backend :**
- Python 3.10+
- FastAPI (API REST + WebSocket)
- InsightFace (modèle ArcFace buffalo_l)
- FAISS (recherche vectorielle)
- OpenCV (traitement d'images)

**Frontend :**
- HTML5 + Vanilla JavaScript
- WebSocket pour communication temps réel
- Canvas API pour overlay des détections

**Machine Learning :**
- Metric learning (ArcFace loss)
- Embeddings 512-dimensions normalisés L2
- Recherche k-NN par cosine similarity

---


## 🚀 Installation
```bash
# Cloner le repository
git clone https://github.com/Rayanmlk/facetrace.git
cd facetrace

# Créer environnement virtuel
python -m venv venv

# Activer l'environnement
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

---

## 📂 Structure du projet
```
facetrace/
├── backend/
│   ├── indexer.py       # Construction de l'index FAISS
│   ├── recognizer.py    # Pipeline de reconnaissance
│   └── main.py          # Serveur FastAPI + WebSocket
├── frontend/
│   └── index.html       # Interface web
├── scripts/
│   ├── build_index.py   # Script CLI pour construire l'index
│   └── add_person.py    # Ajouter une personne à la base
├── data/
│   ├── faces/           # Photos organisées par personne
│   └── index/           # Index FAISS + métadonnées
├── requirements.txt
└── README.md
```

---

## 🎬 Utilisation

### 1. Ajouter des personnes à la base

Créez un dossier pour chaque personne dans `data/faces/` et ajoutez 10-20 photos :
```
data/faces/
├── Alice/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
└── Bob/
    ├── 001.jpg
    └── ...
```

### 2. Construire l'index FAISS
```bash
python scripts/build_index.py --faces_dir ./data/faces
```

### 3. Lancer le serveur
```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. Ouvrir l'interface web

Ouvrez votre navigateur sur : **http://localhost:8000**

Cliquez sur "Start Recognition" et autorisez l'accès à la webcam.

---

## 🧠 Comment ça fonctionne ?

### Architecture
```
Webcam → Détection visage → Extraction embedding → FAISS search → Affichage résultat
```

### Pipeline détaillé

1. **Capture webcam** : Frame envoyé au serveur via WebSocket
2. **Détection** : InsightFace détecte les visages (RetinaFace)
3. **Embedding** : Extraction d'un vecteur 512-dim normalisé (ArcFace)
4. **Recherche** : FAISS trouve les 3 identités les plus proches
5. **Décision** : Si score > 0.25 → identité reconnue, sinon "Unknown"
6. **Affichage** : Bounding box + nom overlay sur la vidéo

### Pourquoi ces choix techniques ?

**InsightFace (ArcFace) :**
- Précision : 99.77% sur le benchmark LFW
- Metric learning : permet d'ajouter des identités sans réentraînement
- Open-source et bien documenté

**FAISS :**
- Recherche ultra-rapide : ~2ms pour 10 000 identités
- Développé par Meta Research
- Standard industriel pour la recherche vectorielle

**FastAPI + WebSocket :**
- Latence < 100ms end-to-end
- Connexion persistante (pas de overhead HTTP)
- Documentation auto-générée

---

## 📊 Performances

| Métrique | Valeur |
|----------|--------|
| Extraction embedding | ~40ms |
| Recherche FAISS (10k faces) | ~2ms |
| Latence end-to-end | ~80ms |
| Précision modèle (LFW) | 99.77% |

---

## 🎓 Contexte académique

**Objectif :** Mettre en pratique les concepts de :
- Deep learning (réseaux de neurones convolutifs)
- Metric learning (ArcFace loss)
- Recherche vectorielle (FAISS)
- APIs temps réel (WebSocket)

---

## 📝 License

MIT License

---

## 👨‍💻 Auteur

**Rayan** 
📧 rayan.malki@edu.ece.fr  

---


- [InsightFace](https://github.com/deepinsight/insightface) pour le modèle ArcFace
- [FAISS](https://github.com/facebookresearch/faiss) pour la recherche vectorielle
```
