"""Données et prédictions pour l'application Pokédex.

Ce module ne fait qu'UTILISER les modèles sauvegardés par src/train_tabular.py :
il ne réentraîne rien. Il est indépendant de Streamlit, ce qui permet de le
tester simplement (et de le réutiliser aux niveaux 2 et 3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, STATS, TYPE_FR
from src.preprocessing import load_clean

STAT_MIN, STAT_MAX = 1, 255        # bornes des stats dans les jeux


@dataclass
class TabularModels:
    scaler: object
    kmeans: object
    clusters: dict          # numéro (str) -> nom, description, couleur, profil moyen…
    legendary: object
    type_clf: object
    metadata: dict

    @property
    def threshold(self) -> float:
        return float(self.metadata["legendaire"]["seuil"])


REQUIRED_FILES = ["scaler.joblib", "kmeans.joblib", "clusters.json", "legendary_classifier.joblib",
                  "type_classifier.joblib", "metadata.json"]


def missing_model_files(models_dir: Path = MODELS_DIR) -> list[str]:
    return [f for f in REQUIRED_FILES if not (Path(models_dir) / f).exists()]


def load_tabular_models(models_dir: Path = MODELS_DIR) -> TabularModels:
    """Charge les 6 fichiers produits par `python -m src.train_tabular`."""
    d = Path(models_dir)
    missing = missing_model_files(d)
    if missing:
        raise FileNotFoundError(f"Modèles manquants dans {d} : {', '.join(missing)}. "
                                "Lancez d'abord : python -m src.train_tabular")
    return TabularModels(
        scaler=joblib.load(d / "scaler.joblib"),
        kmeans=joblib.load(d / "kmeans.joblib"),
        clusters=json.loads((d / "clusters.json").read_text(encoding="utf-8")),
        legendary=joblib.load(d / "legendary_classifier.joblib"),
        type_clf=joblib.load(d / "type_classifier.joblib"),
        metadata=json.loads((d / "metadata.json").read_text(encoding="utf-8")),
    )


def load_pokemon() -> pd.DataFrame:
    """Les 801 Pokémon nettoyés, triés par numéro de Pokédex."""
    return load_clean().sort_values("pokedex_number").reset_index(drop=True)


def _as_frame(stats: dict) -> pd.DataFrame:
    missing = [s for s in STATS if s not in stats]
    if missing:
        raise ValueError(f"Stats manquantes : {missing}")
    x = pd.DataFrame([{s: float(stats[s]) for s in STATS}])
    if ((x < STAT_MIN) | (x > STAT_MAX)).any(axis=None):
        raise ValueError(f"Chaque stat doit être comprise entre {STAT_MIN} et {STAT_MAX}.")
    return x


def predict_stats(models: TabularModels, stats: dict) -> dict:
    """Groupe, probabilité d'être légendaire et top-3 des types pour 6 stats."""
    x = _as_frame(stats)
    z = models.scaler.transform(x)
    c = int(models.kmeans.predict(z)[0])
    dist = models.kmeans.transform(z)[0]           # distance à chaque centre

    p_leg = float(models.legendary.predict_proba(x)[0, 1])

    p_type = models.type_clf.predict_proba(x)[0]
    classes = models.type_clf.classes_
    order = np.argsort(p_type)[::-1]
    types = [{"type": classes[i], "type_fr": TYPE_FR.get(classes[i], classes[i]), "proba": float(p_type[i])}
             for i in order]

    return {
        "total": int(x.iloc[0].sum()),
        "cluster_id": c,
        "cluster": models.clusters[str(c)],
        "cluster_distance": float(dist[c]),
        "cluster_second": models.clusters[str(int(np.argsort(dist)[1]))]["name"],
        "p_legendary": p_leg,
        "legendary_probable": p_leg >= models.threshold,
        "threshold": models.threshold,
        "types": types,                  # tous les types, du plus au moins probable
        "top3": types[:3],
    }


def nearest_pokemon(df: pd.DataFrame, models: TabularModels, stats: dict, k: int = 3,
                    exclude_name: str | None = None) -> pd.DataFrame:
    """Les k Pokémon existants les plus proches (distance euclidienne sur stats standardisées)."""
    z = models.scaler.transform(_as_frame(stats))
    Z = models.scaler.transform(df[STATS])
    d = np.sqrt(((Z - z) ** 2).sum(axis=1))
    out = df.assign(distance=d)
    if exclude_name:
        out = out[out["name"] != exclude_name]
    return out.nsmallest(k, "distance")


# ==========================================================================
# Niveau 2 : prédiction à partir d'une image
# ==========================================================================
IMAGE_MODEL = MODELS_DIR / "image_model.keras"            # approche B : 142 espèces (modèle retenu)
IMAGE_MODEL_TYPES = MODELS_DIR / "image_model_types.keras"  # approche A : 17 types (second avis)
IMAGE_CLASSES = MODELS_DIR / "image_classes.json"         # indice de classe <-> numéro de Pokédex
IMAGE_THRESHOLDS = MODELS_DIR / "seuils_types.npy"        # seuils par type de l'approche A


@dataclass
class ImageModels:
    species: object
    types: object | None
    classes: list[dict]          # [{"index", "pokedex_number", "name", "type1", "type2"}, ...]
    type_names: list[str]
    thresholds: np.ndarray | None


def load_image_models(with_types: bool = True) -> ImageModels:
    """Charge le modèle d'images retenu (et, en option, le modèle de types)."""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import keras
    import src.train_images  # noqa: F401  (enregistre la loss WeightedBCE pour la désérialisation)
    meta = json.loads(IMAGE_CLASSES.read_text(encoding="utf-8"))
    types = None
    if with_types and IMAGE_MODEL_TYPES.exists():
        types = keras.saving.load_model(IMAGE_MODEL_TYPES, compile=False)
    return ImageModels(species=keras.saving.load_model(IMAGE_MODEL, compile=False), types=types,
                       classes=meta["classes"], type_names=meta["types"],
                       thresholds=np.load(IMAGE_THRESHOLDS) if IMAGE_THRESHOLDS.exists() else None)


def prepare_image(source, mode: str = "pad") -> np.ndarray:
    """Même prétraitement qu'à l'entraînement : RGB sur fond blanc, carré, 224 x 224.

    mode="pad" pour une image importée (on ne coupe pas le Pokémon),
    mode="center_crop" pour la caméra (le Pokémon est au centre du cadre).
    """
    from src.image_utils import IMG_SIZE, load_rgb, to_square
    return np.asarray(to_square(load_rgb(source), IMG_SIZE, mode=mode), dtype=np.float32)


def predict_image(models: ImageModels, source, mode: str = "pad", k: int = 5) -> dict:
    """Espèces les plus probables + types (déduits de l'espèce, et second avis du modèle A)."""
    x = prepare_image(source, mode)[None]
    p = models.species.predict(x, verbose=0)[0]
    order = np.argsort(p)[::-1]
    top = [{**models.classes[i], "proba": float(p[i])} for i in order[:k]]

    # P(type) = somme des probabilités des espèces qui ont ce type
    p_type_B = {t: 0.0 for t in models.type_names}
    for c, pc in zip(models.classes, p):
        for t in {c["type1"], c["type2"]} - {"aucun"}:
            p_type_B[t] += float(pc)
    out = {"top": top, "types_espece": [t for t in (top[0]["type1"], top[0]["type2"]) if t != "aucun"],
           "proba_types_espece": dict(sorted(p_type_B.items(), key=lambda kv: -kv[1])),
           "image": x[0].astype(np.uint8)}
    if models.types is not None:
        from src.image_utils import decide_types
        pa = models.types.predict(x, verbose=0)[0]
        d = decide_types(pa[None], models.thresholds if models.thresholds is not None else 0.5)[0]
        out["proba_types_modele_A"] = dict(sorted(zip(models.type_names, map(float, pa)), key=lambda kv: -kv[1]))
        out["types_modele_A"] = [t for t, keep in zip(models.type_names, d) if keep]
    return out
