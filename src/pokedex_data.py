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
