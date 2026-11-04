"""Entraîne les modèles du niveau 1 sur les 801 Pokémon et les sauvegarde.

Usage (depuis la racine du projet) :
    python -m src.train_tabular

Les choix (modèles, hyperparamètres) viennent du notebook
notebooks/1_tabulaire.ipynb, parties 2 à 5. Ce script ne fait AUCUNE
sélection : il réentraîne la configuration retenue sur toutes les données,
une fois l'évaluation terminée (le test a servi une seule fois, dans le
notebook). Plus de données = modèles un peu meilleurs pour l'appli.

Fichiers produits dans models/ :
    scaler.joblib                 StandardScaler des 6 stats (pour K-means)
    kmeans.joblib                 K-means k = 5
    clusters.json                 noms, descriptions, couleurs, profils moyens
    legendary_classifier.joblib   boosting + total, recalibré (sigmoïde)
    type_classifier.joblib        SVM RBF calibré (17 types)
    metadata.json                 seuil, classes, hyperparamètres, versions
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.clustering import cluster_info, name_clusters
from src.config import (MODELS_DIR, N_SPLITS_CV, RANDOM_STATE, STATS, TARGET_LEGENDARY,
                        TARGET_TYPE)
from src.preprocessing import AddTotal, get_X, load_clean

# --------------------------------------------------------------------------
# Configuration retenue dans le notebook
# --------------------------------------------------------------------------
N_CLUSTERS = 5                                   # partie 2 : K-means k = 5
KMEANS_N_INIT = 20

LEGENDARY_PARAMS = dict(learning_rate=0.1, max_depth=3, max_iter=100,  # partie 3.6
                        min_samples_leaf=5)
TYPE_SVM_PARAMS = dict(C=10, gamma=0.03)          # partie 4.3
TYPES_EXCLUS = ["flying"]                        # partie 4.0 : 3 exemples seulement


def make_legendary_model() -> CalibratedClassifierCV:
    """Gradient boosting sur 6 stats + total, recalibré par une sigmoïde (Platt)."""
    boosting = Pipeline([
        ("features", AddTotal()),
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE, **LEGENDARY_PARAMS)),
    ])
    return CalibratedClassifierCV(boosting, method="sigmoid", cv=N_SPLITS_CV)


def make_type_model() -> Pipeline:
    """SVM RBF + calibration de Platt (probabilités pour le top-3)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", CalibratedClassifierCV(SVC(**TYPE_SVM_PARAMS), method="sigmoid",
                                         cv=N_SPLITS_CV, ensemble=False)),
    ])


def choose_threshold(model, X, y) -> tuple[float, dict]:
    """Seuil qui maximise le F1 sur des probabilités HORS ÉCHANTILLON."""
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(thr[i]), {"f1": float(f1[i]), "precision": float(prec[i]), "recall": float(rec[i]),
                           "f1_a_0.5": float(f1_score(y, p >= 0.5))}


def train_and_save(verbose: bool = True) -> dict:
    """Entraîne les 3 modèles sur toutes les données et écrit models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    X = get_X(df)
    say = print if verbose else (lambda *a, **k: None)
    say(f"Données : {len(df)} Pokémon")

    # 1) Clustering ---------------------------------------------------------
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE).fit(Z)
    keys = name_clusters(kmeans.cluster_centers_)
    clusters = {}
    for c, key in keys.items():
        m = kmeans.labels_ == c
        info = cluster_info(key)
        clusters[str(c)] = {**info,
                            "effectif": int(m.sum()),
                            "pct_legendaires": float(df.loc[m, TARGET_LEGENDARY].mean() * 100),
                            "profil_moyen": {s: float(v) for s, v in df.loc[m, STATS].mean().items()},
                            "exemples": df.loc[m].assign(d=kmeans.transform(Z)[m, c])
                                          .nsmallest(5, "d")["name"].tolist()}
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(kmeans, MODELS_DIR / "kmeans.joblib")
    (MODELS_DIR / "clusters.json").write_text(json.dumps(clusters, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    say("Clustering :", {v["name"]: v["effectif"] for v in clusters.values()})

    # 2) Statut légendaire --------------------------------------------------
    y_leg = df[TARGET_LEGENDARY]
    seuil, perf_seuil = choose_threshold(make_legendary_model(), X, y_leg)
    leg_model = make_legendary_model().fit(X, y_leg)
    joblib.dump(leg_model, MODELS_DIR / "legendary_classifier.joblib")
    say(f"Légendaire : seuil = {seuil:.3f} (F1 hors échantillon = {perf_seuil['f1']:.3f})")

    # 3) Type principal -----------------------------------------------------
    dft = df[~df[TARGET_TYPE].isin(TYPES_EXCLUS)]
    type_model = make_type_model().fit(get_X(dft), dft[TARGET_TYPE])
    joblib.dump(type_model, MODELS_DIR / "type_classifier.joblib")
    say(f"Type : {len(type_model.classes_)} classes, entraîné sur {len(dft)} Pokémon")

    # 4) Métadonnées ---------------------------------------------------------
    meta = {
        "date_entrainement": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "n_pokemon": int(len(df)),
        "entrees": STATS,
        "clustering": {"methode": "KMeans", "k": N_CLUSTERS, "n_init": KMEANS_N_INIT},
        "legendaire": {"modele": "HistGradientBoosting (6 stats + total) + calibration sigmoïde",
                       "hyperparametres": LEGENDARY_PARAMS, "seuil": seuil,
                       "seuil_choisi_par": "F1 maximal, probabilités hors échantillon (CV 5 plis)",
                       "perf_au_seuil_hors_echantillon": perf_seuil,
                       "taux_legendaires": float(y_leg.mean())},
        "type": {"modele": "SVC RBF + calibration sigmoïde", "hyperparametres": TYPE_SVM_PARAMS,
                 "classes": type_model.classes_.tolist(), "types_exclus": TYPES_EXCLUS,
                 "n_pokemon": int(len(dft))},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "joblib": joblib.__version__},
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    say(f"Modèles écrits dans {MODELS_DIR}")
    return {"scaler": scaler, "kmeans": kmeans, "clusters": clusters, "legendary": leg_model,
            "type": type_model, "metadata": meta}


if __name__ == "__main__":
    train_and_save()
