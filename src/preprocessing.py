"""Chargement, nettoyage et préparation des données Pokémon.

Fonctions utilisées par le notebook, par src/train.py et (pour la liste des
Pokémon) par app.py. Point de vigilance : aucune statistique (moyenne,
médiane, écart-type) n'est calculée ici sur l'ensemble des données. Les
calculs qui « apprennent » des données (imputation, standardisation) sont
dans un Pipeline scikit-learn, ajusté uniquement sur le train.

Exécution rapide pour vérifier les données :
    python -m src.preprocessing
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import (
    DATA_PATH, FEATURE_SETS, LEAKY_COLS, LEAKY_PREFIXES, PHYSICAL,
    RANDOM_STATE, STATS, TARGET_LEGENDARY, TARGET_TYPE, TEST_SIZE,
)

NO_TYPE2 = "aucun"   # valeur de remplacement pour les Pokémon à un seul type


# ==========================================================================
# 1. Chargement et nettoyage
# ==========================================================================
def load_raw(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Lit le CSV brut de Kaggle sans aucune transformation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Téléchargez « The Complete Pokemon Dataset » (Rounak Banik) sur "
            "Kaggle et placez pokemon.csv dans le dossier data/."
        )
    return pd.read_csv(path)


def _parse_capture_rate(value) -> float:
    """Convertit capture_rate en nombre.

    Une ligne (Minior) vaut « 30 (Meteorite)255 (Core) » : deux formes du
    même Pokémon ont deux taux. On garde le premier nombre (30, forme
    Météore, la forme par défaut). La colonne est de toute façon exclue des
    modèles (fuite de données) ; on la répare seulement pour l'EDA.
    """
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d+", str(value))
    return float(match.group()) if match else np.nan


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage déterministe : aucune statistique n'est apprise des données.

    - noms nettoyés (espaces) ;
    - capture_rate converti en numérique ;
    - type2 manquant -> « aucun » : l'absence de second type est une
      information réelle, pas une donnée perdue, on ne l'impute donc pas ;
    - is_legendary et generation en entiers ;
    - height_m / weight_kg : on NE remplit PAS les valeurs manquantes ici.
      L'imputation par la médiane est faite dans le Pipeline, avec la médiane
      du train uniquement (sinon le test « influence » le train = fuite).
    """
    df = df.copy()
    df["name"] = df["name"].astype(str).str.strip()
    if "capture_rate" in df.columns:
        df["capture_rate"] = df["capture_rate"].map(_parse_capture_rate)
    df["type1"] = df["type1"].astype(str).str.strip().str.lower()
    df["type2"] = df["type2"].fillna(NO_TYPE2).astype(str).str.strip().str.lower()
    df[TARGET_LEGENDARY] = df[TARGET_LEGENDARY].astype(int)
    df["generation"] = df["generation"].astype(int)
    for col in STATS + PHYSICAL:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_clean(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Raccourci : chargement puis nettoyage."""
    return clean(load_raw(path))


# ==========================================================================
# 2. Contrôle qualité
# ==========================================================================
def quality_report(df: pd.DataFrame) -> dict:
    """Résumé des points à surveiller (pour le notebook et le rapport)."""
    n_leg = int(df[TARGET_LEGENDARY].sum())
    return {
        "n_pokemon": len(df),
        "n_colonnes": df.shape[1],
        "doublons_nom": int(df["name"].duplicated().sum()),
        "manquants": df[STATS + PHYSICAL + ["type2"]].isna().sum()
                       .loc[lambda s: s > 0].to_dict(),
        "sans_type2": int((df["type2"] == NO_TYPE2).sum()),
        "pokemon_sans_taille_poids": df.loc[
            df[PHYSICAL].isna().any(axis=1), "name"].tolist(),
        "n_legendaires": n_leg,
        "taux_legendaires": round(n_leg / len(df), 4),
        "n_types1": int(df[TARGET_TYPE].nunique()),
        "effectif_type1_min": df[TARGET_TYPE].value_counts().tail(3).to_dict(),
        "stats_min_max": df[STATS].agg(["min", "max"]).to_dict(),
    }


def leaky_columns(df: pd.DataFrame) -> list[str]:
    """Liste des colonnes présentes dans df qui sont exclues pour fuite."""
    cols = [c for c in LEAKY_COLS if c in df.columns]
    cols += [c for c in df.columns if c.startswith(LEAKY_PREFIXES)]
    return cols


# ==========================================================================
# 3. Variables d'entrée et découpage train/test
# ==========================================================================
def get_X(df: pd.DataFrame, variant: str = "stats") -> pd.DataFrame:
    """Matrice des variables d'entrée selon la variante choisie.

    variant = "stats"      -> les 6 stats de base (modèle de l'appli)
    variant = "stats_phys" -> 6 stats + taille + poids (variante comparée)
    """
    if variant not in FEATURE_SETS:
        raise ValueError(f"Variante inconnue : {variant!r} "
                         f"(choix : {list(FEATURE_SETS)})")
    return df[FEATURE_SETS[variant]].copy()


def split_train_test(df: pd.DataFrame, stratify_col: str = TARGET_LEGENDARY,
                     test_size: float = TEST_SIZE):
    """Découpage train/test STRATIFIÉ, reproductible (random_state=42).

    Stratifier = conserver dans le train ET dans le test la même proportion
    de chaque classe que dans le jeu complet. Indispensable ici : avec ~9 %
    de légendaires, un tirage purement aléatoire pourrait n'en mettre que
    très peu dans le test, et les métriques seraient instables.

    Renvoie (df_train, df_test) avec leurs index d'origine (traçabilité).
    """
    return train_test_split(df, test_size=test_size, stratify=df[stratify_col],
                            random_state=RANDOM_STATE)


# ==========================================================================
# 4. Pré-traitement « apprenant » (à mettre DANS un Pipeline)
# ==========================================================================
def make_preprocessor(variant: str = "stats") -> Pipeline | ColumnTransformer:
    """Construit le pré-traitement à placer en tête de chaque Pipeline.

    Standardisation (z-score) : z = (x - moyenne) / écart-type, calculés sur
    le train. Chaque variable a alors une moyenne de 0 et un écart-type de 1.
    Pourquoi c'est nécessaire :
      - k-NN, K-means et SVM reposent sur des distances euclidiennes : sans
        mise à l'échelle, une variable de grande amplitude (poids jusqu'à
        ~1000 kg) écraserait les autres (stats entre 1 et 255) ;
      - la régression logistique régularisée pénalise les coefficients de la
        même façon : il faut que les variables soient comparables ;
      - pour les arbres (CART, Random Forest, boosting), c'est sans effet
        (les seuils de coupure sont invariants par transformation monotone),
        mais on la garde pour avoir des pipelines homogènes.

    Variante "stats_phys" : taille et poids sont très asymétriques (quelques
    Pokémon géants), on applique log(1 + x) avant la standardisation, et on
    remplace les valeurs manquantes par la médiane du train (robuste aux
    valeurs extrêmes, contrairement à la moyenne).
    """
    if variant == "stats":
        # Les 6 stats n'ont aucune valeur manquante : on standardise seulement.
        return Pipeline([("scaler", StandardScaler())])

    if variant == "stats_phys":
        phys = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ])
        return ColumnTransformer([
            ("stats", StandardScaler(), STATS),
            ("phys", phys, PHYSICAL),
        ])

    raise ValueError(f"Variante inconnue : {variant!r}")


# ==========================================================================
# Vérification rapide en ligne de commande
# ==========================================================================
if __name__ == "__main__":
    from pprint import pprint

    raw = load_raw()
    print(f"Colonnes exclues pour fuite ({len(leaky_columns(raw))}) :")
    print(", ".join(leaky_columns(raw)))
    print(f"capture_rate brut non numérique : "
          f"{raw.loc[pd.to_numeric(raw['capture_rate'], errors='coerce').isna(), 'capture_rate'].tolist()}")

    df = clean(raw)
    print("\n=== Contrôle qualité ===")
    pprint(quality_report(df), sort_dicts=False, width=100)

    train, test = split_train_test(df)
    print("\n=== Split stratifié sur is_legendary ===")
    print(f"train : {len(train)} Pokémon, {train[TARGET_LEGENDARY].mean():.2%} de légendaires")
    print(f"test  : {len(test)} Pokémon, {test[TARGET_LEGENDARY].mean():.2%} de légendaires")

    for variant in ("stats", "stats_phys"):
        Xt = make_preprocessor(variant).fit_transform(get_X(train, variant))
        print(f"{variant:>10} -> {Xt.shape}, NaN restants : {int(np.isnan(Xt).sum())}, "
              f"moyennes ≈ {np.round(Xt.mean(0), 2).tolist()}")
