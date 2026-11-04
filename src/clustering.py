"""Interprétation des clusters : noms, descriptions, profils.

Le numéro d'un cluster produit par K-means (0, 1, 2…) est arbitraire : il
dépend de l'initialisation. Pour que le notebook, src/train_tabular.py et
l'appli affichent les mêmes noms, on nomme chaque cluster à partir de son
CENTRE (profil moyen standardisé), par une règle déterministe :

1. le cluster dont la moyenne des 6 z-scores est la plus basse
   -> « Pokémon faibles / non évolués » ;
2. celui dont la moyenne est la plus haute -> « Puissants / quasi-légendaires » ;
3. pour les autres, on cherche la stat qui se détache le plus du niveau
   général du cluster (z_stat - moyenne des z du cluster). Cette « stat
   dominante » donne le nom : Vitesse -> attaquants rapides, Défense -> tanks…

La règle a été vérifiée à la main sur les profils et les Pokémon
représentatifs (voir le notebook, partie 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import STATS

# Catalogue des profils possibles : clé -> (nom, description, couleur)
# Couleurs : sous-ensemble de la palette catégorielle validé pour les
# daltoniens (5 teintes deux à deux distinguables).
CLUSTER_CATALOG = {
    "faibles": (
        "Pokémon faibles / non évolués",
        "Toutes les stats sous la moyenne. Ce sont surtout des Pokémon de base "
        "ou de premier stade (Chenipan, Rattata, starters avant évolution) : "
        "ils doivent évoluer pour devenir forts.",
        "#1baf7a"),
    "elite": (
        "Puissants / quasi-légendaires",
        "Toutes les stats nettement au-dessus de la moyenne, en particulier "
        "l'Attaque spéciale et la Défense spéciale. Ce groupe rassemble la "
        "grande majorité des légendaires, les pseudo-légendaires (Dracolosse, "
        "Métalosse…) et les méga-évolutions.",
        "#6250d6"),
    "rapides": (
        "Attaquants rapides",
        "Vitesse très élevée, attaques correctes, défenses moyennes : ils "
        "frappent en premier mais encaissent mal (Électrik, Vol, Insecte "
        "évolués…).",
        "#eda100"),
    "tanks": (
        "Tanks défensifs",
        "Défense (et souvent Défense spéciale) très élevée, Vitesse faible : "
        "lents mais très résistants. Beaucoup de types Roche et Acier.",
        "#008300"),
    "endurants": (
        "Colosses à PV",
        "Énorme réserve de PV et bonne Attaque, défenses dans la moyenne, plutôt "
        "lents : ils encaissent grâce à leurs PV plutôt qu'à leur armure.",
        "#e87ba4"),
    "physiques": (
        "Cogneurs physiques",
        "Attaque physique dominante, Attaque spéciale en retrait.",
        "#e34948"),
    "speciaux": (
        "Attaquants spéciaux",
        "Attaque spéciale dominante, Attaque physique en retrait.",
        "#2a78d6"),
}

# Stat dominante -> profil
_DOMINANT_TO_KEY = {
    "speed": "rapides",
    "defense": "tanks",
    "sp_defense": "tanks",
    "hp": "endurants",
    "attack": "physiques",
    "sp_attack": "speciaux",
}


def name_clusters(centers_z: np.ndarray | pd.DataFrame) -> dict[int, str]:
    """Associe à chaque numéro de cluster une clé du CLUSTER_CATALOG.

    centers_z : centres des clusters dans l'espace STANDARDISÉ
                (shape = n_clusters × 6, colonnes dans l'ordre de STATS).
    """
    c = pd.DataFrame(np.asarray(centers_z), columns=STATS)
    level = c.mean(axis=1)                       # niveau général du cluster
    keys: dict[int, str] = {int(level.idxmin()): "faibles",
                            int(level.idxmax()): "elite"}
    used = set(keys.values())
    # Les clusters restants : du plus « typé » au moins typé
    rest = [i for i in c.index if i not in keys]
    rel = c.loc[rest].sub(level.loc[rest], axis=0)    # écart à son propre niveau
    for i in rel.max(axis=1).sort_values(ascending=False).index:
        for stat in rel.loc[i].sort_values(ascending=False).index:
            key = _DOMINANT_TO_KEY[stat]
            if key not in used:
                keys[int(i)] = key
                used.add(key)
                break
        else:                                    # tous les profils déjà pris
            keys[int(i)] = "polyvalents"
    return dict(sorted(keys.items()))


def cluster_info(key: str) -> dict:
    """Nom, description et couleur d'un profil (pour l'affichage)."""
    name, desc, color = CLUSTER_CATALOG.get(
        key, ("Polyvalents", "Profil équilibré, sans stat dominante.", "#898781"))
    return {"key": key, "name": name, "description": desc, "color": color}


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Profil moyen (stats brutes) de chaque cluster + effectif et % légendaires."""
    g = df.assign(cluster=labels).groupby("cluster")
    prof = g[STATS + ["base_total"]].mean()
    prof["effectif"] = g.size()
    prof["% légendaires"] = g["is_legendary"].mean() * 100
    return prof
