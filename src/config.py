"""Configuration centrale du projet Pokédex (niveau 1).

Tout ce qui doit être identique entre le notebook, src/train.py et app.py
est défini ici : graine aléatoire, chemins, listes de colonnes.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Reproductibilité
# --------------------------------------------------------------------------
# Une seule graine, utilisée pour le split, la validation croisée, K-means,
# les forêts aléatoires, SMOTE, etc. Mêmes données + même graine => mêmes
# résultats à chaque exécution.
RANDOM_STATE = 42
TEST_SIZE = 0.20      # 20 % des Pokémon mis de côté pour l'évaluation finale
N_SPLITS_CV = 5       # validation croisée stratifiée à 5 plis

# --------------------------------------------------------------------------
# Chemins (relatifs à la racine du projet, quel que soit le dossier courant)
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "pokemon.csv"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# --------------------------------------------------------------------------
# Colonnes
# --------------------------------------------------------------------------
# Les 6 statistiques de base : c'est exactement ce que l'utilisateur saisit
# dans l'application, donc les SEULES variables d'entrée des modèles.
STATS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]

# Variante « physique » à comparer : on ajoute taille et poids.
PHYSICAL = ["height_m", "weight_kg"]

FEATURE_SETS = {
    "stats": STATS,
    "stats_phys": STATS + PHYSICAL,
}

# Libellés français pour les graphiques et l'appli
STAT_LABELS = {
    "hp": "PV",
    "attack": "Attaque",
    "defense": "Défense",
    "sp_attack": "Att. Spé.",
    "sp_defense": "Déf. Spé.",
    "speed": "Vitesse",
    "height_m": "Taille (m)",
    "weight_kg": "Poids (kg)",
}

# Cibles
TARGET_LEGENDARY = "is_legendary"   # classification binaire
TARGET_TYPE = "type1"               # classification multiclasse (18 types)

# Colonnes conservées pour l'analyse (EDA, croisements), pas comme entrées
INFO_COLS = ["pokedex_number", "name", "type1", "type2", "generation",
             "is_legendary", "base_total"]

# --------------------------------------------------------------------------
# Colonnes exclues pour éviter une fuite de données
# --------------------------------------------------------------------------
# Une variable « fuit » si elle contient (directement ou presque) la réponse,
# OU si elle ne sera pas disponible au moment de la prédiction. Ici l'appli ne
# reçoit que les 6 stats : tout le reste est exclu des modèles.
LEAKY_COLS = {
    "capture_rate": "fixé par le jeu : quasi toujours 3 pour les légendaires",
    "base_egg_steps": "nombre de pas pour éclore : très élevé (~30 000) pour les légendaires",
    "base_happiness": "bonheur de base : souvent 0 ou 35 pour les légendaires",
    "experience_growth": "courbe d'XP : « lente » pour presque tous les légendaires",
    "percentage_male": "manquant pour les Pokémon asexués, dont la plupart des légendaires",
    "classfication": "texte descriptif (ex. « Legendary Pokémon »)",
    "pokedex_number": "les légendaires sont regroupés en fin de génération",
    "base_total": "somme exacte des 6 stats : redondante (combinaison linéaire)",
    "abilities": "talents spécifiques aux légendaires (ex. « Pressure »)",
    "japanese_name": "identifiant, aucune information généralisable",
    "name": "identifiant, aucune information généralisable",
}
# Les 18 colonnes against_* (multiplicateurs de dégâts reçus) sont une
# fonction déterministe du (type1, type2) : elles « trichent » pour la
# prédiction du type et n'existent pas dans l'entrée de l'appli.
LEAKY_PREFIXES = ("against_",)
