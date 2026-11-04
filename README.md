# Mon Pokédex Pokémon : niveau 1

Projet du module *Fondements mathématiques de l'IA* : clustering des Pokémon,
prédiction du statut légendaire et du type à partir des 6 statistiques de base,
et application Pokédex (Streamlit).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Données

Télécharger `pokemon.csv` (« The Complete Pokemon Dataset », Kaggle, 801 Pokémon)
et le placer dans `data/pokemon.csv`. Vérification :

```bash
python -m src.preprocessing
```

## Structure

```
data/            pokemon.csv
notebook/        pokedex_niveau1.ipynb (analyse complète, commentée)
src/config.py    graine (42), chemins, colonnes utilisées / exclues
src/preprocessing.py  chargement, nettoyage, split stratifié, pré-traitement
src/train.py     entraîne et sauvegarde les modèles (à venir)
models/          modèles .joblib
reports/figures/ figures pour le rapport
app.py           application Pokédex (à venir)
```
