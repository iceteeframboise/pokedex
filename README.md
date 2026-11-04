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
data/pokemon.csv          dataset Kaggle (801 Pokémon)
data/images/              images de la 1re génération (niveau 2, à venir)
notebooks/1_tabulaire.ipynb   niveau 1 : EDA, clustering, classification (commenté)
notebooks/2_images.ipynb      niveau 2 : images (à venir)
src/config.py             graine (42), chemins, colonnes utilisées / exclues
src/preprocessing.py      chargement, nettoyage, split stratifié, prétraitement
src/clustering.py         noms et interprétation des clusters
src/viz.py                style graphique commun
src/train_tabular.py      entraîne et sauvegarde les modèles du niveau 1 (à venir)
src/train_images.py, src/image_utils.py, src/pokedex_data.py   (niveaux 2 et 3, à venir)
models/                   modèles sauvegardés (.joblib, .pt)
reports/figures/          figures exportées pour le rapport
app.py                    application Pokédex Streamlit (à venir)
```

Exécuter le notebook : `jupyter notebook notebooks/1_tabulaire.ipynb`
