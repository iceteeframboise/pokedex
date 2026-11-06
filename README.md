# Mon Pokédex Pokémon : niveau 1

Projet du module *Fondements mathématiques de l'IA* : clustering des Pokémon,
prédiction du statut légendaire et du type à partir des 6 statistiques de base,
et application Pokédex (Streamlit).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Données

Télécharger `pokemon.csv` (« The Complete Pokemon Dataset », Kaggle, 801 Pokémon)
et le placer dans `data/pokemon.csv`. Vérification :

```bash
python -m src.preprocessing
```

## Entraîner les modèles (niveau 1)

```bash
python -m src.train_tabular
```

Ce script réentraîne sur les 801 Pokémon la configuration choisie dans
`notebooks/1_tabulaire.ipynb`, en environ 10 s sur CPU, et écrit dans `models/` :

| Fichier | Contenu |
|---|---|
| `scaler.joblib`, `kmeans.joblib` | standardisation des 6 stats + K-means (k = 5) |
| `clusters.json` | nom, description, couleur et profil moyen de chaque groupe |
| `legendary_classifier.joblib` | gradient boosting (6 stats + total), recalibré |
| `type_classifier.joblib` | SVM RBF calibré (17 types) |
| `metadata.json` | seuil « légendaire probable », classes, hyperparamètres, versions |

Les modèles fournis dans le dépôt sont déjà entraînés : l'appli les charge sans rien réentraîner.

## Lancer l'application

```bash
streamlit run app.py
```

L'appli s'ouvre dans le navigateur (http://localhost:8501). Onglet **Statistiques** (niveau 1) :
choisir un Pokémon existant pour préremplir les 6 stats, ou les saisir librement. L'appli affiche :
- le **groupe** (cluster) avec son nom et son interprétation ;
- la **probabilité d'être légendaire** (calibrée), avec le verdict « légendaire probable » au-dessus du seuil ;
- le **type prédit** et les 3 types les plus probables ;
- un **radar** des stats comparé au profil moyen du groupe, et les Pokémon existants les plus proches.

Onglet **Image** (niveau 2) : importer une image d'un Pokémon de 1re génération → Pokémon reconnu (top-3),
type(s) prédit(s), second avis du modèle de types et fiche complète.

Onglet **Caméra** (niveau 3) : photo par la webcam (le navigateur demande l'autorisation), recadrage
centré → Pokémon identifié si la confiance dépasse 70 % (sinon « Pokémon non reconnu ») et **fiche
complète** : nom japonais, image de référence, types, taille, poids, talents, stats (radar), groupe,
faiblesses, résistances et immunités. Sous la photo, « 📝 Test réel : enregistrer cet essai » note ce
qui était vraiment présenté (`reports/tables/test_camera.csv`, photos dans `reports/test_camera/`) : voir
le protocole `docs/protocole_test_camera.md` ; l'analyse est la partie 8 de `notebooks/2_images.ipynb`.

**Mise en ligne** (tablette, téléphone) : Streamlit Community Cloud, https://share.streamlit.io → dépôt
`iceteeframboise/pokedex`, branche `main`, fichier `app.py`, Python 3.13 (étapes détaillées dans le protocole).

Onglet **Pokédex** (niveau 3) : catalogue des 801 Pokémon (vignettes, stats, groupe), recherche par nom,
filtres (type 1 ou 2, génération, légendaire, groupe), tri par n'importe quelle stat, clic sur une ligne →
fiche complète ; vue **« Organisation par types »** : nombre de Pokémon par type, heatmap des stats
moyennes par type et liste des Pokémon d'un type.

## Niveau 2 : images

Les modèles d'images sont fournis dans `models/` (`image_model.keras`, `image_model_types.keras`,
`image_classes.json`, `seuils_types.npy`) : l'onglet **Image** de l'appli fonctionne sans les données.

Pour refaire les expériences du notebook `notebooks/2_images.ipynb`, il faut les images (≈ 3 Go, hors git) :

```bash
python -m src.image_utils          # télécharge le dataset Kaggle, nettoie, exporte data/images_clean/
python -m src.train_images --model cnn_scratch --approach B        # CNN de zéro (≈ 1 h 30 sur CPU)
python -m src.train_images --finetune efficientnet_b0 --approach B # fine-tuning (≈ 50 min sur CPU)
```

Sans accès à kaggle.com, téléchargez l'archive à la main
(https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files)
et décompressez-la dans `data/images/` (on doit obtenir `data/images/pokemon/<NomDuPokémon>/…`).

## Tests

```bash
python -m pytest
```

## Structure

```
data/pokemon.csv          dataset Kaggle (801 Pokémon)
data/images/              images de la 1re génération (niveau 2, hors git)
data/images_index.csv     inventaire et décision de nettoyage de chaque image
data/images_splits.csv    découpages train/val/test et plis groupés par espèce
notebooks/1_tabulaire.ipynb   niveau 1 : EDA, clustering, classification (commenté)
notebooks/2_images.ipynb      niveau 2 : images (nettoyage, référence classique, CNN, transfer learning, couverture, Grad-CAM)
src/config.py             graine (42), chemins, colonnes utilisées / exclues
src/preprocessing.py      chargement, nettoyage, split stratifié, prétraitement
src/clustering.py         noms et interprétation des clusters
src/viz.py                style graphique commun
src/pokedex_data.py       chargement des modèles et prédictions pour l'appli
src/train_tabular.py      entraîne et sauvegarde les modèles du niveau 1
src/image_utils.py        images : association au CSV, nettoyage, doublons, découpages, métriques
src/train_images.py       réseaux de neurones : augmentation, CNN, transfer learning, Grad-CAM
models/                   modèles sauvegardés (.joblib, .pt)
assets/reference/         une image de référence par Pokémon de 1re génération (fiche de l'appli)
reports/figures/          figures exportées pour le rapport
reports/tables/           tableaux récapitulatifs (CSV) pour le rapport
app.py                    application Pokédex Streamlit
tests/                    tests des prédictions (pytest)
```

Exécuter le notebook : `jupyter notebook notebooks/1_tabulaire.ipynb`
