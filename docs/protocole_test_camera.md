# Protocole du test caméra en conditions réelles (niveau 3)

**Objectif** : mesurer le taux de réussite de l'appli devant une vraie webcam, et le comparer au score du
modèle sur le jeu de test (91 % de Pokémon bien identifiés). L'écart mesure l'effet du « changement de
domaine » : reflets, lumière, flou, perspective, fond, bord du téléphone… Ce résultat va dans la partie
**Limites** du rapport.

Durée : environ 15 minutes pour 20 essais.

## 1. Préparer

1. Lancer l'appli **sur votre ordinateur**, dans le dossier `pokedex` : `streamlit run app.py`
   (installation : voir le README). Ouvrir l'onglet **📷 Caméra** et autoriser la caméra dans le navigateur.
2. Préparer les images à présenter, par exemple sur votre téléphone (recherche d'images), ou en
   imprimant quelques images :
   - **15 Pokémon différents de 1re génération** (n° 1 à 151), variés : faciles (Pikachu, Dracaufeu,
     Ectoplasma…) et difficiles (Roucoups, Kadabra, Tartard, des Pokémon d'une même lignée…) ;
   - **3 Pokémon inconnus du modèle** : 1 ou 2 parmi les 9 absents du dataset (Nidoran♀, Nidoran♂,
     Paras, Persian, Grolem, Grotadmorv, Onix, Krabby, Kabuto) et 1 d'une autre génération (ex. Lucario) ;
   - **2 objets qui ne sont pas des Pokémon** (une tasse, une main, un logo…).
3. Les images de test ne doivent **pas** venir du dataset d'entraînement, autant que possible :
   prenez des images trouvées ailleurs (sites, jeux, anime, cartes, peluches…).

## 2. Faire les essais (20 au total)

Pour chaque essai :

1. Présenter l'image **au centre** du cadre, assez grande (le carré central est ce que voit le modèle).
2. Cliquer sur **Take Photo**.
3. Noter ce que l'appli affiche (le résultat s'affiche tout seul).
4. Ouvrir **« 📝 Test réel : enregistrer cet essai »**, choisir ce que vous avez **vraiment** présenté,
   le support et les conditions, puis **Enregistrer l'essai**.
5. Cliquer sur **Clear photo** et passer à l'essai suivant.

Variez les conditions sur quelques essais (c'est intéressant pour l'analyse) : lumière du jour /
lumière artificielle, avec et sans reflet sur l'écran, image plus petite ou un peu de travers.

Le bilan en direct s'affiche dans **« 📊 Bilan du test réel »**.

## 3. Me transmettre les résultats

Les essais sont enregistrés dans :
- `reports/tables/test_camera.csv` (une ligne par essai) ;
- `reports/test_camera/essai_XX.jpg` (les photos prises).

Deux possibilités :
- **pousser sur GitHub** : `git add reports/ && git commit -m "Test caméra réel" && git push` ;
- **ou** déposer le fichier `test_camera.csv` (et, si possible, quelques photos) sur GitHub avec
  « Add file → Upload files » dans `reports/tables/`.

J'analyse ensuite les résultats dans le notebook 2 (partie 8) : taux de réussite comparé au test,
rejet des inconnus, influence du support et des conditions, exemples d'erreurs.

## Bon à savoir

- Les photos prises sont enregistrées **sur votre ordinateur**, dans le dossier du projet, et ne sont
  envoyées nulle part ailleurs. Évitez de photographier des personnes.
- Le modèle ne connaît que les **142 Pokémon de 1re génération** présents dans le dataset.
