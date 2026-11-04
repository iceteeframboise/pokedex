# Rapport : résultats et figures du niveau 1

Liste de ce qu'il faut mettre dans le rapport pour le niveau 1. Pour chaque section : ce qu'il faut dire, les figures et tableaux à reprendre (avec une proposition de légende), et les chiffres à citer.

- Figures : `reports/figures/`
- Tableaux : `reports/tables/` (CSV)
- Tout est reproductible avec `random_state = 42` : relancer `notebooks/1_tabulaire.ipynb`, puis `python -m src.train_tabular`.

**Budget conseillé pour le niveau 1 : environ 5 pages**, soit à peu près 12 figures. Les figures marquées ★ sont indispensables. Les autres peuvent aller en annexe si la place manque.

---

## 1. Introduction et données (≈ ½ page)

**À dire**
- **Le jeu de données :** Kaggle « The Complete Pokemon Dataset » (serebii.net), 801 Pokémon des générations 1 à 7, 41 colonnes.
- **Les trois objectifs :**
  1. regrouper les Pokémon par profil de stats (clustering) ;
  2. prédire la probabilité d'être légendaire ;
  3. prédire le type principal.
  
  Tout cela à partir des **6 stats de base**, qui sont la seule entrée de l'appli.
- **Le nettoyage :**
  - `capture_rate` contient une valeur mal formée pour Minior (« 30 (Meteorite)255 (Core) »), corrigée en 30 ;
  - `type2` est manquant pour 384 Pokémon à un seul type, remplacé par « aucun » (c'est une information, pas une donnée perdue) ;
  - `height_m` et `weight_kg` sont manquants pour 20 Pokémon, surtout des formes d'Alola. Les valeurs sont imputées par la médiane **dans le pipeline**, sur le train seulement.

**Fuite de données (paragraphe important)**
- **29 colonnes sont exclues.** Exemples : `capture_rate`, `base_egg_steps`, `base_happiness`, `experience_growth`, `percentage_male`, `pokedex_number`, `against_*`.
- **Preuve chiffrée** : des règles naïves, sans aucun apprentissage, retrouvent presque tous les légendaires.
  - `capture_rate ≤ 3` : 53 légendaires sur 70 ;
  - `base_egg_steps ≥ 20 480` : 63 sur 70 ;
  - `percentage_male` manquant : 90 % des légendaires, contre 5 % des autres.
- Les colonnes `against_*` sont une fonction directe des types.
- `base_total` est redondante pour un modèle linéaire (c'est la somme exacte des 6 stats), mais elle est utile pour les arbres. Elle est donc **recalculée dans le pipeline** (voir §4).

| Figure | Légende proposée |
|---|---|
| ★ `01_fuite_de_donnees.png` | « Les colonnes exclues trichent : `capture_rate` vaut 3 pour la plupart des légendaires, et 90 % d'entre eux sont asexués. » |

---

## 2. Exploration des données (≈ 1 page)

**Chiffres à citer**
- **Les stats :**
  - moyennes de 66 à 78, écarts-types de 27 à 32 ;
  - distributions asymétriques à droite (asymétrie de 1,8 pour les PV) ;
  - valeurs extrêmes réelles : Leuphorie (255 PV), Caratroc (230 en Défense et en Défense spéciale), Mewtwo (194 en Attaque spéciale).
- **Les corrélations :**
  - toutes positives, de 0,01 à 0,53 avec Pearson, ce qui révèle un facteur de « puissance » ;
  - les plus fortes : Défense–Déf. Spé. (0,53), Att. Spé.–Déf. Spé. (0,51) ;
  - Défense–Vitesse vaut 0,01 : c'est le compromis entre Pokémon blindé et Pokémon rapide.
- **Les légendaires :**
  - 70 sur 801, soit 8,7 % ;
  - total médian de 600, contre 418 pour les autres ;
  - 31 non-légendaires dépassent 600 (pseudo-légendaires et méga-évolutions) ;
  - taux par génération : de 3 % (génération 1) à 21 % (génération 7).
- **Les types :**
  - très déséquilibrés : Eau 114, Normal 105, … Fée 18, Vol 3 ;
  - Vol est surtout un second type (95 Pokémon) ;
  - les types typés ont une signature de stats : Acier +1,5 en Défense, Vol +1,2 en Vitesse. L'Eau, la Plante et le Poison n'en ont pas.

| Figure | Légende proposée |
|---|---|
| ★ `02_distributions_stats.png` | Distribution des 6 stats, avec la médiane. Asymétrie à droite, d'où la standardisation. |
| ★ `03_stats_par_statut.png` (ou `04_base_total_par_statut.png`) | Les légendaires sont plus forts sur toutes les stats, mais les distributions se chevauchent (pseudo-légendaires, méga-évolutions). |
| ★ `05_correlations.png` | Corrélations de Pearson et de Spearman : toutes positives, ce qui indique un facteur général de « puissance ». |
| ★ `09_profil_stats_par_type.png` | Profil moyen des stats par type (z-scores). Certains types ont une signature (Acier, Combat, Électrik), d'autres non (Eau, Plante). |
| `07_repartition_types.png` | Effectifs de `type1` et `type2` : classes déséquilibrées, Vol presque toujours en second type. |
| `10_legendaires.png` | 8,7 % de légendaires, une proportion qui augmente avec les générations et se concentre sur les types Psy, Dragon et Acier. |
| annexe : `06_nuages_points.png`, `08_type1_x_type2.png`, `11_taille_poids.png` | |

---

## 3. Clustering (≈ 1 page)

**Méthodologie (formules à mettre)**
- **Standardisation :** z = (x − μ) / σ. Elle est indispensable pour la distance euclidienne.
- **K-means :** l'algorithme de Lloyd minimise l'inertie W(k) = Σ ‖z − μ_c‖². On utilise `n_init = 20`, car l'algorithme converge vers un minimum local.
- **Silhouette :** s(i) = (b − a) / max(a, b), où a est la distance moyenne aux membres du groupe et b la distance moyenne aux membres du groupe voisin.
- **Méthodes de comparaison :**
  - **CAH de Ward** : on fusionne les groupes qui font le moins augmenter l'inertie ;
  - **GMM** (algorithme EM) : le nombre de groupes et la forme des covariances sont choisis par le critère BIC = −2 ln L + p ln n.
- **ACP :** vecteurs propres de la matrice de corrélation. Elle sert uniquement à visualiser.

**Résultats à citer**
- **Choix de k = 5** :
  - la silhouette est maximale à k = 2 (0,29), mais cette partition est triviale (faibles / forts) ;
  - au-delà, elle forme un plateau entre 0,23 et 0,26 ;
  - le Davies-Bouldin est plus bas à k = 5 qu'à k = 4 ;
  - le BIC du GMM est minimal à k = 5.
- **Qualité :** silhouette de 0,234, Davies-Bouldin de 1,48. Il faut assumer que **la structure en groupes est faible** : les Pokémon forment un continuum.
- **Stabilité :**
  - ARI moyen de 0,81 ± 0,05 sur 30 rééchantillonnages bootstrap ;
  - avec une seule initialisation, l'ARI descend jusqu'à 0,52, ce qui justifie `n_init = 20`.
- **Comparaison des méthodes :**
  - CAH Ward : silhouette 0,217, ARI de 0,58 avec K-means ;
  - GMM full : silhouette −0,005, ARI de 0,12. Il modélise la densité et ses groupes se chevauchent ;
  - GMM spherical : ARI de 0,67.
- **ACP :**
  - CP1 = puissance globale (45 % de la variance) ;
  - CP2 = Vitesse contre Défense (18 %) ;
  - CP3 = physique contre spécial (13 %).
- **Les 5 groupes :**

| Groupe | Effectif | Signature (z-score) |
|---|---|---|
| faibles / non évolués | 297 | toutes les stats autour de −0,7 |
| attaquants rapides | 188 | Vitesse +0,9 |
| colosses à PV | 100 | PV +1,5 |
| tanks défensifs | 93 | Défense +1,6, Vitesse −0,7 |
| puissants / quasi-légendaires | 123 | Att. Spé. +1,5 |

- **Croisements :**
  - **avec les légendaires :** 57 des 70 légendaires (81 %) sont dans le groupe « puissants », alors que l'étiquette n'a jamais été utilisée. Les exceptions sont explicables : Deoxys, Regirock, Cosmog ;
  - **avec les types :** le lien est modéré (V de Cramér = 0,25). Acier et Roche vont chez les tanks, Électrik chez les rapides, Dragon et Psy chez les puissants.

| Figure | Légende proposée |
|---|---|
| ★ `12_coude_silhouette.png` | Coude peu marqué, silhouette en plateau au-delà de k = 2 : choix de k = 5. |
| ★ `17_acp_clusters.png` | Les 5 clusters projetés sur le plan de l'ACP (CP1 = puissance, CP2 = vitesse contre défense). Les légendaires sont à droite. |
| ★ `18_profils_clusters.png` | Centres des clusters (z-scores) et radar des profils moyens : base des noms de groupes. |
| ★ `19_clusters_legendaires.png` | 81 % des légendaires sont dans le groupe « puissants / quasi-légendaires ». |
| `16_acp_variance_cercle.png` | Variance expliquée et cercle des corrélations. |
| `20_clusters_types.png` | Répartition de chaque type entre les clusters. |
| annexe : `13_silhouette_k5.png`, `14_dendrogramme_ward.png`, `15_gmm_bic.png` | |

| Tableau | Fichier |
|---|---|
| Comparaison des méthodes de clustering | `tables/recap_clustering.csv` |

---

## 4. Prédiction du statut légendaire (≈ 1 page ½)

**Méthodologie**
- **Protocole :**
  - split stratifié 80/20 : 640 Pokémon pour le train, 161 pour le test, avec 14 légendaires dans le test, mis sous clé ;
  - validation croisée stratifiée à 5 plis sur le train ;
  - `Pipeline` qui enchaîne (total), scaler, (SMOTE) et modèle, pour éviter toute fuite ;
  - comparaison finale par validation croisée **répétée** 10 × 5 (50 plis), **appariée**.
- **Les métriques et leurs formules :** précision, rappel, F1, ROC-AUC, **PR-AUC** (critère principal), Brier, log-loss.
  - **Pourquoi pas l'accuracy :** répondre « toujours non légendaire » donne 91,2 % d'accuracy avec un rappel de 0.
- **Régression logistique :**
  - sigmoïde σ(η) = 1 / (1 + e^−η), avec un logit linéaire ;
  - odds ratio e^β ;
  - log-loss + pénalité L2, un problème convexe ;
  - la log-loss est une règle de score propre, d'où des probabilités naturellement calibrées.
- **Ingénierie de variable :** le total recalculé à partir des 6 stats (`AddTotal`). C'est inutile pour un modèle linéaire, mais décisif pour un arbre, qui ne coupe que sur une variable à la fois.

**Résultats à citer**
- **La règle « total ≥ 570 » est une référence redoutable :** rappel de 0,98, F1 de 0,78, PR-AUC de 0,70. Les légendaires ont des totaux « ronds » : 570, 580, 600, 680.
- **Rééquilibrer les classes ne classe pas mieux :**
  - `class_weight` et SMOTE font monter le rappel au seuil de 0,5 (de 0,38 à 0,95 pour la logistique), mais la PR-AUC ne bouge pas ;
  - ils **dégradent** la calibration : le Brier passe de 0,046 à 0,076 ;
  - **démonstration :** avec la pondération, l'ordonnée à l'origine est décalée de +2,05, proche de ln(w) = ln(10,4) = 2,34.
- **Ajouter le total :** la PR-AUC gagne +0,25 pour l'arbre, +0,15 pour le boosting, +0,14 pour la Random Forest, et environ 0 pour la logistique et le SVM.
- **Comparaison des modèles** (50 plis) :

| Modèle | PR-AUC | log-loss | Remarque |
|---|---|---|---|
| boosting | 0,795 ± 0,096 | | |
| Random Forest | 0,794 ± 0,107 | | |
| **boosting recalibré (retenu)** | 0,792 ± 0,099 | **0,114 ± 0,021** (meilleure) | |
| arbre | 0,772 | | |
| k-NN | 0,742 | | mémorise le train : 100 % pour un Pokémon connu |
| SVM | 0,718 | | |
| logistique | 0,697 | | |

  Le boosting bat le SVM, la logistique et le k-NN sur 74 à 82 % des plis appariés. Il est à égalité avec la Random Forest, mais environ 7 fois plus rapide.
- **Calibration :**
  - le boosting brut est **trop confiant** : il annonce environ 95 % pour des Pokémon dont seulement environ 73 % sont légendaires ;
  - la recalibration sigmoïde (Platt) corrige ce défaut.
- **Seuil :** 0,29 sur le train, 0,24 sur les 801 Pokémon (F1 maximal hors échantillon). Au seuil de 0,5, le F1 est de 0,70.
- **Test (une seule fois) :**
  - ROC-AUC 0,992, PR-AUC 0,935, F1 0,839, rappel 13/14, précision 0,77 ;
  - **les 4 faux positifs sont tous des méga-évolutions** : Alakazam, Gardevoir, Jungko, Lockpin ;
  - **le seul faux négatif est Cosmovum**, un cocon.
- **Interprétation :** les odds ratios de la logistique, par écart-type, sont : Vitesse ×3,4, Att. Spé. ×2,7, PV ×2,7, … Attaque ×1,3. L'Attaque compte peu, ce qui permet d'écarter les pseudo-légendaires.
- **Variante taille + poids :** PR-AUC de 0,795 à 0,850 (gain sur 70 % des plis). Elle n'est pas retenue, puisque l'entrée de l'appli reste les 6 stats, mais c'est une piste d'amélioration.

| Figure | Légende proposée |
|---|---|
| ★ `22_strategies_desequilibre.png` | Le rééquilibrage augmente le rappel au seuil de 0,5 sans changer la PR-AUC, et dégrade le Brier. |
| ★ `23_comparaison_cv_repetee.png` | Comparaison sur 50 plis : les modèles à base d'arbres, avec le total, sont en tête. |
| ★ `25_calibration.png` | Courbe de calibration : le boosting brut est trop confiant, la recalibration le corrige. |
| ★ `27_test_legendaire.png` | Test : matrice de confusion, courbes ROC et précision-rappel, comparées au total seul. |
| `24_interpretation_legendaire.png` | Odds ratios de la logistique et importance par permutation du boosting. |
| `26_choix_seuil.png` | Précision, rappel et F1 selon le seuil : maximum vers 0,25–0,3, et non à 0,5. |
| annexe : `21_sigmoide.png` | |

| Tableau | Fichier |
|---|---|
| Récapitulatif des modèles (moyenne ± écart-type) | `tables/recap_legendaire.csv` |
| Scores de test | `tables/recap_test.csv` |
| Erreurs du test (5 Pokémon) | notebook, section 3.11 (à recopier) |

---

## 5. Prédiction du type (≈ 1 page)

**Méthodologie**
- **Le type Vol est exclu**, faute d'exemples (3 Pokémon). Il reste 17 classes et 798 Pokémon, avec un split stratifié sur `type1`.
- **Softmax :** P(k) = e^η_k / Σ e^η_j, et l'entropie croisée comme fonction de coût.
- **Les métriques :** accuracy, F1 macro, F1 pondéré, **top-3** et log-loss.
  - **Piège à signaler :** le score `"top_k_accuracy"` de scikit-learn utilise k = 2 par défaut. Le top-3 a donc été calculé à la main.
- **Les références :** le hasard (top-3 de 18 %) et les **fréquences**, qui proposent toujours Eau, Normal et Plante (top-3 de 37 %).

**Résultats à citer**
- **SVM retenu (CV, 25 plis) :** accuracy 0,208, F1 macro 0,105, **top-3 0,50**, log-loss 2,43. La référence par fréquences donne 0,143 / 0,015 / 0,371 / 2,676.
- **Test :** accuracy 0,225, F1 macro 0,142, top-3 0,556.
- **Théorie de l'information :** le gain de log-loss, 0,245 nat sur une entropie de 2,676, signifie que **les stats ne lèvent qu'environ 9 % de l'incertitude sur le type**.
- **Pondération des classes :**
  - pour la logistique, `balanced` fait passer le F1 macro de 0,094 à 0,121, mais le top-3 baisse de 0,463 à 0,383 ;
  - pour le SVM calibré, l'effet est quasi nul, car la calibration l'annule.
- **Analyse par type :**
  - le rappel d'un type est très corrélé à la force de sa signature statistique (**ρ de Spearman = 0,86**, p ≈ 10⁻⁵) ;
  - bien reconnus : Fée (43 %), Acier (42 %), Combat (36 %), Normal (33 %), Électrik (32 %) ;
  - mal reconnus : Eau (1 %), Glace (0 %), Poison, Ténèbres, Plante.
- **Les confusions suivent les profils de combat :** Feu → Électrik, Psy → Fée, Acier → Roche, Sol → Combat.
- **Conclusion à écrire :** le type est un choix **thématique**, porté par l'apparence. Le problème est mal posé avec les stats, ce qui motive le niveau 2 (images).
- **Variante taille + poids :** le top-3 passe de 0,50 à 0,53.

| Figure | Légende proposée |
|---|---|
| ★ `28_types_comparaison.png` | Modèles contre la référence par fréquences : gain réel mais modeste (top-3 de 50 % contre 37 %). |
| ★ `29_types_reconnaissables.png` | Plus la signature statistique d'un type est forte, mieux il est reconnu (ρ = 0,86). |
| `30_matrice_confusion_types.png` | Matrice de confusion : confusions entre profils de combat voisins. |

| Tableau | Fichier |
|---|---|
| Récapitulatif des modèles | `tables/recap_type.csv` |

---

## 6. Choix finaux et application (≈ ½ page)

- **Tableau de justification**, « tâche → modèle retenu → pourquoi → alternatives écartées » : à recopier depuis le notebook, section 5.2.
- **Critère commun :** chaque modèle est choisi selon **ce que l'appli affiche** (un groupe interprétable, une probabilité calibrée, un top-3), et pas selon un seul score global.
- **Réentraînement final sur les 801 Pokémon** avec la configuration retenue, sans nouvelle sélection.
  - Les performances à annoncer restent celles de la validation croisée et du test.
  - Le K-means rechargé redonne 100 % des groupes à l'identique.
  - L'analyse d'un Pokémon prend environ 25 ms sur CPU.
- **Fichiers sauvegardés :** `scaler`, `kmeans`, `clusters.json`, `legendary_classifier`, `type_classifier`, `metadata.json`.

| Figure | Légende proposée |
|---|---|
| ★ `31_appli_statistiques.png` | Onglet Statistiques de l'appli (exemple : Carchacrok). Groupe, probabilité d'être légendaire, top-3 des types, radar comparé au groupe. |

---

## 7. Limites et pistes d'amélioration (≈ ½ page)

**Limites**
1. **Méga-évolutions dans les données.** Pour Florizarre, Gardevoir, Carchacrok… le CSV donne les stats de la forme méga. Ce sont **tous** les faux positifs du test.
2. **Peu de légendaires** (70, dont 14 dans le test). La variance des scores est forte (± 0,1 de PR-AUC), d'où la validation croisée répétée et les comparaisons appariées.
3. **Légendaires atypiques** (Cosmog, Cosmovum, Regirock). Leur statut vient de l'histoire du jeu, pas des stats.
4. **Le type est peu déterminé par les stats** (environ 9 % de l'incertitude levée). C'est une limite de l'**information disponible**, pas des modèles. Le type Vol a été exclu.
5. **Structure en groupes faible** (silhouette de 0,23). Les clusters résument un continuum, et les Pokémon proches des frontières changent facilement de groupe.
6. **Dérive dans le temps :** de 3 % à 21 % de légendaires selon la génération. Le modèle pourrait mal se comporter sur les générations 8 et 9.

**Pistes d'amélioration**
- Ajouter la taille et le poids en entrée optionnelle : +0,05 de PR-AUC pour le statut légendaire, +3 points de top-3 pour le type.
- Retirer les formes méga, ou les signaler par une variable dédiée.
- Utiliser un GMM pour détecter les profils atypiques (faible densité).
- Prédire le type à partir de l'image (niveau 2).

---

## Questions probables à l'oral (et réponses courtes)

| Question | Réponse |
|---|---|
| Pourquoi standardiser ? | K-means, k-NN et SVM utilisent des distances euclidiennes : sans standardisation, les stats les plus dispersées pèseraient davantage, sans raison. |
| Pourquoi pas k = 2 ? | Meilleure silhouette, mais la partition se réduit à « faibles contre forts », c'est-à-dire à CP1 : elle n'apprend rien de plus que le total. |
| Pourquoi la PR-AUC plutôt que la ROC-AUC ? | Avec 8,7 % de positifs, la PR-AUC se concentre sur la classe rare. Sa référence vaut 0,087, au lieu de 0,5 pour la ROC-AUC. |
| SMOTE ne sert donc à rien ? | Ici, il déplace le seuil sans mieux classer et il gonfle les probabilités. On obtient le même effet de façon contrôlée en choisissant le seuil explicitement. |
| Comment savoir qu'il n'y a pas de fuite ? | Les colonnes qui trichent sont exclues, le prétraitement est dans le `Pipeline`, le test a été évalué une seule fois, et le total est recalculé à partir des seules entrées. |
| Pourquoi recalibrer le boosting ? | Il était trop confiant au-dessus de 80 %. L'appli affiche une probabilité, qui doit avoir un sens fréquentiel. |
| Pourquoi le type est-il si mal prédit ? | Ce n'est pas un défaut des modèles : les stats ne contiennent qu'environ 9 % de l'information sur le type. |
