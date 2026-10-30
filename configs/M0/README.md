# M0 ? R?gression logistique

## Version v1_logistic (`m0`)

R?f?rence classique sans couche cach?e : RGB 64 ? 64, pixels entre 0 et 1, aplatis.
C choisi parmi 0,01 / 0,1 / 1 sur validation ; solveur limit? ? 1 000 it?rations.
R?glages dans `v1_logistic.json`. Le mod?le sauvegard? est `best.joblib`.

```bash
python main.py train --model m0
```

R?sultat du groupe : 37,78 % test. Pas de courbe par ?poque, car ce n'est pas
un entra?nement PyTorch ; `selection.json` d?crit les scores de validation par C.

## Version v2_features (`m0_features`)

Même classifieur (régression logistique), entrées différentes. Au lieu des 12 288 pixels bruts, chaque peinture
est décrite par 190 nombres calculés dans `src/features.py` :

- histogramme de couleurs HSV (8 × 4 × 4 = 128) : quelles couleurs, où qu'elles soient ;
- disposition des couleurs (couleur moyenne de chaque case d'une grille 4 × 4 = 48) ;
- orientations des contours (Sobel, 8 orientations pondérées par l'intensité + densité de contours = 9) :
  facettes droites du cubisme, petites touches dispersées de l'impressionnisme ;
- texture (5) : luminosité moyenne et dispersion, intensité moyenne et dispersion des contours, part de
  contours forts.

Les features sont standardisées (`StandardScaler`, statistiques du train) et C est choisi sur le **macro F1 de
validation** dans une grille élargie de 0,0001 à 10. Motivation : un pixel isolé ne dit rien du style et un
décalage de quelques pixels change tout le vecteur ; de plus, le meilleur C de v1 (0,01) était au bord de sa
grille. Réglages dans `v2_features.json`.

```bash
python main.py train --model m0_features
```

Run `m0_features_1793372135499282000` : C = 1 choisi (macro F1 de validation 0,530) ; **50,67 % test**
(macro F1 0,4932) contre 37,78 % pour v1. Test de McNemar sur les mêmes 225 images
(`python main.py compare`) : 33 images justes seulement pour v1, 62 seulement pour v2, **p = 0,0038 :
amélioration significative**. Limites : features choisies à la main ; environ ± 6 points d'incertitude sur
225 images de test.
