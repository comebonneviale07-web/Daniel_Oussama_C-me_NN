# M3 ? ResNet-18 : versions et diff?rences

Pr?entra?nement ImageNet, pr?traitement officiel produisant des images 224 ? 224.
Le redimensionnement de la source 128 px ne recr?e pas les d?tails perdus.

| R?glage | v1_frozen (`m3_frozen`) | v2_finetuned (`m3`) |
| --- | --- | --- |
| Param?tres entra?n?s | couche finale seulement | toutes les couches |
| Backbone et batch normalization | gel?s | ajust?s |
| Adam learning rate | 0,001 | 0,0001 |
| ?poques | 10 | 10 |
| Test du groupe | 75,11 % | 83,11 % |

La v2 adapte les caract?ristiques aux peintures avec un co?t CPU plus ?lev?.
Configurations s?par?es dans `v1_frozen.json` et `v2_finetuned.json`.
Les poids r?cents restent locaux ; les rapports seuls ne permettent pas de pr?dire.

```bash
python main.py train --model m3_frozen
python main.py train --model m3
```

Comparer les dur?es avec prudence : le nombre de threads diff?rait entre certains runs.

## Versions v3_augmented (`m3_augmented`) et v4_frozen_augmented (`m3_frozen_augmented`)

Mêmes réseaux que v2 (fine-tuné) et v1 (gelé, batch normalization comprise). Changements d'entraînement
seulement : augmentation des images **du train** qui imite une photo de visiteur (recadrage de 60 à 100 %,
miroir gauche-droite, rotation de ± 10°, perspective oblique, luminosité et contraste ± 20 %, saturation
± 10 %, sans changement de teinte car la couleur est un indice de style, ni retournement vertical), AdamW
(weight decay 0,0001), scheduler cosinus, label smoothing 0,1, checkpoint choisi sur le macro F1 de
validation, arrêt après 4 époques sans amélioration (15 au maximum). Validation et test restent sans
augmentation. Réglages dans `v3_augmented.json` et `v4_frozen_augmented.json`.

```bash
python main.py train --model m3_augmented
python main.py train --model m3_frozen_augmented
```

**Sur les images de test propres (scans WikiArt), pas d'amélioration :**

| Version | Test | Référence | McNemar |
| --- | ---: | ---: | --- |
| v4_frozen_augmented, run `m3_frozen_augmented_1793377409306562700` | 70,67 % | v1 : 75,11 % | p = 0,053, non significatif |
| v3_augmented, run `m3_augmented_1793378930229182800` | 79,11 % | v2 : 83,11 % | p = 0,12, non significatif |

v3 s'est arrêtée à l'époque 8 (checkpoint de l'époque 4, validation 82,7 % contre 82,2 % pour v2) ; v4 a fait
ses 15 époques et reste à 77 % au train : avec une seule couche entraînable, des images augmentées sont plus
difficiles à ajuster.

**Sur des photos de visiteur simulées, v3 résiste mieux.** `python main.py robustness` dégrade les 225 images de
test de façon reproductible (inclinaison de 10°, vue oblique, cadrage serré, lumière faible, flou, compression
JPEG, puis tout à la fois) ; run `results/robustness/robustness_1793381065190766500` :

| Condition | v2_finetuned | v3_augmented | McNemar |
| --- | ---: | ---: | --- |
| propre | 83,1 % | 79,1 % | p = 0,12 |
| inclinaison 10° | 64,0 % | 68,9 % | p = 0,12 |
| vue oblique | 78,2 % | 76,9 % | |
| cadrage serré | 72,0 % | 72,0 % | |
| lumière faible | 76,9 % | 76,4 % | |
| flou | 40,9 % | 44,4 % | p = 0,20 |
| compression JPEG | 74,7 % | 71,1 % | |
| **toutes les dégradations** | **30,7 %** | **38,7 %** | **p = 0,001, significatif** |

Conclusion : l'augmentation échange environ 4 points (non significatifs) sur des scans propres contre une
meilleure résistance, significative, quand les défauts d'une vraie photo se cumulent, ce qui correspond à
l'usage sur borne ou téléphone. Le flou reste le point faible des deux versions (41-44 %) : les sources ne
font que 128 px, donc le moindre flou efface les détails. Limites : dégradations simulées et non de vraies
photos de galerie ; un seul entraînement par version.
