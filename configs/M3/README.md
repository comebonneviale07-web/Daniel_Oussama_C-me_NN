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
