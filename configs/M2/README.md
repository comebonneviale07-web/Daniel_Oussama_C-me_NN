# M2 ? MLP profond : versions et diff?rences

Cinq couches cach?es identiques : 512 ? 256 ? 128 ? 64 ? 32 ; entr?e RGB 64 ? 64.

| R?glage | v1_sigmoid (`m2`) | v2_relu (`m2_fixed`) |
| --- | --- | --- |
| Activation | Sigmoid | ReLU |
| Adam learning rate | 0,01 | 0,0001 |
| Dropout | 0,3 | aucun |
| Normalisation | pixels 0?1 | moyenne/?cart-type par canal sur le train uniquement |
| ?poques | 15 | 20 |
| Test de r?f?rence | 20 % | 41,78 % |

## v1_sigmoid : ?chec conserv?

Pr?dit Cubisme pour les 225 images test. Les sigmo?des saturent et les gradients
des premi?res couches deviennent presque nuls. Cette version reste reproductible
pour expliquer l'?chec. Configuration : `v1_sigmoid.json`.

## v2_relu : correction

Le learning rate r?duit limite les mises ? jour excessives ; ReLU retire la
saturation des sigmo?des. Le dropout est retir? des couches ?troites et les pixels
sont standardis?s. Les choix viennent des ablations de validation du groupe, pas
d'une s?lection sur le test. Configuration : `v2_relu.json`.

Run de r?f?rence `m2_fixed_1793368290720728000` : 41,78 % test, macro F1 0,4060,
cinq styles pr?dits ; checkpoint ?poque 6 par perte de validation. Surapprentissage
ensuite (97,33 % train ? l'?poque 20). Les statistiques sont sauvegard?es avec les poids.

```bash
python main.py train --model m2        # reproduire l'?chec
python main.py train --model m2_fixed  # version corrig?e
```

## Version v3_regularised (`m2_regularised`)

Part de v2_relu (mêmes largeurs, ReLU, sans dropout, entrées standardisées, lr 0,0001) et vise son
surapprentissage (97 % au train) : BatchNorm1d après chaque couche cachée, AdamW avec weight decay 0,01,
augmentation par miroir (train seulement), checkpoint choisi sur le **macro F1 de validation**, arrêt après
8 époques sans amélioration (30 au maximum). Réglages dans `v3_regularised.json`.

```bash
python main.py train --model m2_regularised
```

Run `m2_regularised_1793377157734286200` : arrêt à l'époque 16, checkpoint de l'époque 8. En **validation**
il paraît meilleur (macro F1 0,461 contre 0,379 pour v2 à son époque retenue), mais en **test** il obtient
36,44 % (macro F1 0,3579) contre 41,78 % pour v2. McNemar : 37 contre 25 images, **p = 0,16 : aucune
différence démontrée**. Conclusion : la régularisation ne fournit pas de preuve d'amélioration ; v2_relu reste
la référence de M2. Ce résultat montre aussi qu'un écart de quelques points en validation (225 images) peut
être du bruit. Le test n'a servi à aucun choix.
