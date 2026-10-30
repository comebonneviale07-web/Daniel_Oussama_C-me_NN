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
