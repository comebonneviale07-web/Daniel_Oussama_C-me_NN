# M1 ? MLP simple

## Version v1_baseline (`m1`)

Par rapport ? M0 : ajoute une couche cach?e ReLU de 256 neurones. RGB 64 ? 64
aplati, pixels entre 0 et 1, Adam lr 0,001, 3 ?poques, sans dropout.
R?glages dans `v1_baseline.json`. Checkpoint choisi par perte de validation.

```bash
python main.py train --model m1
```

R?sultat de r?f?rence : 30,22 % test. Les diagnostics du groupe ?tudient ensuite
normalisation, learning rate et augmentation ; voir `docs/DIAGNOSIS.md`.

## Version v2_tuned (`m1_tuned`)

Même architecture que v1 (une couche ReLU de 256 neurones, 64 px). Changements tirés des ablations de
validation de `docs/DIAGNOSIS.md` : pixels standardisés par canal (statistiques du train uniquement),
Adam lr 0,0001 au lieu de 0,001, augmentation par miroir gauche-droite (train seulement), jusqu'à 30 époques
avec checkpoint choisi sur le **macro F1 de validation** et arrêt après 8 époques sans amélioration.
Réglages dans `v2_tuned.json`.

```bash
python main.py train --model m1_tuned
```

Run `m1_tuned_1793372169401909300` : arrêt à l'époque 12, checkpoint de l'époque 4 (validation : accuracy
0,440, macro F1 0,438, contre 0,316 d'accuracy pour v1). **37,33 % test** (macro F1 0,3645) contre 30,22 %.
McNemar : 29 contre 45 images, **p = 0,081 : gain probable mais non significatif à 5 %** sur 225 images.
Limite : un MLP sur pixels aplatis ne voit pas les formes ; voir M0 v2 et M3 pour aller plus loin.
