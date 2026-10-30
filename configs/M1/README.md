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
