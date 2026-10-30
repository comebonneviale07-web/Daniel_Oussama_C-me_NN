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
