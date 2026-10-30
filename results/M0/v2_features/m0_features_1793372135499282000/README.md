# Entraînement m0_features_1793372135499282000

Version : `M0/v2_features` ; identifiant : `m0_features`.

## Réglages de ce run

```json
{
  "model": "m0_features",
  "data": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\data\\wikiart_style5",
  "split": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\data\\splits\\style5.json",
  "output": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\results\\M0\\v2_features",
  "epochs": 1,
  "batch_size": 32,
  "lr": 0.001,
  "seed": 42,
  "threads": 8,
  "device": "cpu",
  "split_only": false,
  "model_version": "m0/v2_features",
  "version_settings": {
    "id": "m0_features",
    "version": "m0/v2_features",
    "kind": "logistic",
    "input_size": 64,
    "normalization": "unit",
    "features": "handcrafted",
    "standardize": true,
    "selection_metric": "val_macro_f1",
    "epochs": 1,
    "lr": 0.001,
    "c_candidates": [
      0.0001,
      0.0003,
      0.001,
      0.003,
      0.01,
      0.03,
      0.1,
      0.3,
      1.0,
      3.0,
      10.0
    ],
    "max_iter": 5000
  },
  "classes": [
    "Baroque",
    "Cubism",
    "Impressionism",
    "Pop_Art",
    "Ukiyo_e"
  ],
  "input_normalization": null,
  "activation": null,
  "dropout": null,
  "torch_version": "2.7.0+cpu",
  "numpy_version": "2.2.3",
  "python_version": "3.13.1",
  "package_versions": {
    "torch": "2.7.0",
    "torchvision": "0.22.0+cpu",
    "scikit-learn": "1.7.0",
    "numpy": "2.2.3",
    "pillow": "11.1.0",
    "matplotlib": "3.10.0"
  },
  "input_size": 64
}
```

## Résultats

Accuracy test : 50.67%.
Macro F1 test : 0.4932.

Voir history.json, test_metrics.json et test_predictions.csv pour les mesures détaillées.

## Fiche de version (copie au moment de la documentation)

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
