# Entraînement m0_1793355775110682900

Version : `m0/v1_logistic` ; identifiant : `m0`.

## Réglages de ce run

```json
{
  "model": "m0",
  "data": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\data\\wikiart_style5",
  "split": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\splits\\style5.json",
  "output": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\runs",
  "epochs": 1,
  "batch_size": 32,
  "lr": 0.001,
  "seed": 42,
  "threads": 8,
  "device": "cpu",
  "split_only": false,
  "classes": [
    "Baroque",
    "Cubism",
    "Impressionism",
    "Pop_Art",
    "Ukiyo_e"
  ],
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

Accuracy test : 37.78%.
Macro F1 test : 0.3723.

Voir history.json, test_metrics.json et test_predictions.csv pour les mesures détaillées.

## Fiche de version (copie au moment de la documentation)

# M0 — v1 : régression logistique

## Rôle et différences

Référence classique, sans couche cachée. Contrairement aux MLP et à ResNet,
la décision est une combinaison linéaire des pixels. Cette version utilise le
split partagé, contrairement aux anciens essais non comparables.

## Réglages

Images RGB 64 × 64, aplaties, pixels entre 0 et 1. Scikit-learn LogisticRegression,
maximum 1 000 itérations. C choisi parmi 0,01 / 0,1 / 1 sur la validation.
Pas de courbe par époque : `selection.json` décrit la sélection de C.

```bash
python -m model_versions.m0.v1_logistic.train
```

Résultat de référence : 37,78 % sur le test (run du groupe dans `docs/RESULTS.md`).
Les itérations du solveur ne sont pas des époques PyTorch ; le champ epochs est sans effet ici.

Documentation ajoutée rétrospectivement lors du rangement ; mesures originales inchangées.
