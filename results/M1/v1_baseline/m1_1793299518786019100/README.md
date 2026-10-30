# Entraînement m1_1793299518786019100

Version : `m1/v1_baseline` ; identifiant : `m1`.

## Réglages de ce run

```json
{
  "model": "m1",
  "data": "C:\\Users\\DANIEL\\Documents\\Daniel\\1. ALBERT SCHOOL-PSL\\2nd YEAR\\Projet Neural Networks\\Daniel_Oussama_C-me_NN\\data\\wikiart_style5",
  "split": "C:\\Users\\DANIEL\\Documents\\Daniel\\1. ALBERT SCHOOL-PSL\\2nd YEAR\\Projet Neural Networks\\Daniel_Oussama_C-me_NN\\splits\\style5.json",
  "output": "C:\\Users\\DANIEL\\Documents\\Daniel\\1. ALBERT SCHOOL-PSL\\2nd YEAR\\Projet Neural Networks\\Daniel_Oussama_C-me_NN\\runs",
  "epochs": 3,
  "batch_size": 32,
  "lr": 0.001,
  "seed": 42,
  "threads": 2,
  "device": "cpu",
  "split_only": false,
  "classes": [
    "Baroque",
    "Cubism",
    "Impressionism",
    "Pop_Art",
    "Ukiyo_e"
  ],
  "torch_version": "2.15.0+cpu",
  "numpy_version": "2.5.4",
  "python_version": "3.13.7",
  "package_versions": {
    "torch": "2.15.0",
    "torchvision": "0.29.2",
    "scikit-learn": "1.9.1",
    "numpy": "2.5.4",
    "pillow": "13.0.0",
    "matplotlib": "3.11.2"
  },
  "input_size": 64
}
```

## Résultats

Accuracy test : 30.22%.
Macro F1 test : 0.2565.

Voir history.json, test_metrics.json et test_predictions.csv pour les mesures détaillées.

## Fiche de version (copie au moment de la documentation)

# M1 — v1 : MLP simple

## Rôle et différences

Ajoute une couche cachée non linéaire à la référence classique M0 :
12 288 pixels → 256 neurones ReLU → 5 styles. Version de référence avant les
variantes de normalisation et d'augmentation étudiées dans `docs/DIAGNOSIS.md`.

## Réglages

RGB 64 × 64, pixels entre 0 et 1 ; Adam, lr 0,001, 3 époques, sans dropout.
Checkpoint retenu sur la perte de validation, split partagé 70/15/15.

```bash
python -m model_versions.m1.v1_baseline.train
```

Résultat de référence : 30,22 % sur le test. Un MLP aplati ne modélise pas
explicitement les motifs spatiaux comme les convolutions de ResNet.

Documentation ajoutée rétrospectivement lors du rangement ; mesures originales inchangées.
