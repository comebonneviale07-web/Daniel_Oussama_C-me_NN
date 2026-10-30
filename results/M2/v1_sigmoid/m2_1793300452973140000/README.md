# Entraînement m2_1793300452973140000

Version : `m2/v1_sigmoid` ; identifiant : `m2`.

## Réglages de ce run

```json
{
  "model": "m2",
  "data": "C:\\Users\\DANIEL\\Documents\\Daniel\\1. ALBERT SCHOOL-PSL\\2nd YEAR\\Projet Neural Networks\\Daniel_Oussama_C-me_NN\\data\\wikiart_style5",
  "split": "C:\\Users\\DANIEL\\Documents\\Daniel\\1. ALBERT SCHOOL-PSL\\2nd YEAR\\Projet Neural Networks\\Daniel_Oussama_C-me_NN\\splits\\style5.json",
  "output": "C:\\Users\\DANIEL\\Documents\\Daniel\\1. ALBERT SCHOOL-PSL\\2nd YEAR\\Projet Neural Networks\\Daniel_Oussama_C-me_NN\\runs",
  "epochs": 15,
  "batch_size": 32,
  "lr": 0.01,
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

Accuracy test : 20.00%.
Macro F1 test : 0.0667.

Voir history.json, test_metrics.json et test_predictions.csv pour les mesures détaillées.

## Fiche de version (copie au moment de la documentation)

# M2 — v1 : sigmoid, expérience d'échec conservée

## Différences avec M1

Cinq couches cachées : 512 → 256 → 128 → 64 → 32, avec sigmoid et dropout 0,3
après chaque couche. Adam lr 0,01, 15 époques ; RGB 64 × 64 entre 0 et 1.

## Pourquoi conserver cette version

Le run de référence prédit Cubisme pour les 225 images de test (20 % de bonnes
réponses). Les diagnostics du groupe montrent des sigmoïdes saturées et une
forte atténuation des gradients vers l'entrée. Voir `docs/DIAGNOSIS.md`.
Cette version reste reproductible pour expliquer l'échec, pas pour le déploiement.

```bash
python -m model_versions.m2.v1_sigmoid.train
```

La correction est dans le dossier voisin `v2_relu/` ; l'identifiant historique
`m2` continue de désigner cette v1 pour charger les anciens checkpoints correctement.

Documentation ajoutée rétrospectivement lors du rangement ; mesures originales inchangées.
