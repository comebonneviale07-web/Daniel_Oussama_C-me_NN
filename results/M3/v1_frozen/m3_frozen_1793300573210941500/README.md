# Entraînement m3_frozen_1793300573210941500

Version : `m3/v1_frozen` ; identifiant : `m3_frozen`.

## Réglages de ce run

```json
{
  "model": "m3_frozen",
  "data": "C:\\Users\\DANIEL\\Documents\\Daniel\\1. ALBERT SCHOOL-PSL\\2nd YEAR\\Projet Neural Networks\\Daniel_Oussama_C-me_NN\\data\\wikiart_style5",
  "split": "C:\\Users\\DANIEL\\Documents\\Daniel\\1. ALBERT SCHOOL-PSL\\2nd YEAR\\Projet Neural Networks\\Daniel_Oussama_C-me_NN\\splits\\style5.json",
  "output": "C:\\Users\\DANIEL\\Documents\\Daniel\\1. ALBERT SCHOOL-PSL\\2nd YEAR\\Projet Neural Networks\\Daniel_Oussama_C-me_NN\\runs",
  "epochs": 10,
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
  "input_size": 224
}
```

## Résultats

Accuracy test : 75.11%.
Macro F1 test : 0.7485.

Voir history.json, test_metrics.json et test_predictions.csv pour les mesures détaillées.

## Fiche de version (copie au moment de la documentation)

# M3 — v1 : ResNet-18 gelé

## Différences avec les MLP

Utilise les caractéristiques visuelles d'un ResNet-18 préentraîné sur ImageNet.
Le backbone et les statistiques de batch normalization restent gelés ; seule
la couche finale apprend les cinq styles.

## Réglages et résultats

Prétraitement officiel ImageNet, sortie 224 × 224, Adam lr 0,001, 10 époques.
Le redimensionnement des images source 128 px ne recrée pas les détails perdus.
Résultat de référence : 75,11 % sur le test ; coût CPU inférieur au fine-tuning.

```bash
python -m model_versions.m3.v1_frozen.train
```

Identifiant stable : `m3_frozen`. La version suivante entraîne tout le backbone.

Documentation ajoutée rétrospectivement lors du rangement ; mesures originales inchangées.
