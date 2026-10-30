# Entraînement m2_fixed_1793368666087080100

Version : `m2/v2_relu` ; identifiant : `m2_fixed`.

## Réglages de ce run

```json
{
  "model": "m2_fixed",
  "data": "C:\\Users\\DANIEL\\Documents\\Daniel\\1. ALBERT SCHOOL-PSL\\2nd YEAR\\Projet Neural Networks\\Daniel_Oussama_C-me_NN\\data\\wikiart_style5",
  "split": "C:\\Users\\DANIEL\\Documents\\Daniel\\1. ALBERT SCHOOL-PSL\\2nd YEAR\\Projet Neural Networks\\Daniel_Oussama_C-me_NN\\splits\\style5.json",
  "output": "C:\\Users\\DANIEL\\Documents\\Daniel\\1. ALBERT SCHOOL-PSL\\2nd YEAR\\Projet Neural Networks\\Daniel_Oussama_C-me_NN\\runs",
  "epochs": 20,
  "batch_size": 32,
  "lr": 0.0001,
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
  "input_normalization": {
    "mean": [
      0.5471431336850863,
      0.499235984659303,
      0.43746695592635326
    ],
    "std": [
      0.2632974281129757,
      0.25445071610516967,
      0.2512731548001593
    ],
    "source": "training split only"
  },
  "activation": "relu",
  "dropout": 0.0,
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

Accuracy test : 41.78%.
Macro F1 test : 0.4060.

Voir history.json, test_metrics.json et test_predictions.csv pour les mesures détaillées.

## Fiche de version (copie au moment de la documentation)

# M2 — v2 : correction ReLU

## Différences avec v1_sigmoid

| Réglage | v1 | v2 |
| --- | --- | --- |
| Couches cachées | 512, 256, 128, 64, 32 | identiques |
| Activation | sigmoid | ReLU |
| Adam learning rate | 0,01 | 0,0001 |
| Dropout | 0,3 | aucun |
| Pixels | entre 0 et 1 | standardisation par canal sur le train uniquement |
| Époques par défaut | 15 | 20 |

Le petit learning rate évite les premières mises à jour excessives. ReLU retire
la saturation des sigmoïdes. Le retrait du dropout et la standardisation viennent
des ablations de validation du groupe ; cette comparaison évalue l'ensemble des
corrections, pas l'effet isolé de chacune.

## Utilisation et résultats

```bash
python -m model_versions.m2.v2_relu.train
# Équivalent : python experiments.py --model m2_fixed
```

Run de référence `m2_fixed_1793368290720728000` : 41,78 % sur le test, macro F1
0,4060, cinq styles prédits. Checkpoint choisi à l'époque 6 par perte de validation.
Surapprentissage ensuite : 97,33 % sur le train à l'époque 20, validation dégradée.
Les statistiques de normalisation sont dans le checkpoint et réutilisées à l'inférence.

Identifiant stable : `m2_fixed`. L'ancienne v1 reste disponible pour comparaison.

Documentation ajoutée rétrospectivement lors du rangement ; mesures originales inchangées.
