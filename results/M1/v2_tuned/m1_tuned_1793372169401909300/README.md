# Entraînement m1_tuned_1793372169401909300

Version : `M1/v2_tuned` ; identifiant : `m1_tuned`.

## Réglages de ce run

```json
{
  "model": "m1_tuned",
  "data": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\data\\wikiart_style5",
  "split": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\data\\splits\\style5.json",
  "output": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\results\\M1\\v2_tuned",
  "epochs": 30,
  "batch_size": 32,
  "lr": 0.0001,
  "seed": 42,
  "threads": 8,
  "device": "cpu",
  "split_only": false,
  "model_version": "m1/v2_tuned",
  "version_settings": {
    "id": "m1_tuned",
    "version": "m1/v2_tuned",
    "kind": "mlp",
    "input_size": 64,
    "hidden_sizes": [
      256
    ],
    "activation": "relu",
    "dropout": 0.0,
    "normalization": "training_channel",
    "augmentation": "flip",
    "selection_metric": "val_macro_f1",
    "patience": 8,
    "epochs": 30,
    "lr": 0.0001
  },
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

Accuracy test : 37.33%.
Macro F1 test : 0.3645.
Checkpoint choisi sur le macro F1 de validation : époque 4.

Voir history.json, test_metrics.json et test_predictions.csv pour les mesures détaillées.

## Fiche de version (copie au moment de la documentation)

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
