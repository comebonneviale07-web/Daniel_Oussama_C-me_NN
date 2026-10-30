# Entraînement m3_augmented_1793378930229182800

Version : `M3/v3_augmented` ; identifiant : `m3_augmented`.

## Réglages de ce run

```json
{
  "model": "m3_augmented",
  "data": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\data\\wikiart_style5",
  "split": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\data\\splits\\style5.json",
  "output": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\results\\M3\\v3_augmented",
  "epochs": 15,
  "batch_size": 32,
  "lr": 0.0001,
  "seed": 42,
  "threads": 8,
  "device": "cpu",
  "split_only": false,
  "model_version": "m3/v3_augmented",
  "version_settings": {
    "id": "m3_augmented",
    "version": "m3/v3_augmented",
    "kind": "resnet18",
    "input_size": 224,
    "normalization": "imagenet",
    "frozen": false,
    "augmentation": "paintings",
    "optimizer": "adamw",
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "label_smoothing": 0.1,
    "selection_metric": "val_macro_f1",
    "patience": 4,
    "epochs": 15,
    "lr": 0.0001
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
  "input_size": 224
}
```

## Résultats

Accuracy test : 79.11%.
Macro F1 test : 0.7885.
Checkpoint choisi sur le macro F1 de validation : époque 4.

Voir history.json, test_metrics.json et test_predictions.csv pour les mesures détaillées.

## Fiche de version (copie au moment de la documentation)

# M3 ? ResNet-18 : versions et diff?rences

Pr?entra?nement ImageNet, pr?traitement officiel produisant des images 224 ? 224.
Le redimensionnement de la source 128 px ne recr?e pas les d?tails perdus.

| R?glage | v1_frozen (`m3_frozen`) | v2_finetuned (`m3`) |
| --- | --- | --- |
| Param?tres entra?n?s | couche finale seulement | toutes les couches |
| Backbone et batch normalization | gel?s | ajust?s |
| Adam learning rate | 0,001 | 0,0001 |
| ?poques | 10 | 10 |
| Test du groupe | 75,11 % | 83,11 % |

La v2 adapte les caract?ristiques aux peintures avec un co?t CPU plus ?lev?.
Configurations s?par?es dans `v1_frozen.json` et `v2_finetuned.json`.
Les poids r?cents restent locaux ; les rapports seuls ne permettent pas de pr?dire.

```bash
python main.py train --model m3_frozen
python main.py train --model m3
```

Comparer les dur?es avec prudence : le nombre de threads diff?rait entre certains runs.
