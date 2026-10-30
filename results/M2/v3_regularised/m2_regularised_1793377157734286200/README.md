# Entraînement m2_regularised_1793377157734286200

Version : `M2/v3_regularised` ; identifiant : `m2_regularised`.

## Réglages de ce run

```json
{
  "model": "m2_regularised",
  "data": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\data\\wikiart_style5",
  "split": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\data\\splits\\style5.json",
  "output": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\results\\M2\\v3_regularised",
  "epochs": 30,
  "batch_size": 32,
  "lr": 0.0001,
  "seed": 42,
  "threads": 8,
  "device": "cpu",
  "split_only": false,
  "model_version": "m2/v3_regularised",
  "version_settings": {
    "id": "m2_regularised",
    "version": "m2/v3_regularised",
    "kind": "mlp",
    "input_size": 64,
    "hidden_sizes": [
      512,
      256,
      128,
      64,
      32
    ],
    "activation": "relu",
    "dropout": 0.0,
    "batch_norm": true,
    "normalization": "training_channel",
    "augmentation": "flip",
    "optimizer": "adamw",
    "weight_decay": 0.01,
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

Accuracy test : 36.44%.
Macro F1 test : 0.3579.
Checkpoint choisi sur le macro F1 de validation : époque 8.

Voir history.json, test_metrics.json et test_predictions.csv pour les mesures détaillées.

## Fiche de version (copie au moment de la documentation)

# M2 ? MLP profond : versions et diff?rences

Cinq couches cach?es identiques : 512 ? 256 ? 128 ? 64 ? 32 ; entr?e RGB 64 ? 64.

| R?glage | v1_sigmoid (`m2`) | v2_relu (`m2_fixed`) |
| --- | --- | --- |
| Activation | Sigmoid | ReLU |
| Adam learning rate | 0,01 | 0,0001 |
| Dropout | 0,3 | aucun |
| Normalisation | pixels 0?1 | moyenne/?cart-type par canal sur le train uniquement |
| ?poques | 15 | 20 |
| Test de r?f?rence | 20 % | 41,78 % |

## v1_sigmoid : ?chec conserv?

Pr?dit Cubisme pour les 225 images test. Les sigmo?des saturent et les gradients
des premi?res couches deviennent presque nuls. Cette version reste reproductible
pour expliquer l'?chec. Configuration : `v1_sigmoid.json`.

## v2_relu : correction

Le learning rate r?duit limite les mises ? jour excessives ; ReLU retire la
saturation des sigmo?des. Le dropout est retir? des couches ?troites et les pixels
sont standardis?s. Les choix viennent des ablations de validation du groupe, pas
d'une s?lection sur le test. Configuration : `v2_relu.json`.

Run de r?f?rence `m2_fixed_1793368290720728000` : 41,78 % test, macro F1 0,4060,
cinq styles pr?dits ; checkpoint ?poque 6 par perte de validation. Surapprentissage
ensuite (97,33 % train ? l'?poque 20). Les statistiques sont sauvegard?es avec les poids.

```bash
python main.py train --model m2        # reproduire l'?chec
python main.py train --model m2_fixed  # version corrig?e
```
