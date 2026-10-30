# Entraînement m3_1793355952561843000

Version : `m3/v2_finetuned` ; identifiant : `m3`.

## Réglages de ce run

```json
{
  "model": "m3",
  "data": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\data\\wikiart_style5",
  "split": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\splits\\style5.json",
  "output": "C:\\Users\\Oussama\\Daniel_Oussama_C-me_NN\\runs",
  "epochs": 10,
  "batch_size": 32,
  "lr": 0.0001,
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
  "input_size": 224
}
```

## Résultats

Accuracy test : 83.11%.
Macro F1 test : 0.8296.

Voir history.json, test_metrics.json et test_predictions.csv pour les mesures détaillées.

## Fiche de version (copie au moment de la documentation)

# M3 — v2 : ResNet-18 entièrement ajusté

## Différences avec v1_frozen

Même ResNet-18 préentraîné et même prétraitement ImageNet, mais tous les paramètres
et les statistiques de batch normalization sont ajustés. Adam lr 0,0001 au lieu de
0,001 ; 10 époques. Les convolutions s'adaptent aux peintures, avec un coût CPU accru.

```bash
python -m model_versions.m3.v2_finetuned.train
```

Résultat du groupe : 83,11 % sur le test, contre 75,11 % pour le backbone gelé,
sur le même split. Les durées sur différents nombres de threads ne se comparent
pas directement. Voir `docs/RESULTS.md` et les mesures de benchmark.

Identifiant historique conservé : `m3`. Les poids ne sont pas dans Git ; utiliser
les poids locaux ou un transfert séparé pour l'inférence.

Documentation ajoutée rétrospectivement lors du rangement ; mesures originales inchangées.
