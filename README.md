# Museum AI Guide

Classify WikiArt paintings into Baroque, Cubism, Impressionism, Pop Art and Ukiyo-e.
The supplied dataset contains 300 images per class (1,500 total), cropped to 128 pixels.

## Corrected experiments

Run commands from this repository using Python 3.13:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python experiments.py --split-only
.venv\Scripts\python train_m0.py
.venv\Scripts\python train_m1.py
.venv\Scripts\python train_m2.py
.venv\Scripts\python train_m3.py
.venv\Scripts\python experiments.py --model m3_frozen
.venv\Scripts\python -m unittest -v test_experiments
```

CPU is the default. Full ResNet fine-tuning can take substantial time. Use `--epochs 1`
for an initial smoke run, and `--device cuda` only when CUDA is available.
If the Python launcher is unavailable, create the environment with
`& "$env:LocalAppData\Programs\Python\Python313\python.exe" -m venv .venv` in PowerShell.
CPU threads default to two and can be changed with `--threads`.

In Git Bash, activate the same environment with `source .venv/Scripts/activate`,
then use `python` for the commands above (for example, `python train_m1.py`).
Check `python --version` reports Python 3.13 before installing or training.
Pretrained ResNet weights require a first-time download.
`requirements-lock.txt` records the exact packages installed in the Windows Python 3.13
environment. Use `python -m pip install -r requirements-lock.txt` to reproduce those versions.

All models use the same saved, stratified 70/15/15 split in `splits/style5.json`
(1,050 training, 225 validation, 225 test images). It records relative image paths,
class names, a path fingerprint and the split seed. Reusing it is mandatory for comparison;
changing the training seed does not regenerate an existing split. The fingerprint checks
the image inventory, not image contents. This is an image-level split: artists may occur
in multiple partitions, so it does not measure generalization to unseen artists.

| Model | Approach | Input |
| --- | --- | --- |
| M0 | scikit-learn logistic regression; C selected using validation accuracy | 64 x 64 RGB, flattened, values in [0,1] |
| M1 | 256-unit ReLU MLP | 64 x 64 RGB, values in [0,1] |
| M2 | Five sigmoid layers, dropout 0.3; original learning rate retained for diagnosis | 64 x 64 RGB, values in [0,1] |
| M3 | ImageNet ResNet-18; all layers fine-tuned | pretrained weights' resize/crop/normalization |
| M3 frozen | ResNet backbone and batch-normalization statistics frozen; classifier trained | same as M3 |

M1/M2 now use 64-pixel inputs to reduce CPU cost. ResNet preprocessing produces
224-pixel inputs; upscaling the supplied 128-pixel images cannot restore lost detail.
M2 retains sigmoid deliberately: stagnant loss is a hypothesis to investigate, not
proof of vanishing gradients. `gradients.json` records mean per-batch weight-gradient
L2 norms for every layer each epoch. Layer widths affect these norms too.

Each execution creates a unique `runs/<model>_<timestamp>/` directory containing:

- `config.json` and the exact `split.json` used.
- Neural models: `history.json`, `curves.png`, and `best.pt`, selected by validation loss.
- M2: `gradients.json` and `gradients.png`; M0: `selection.json` and `best.joblib`.
- `test_metrics.json`: accuracy, macro F1, confusion matrix (rows=true, columns=predicted),
  and total experiment duration. Duration is not inference latency.
- `test_predictions.csv`: individual test paths and predictions for error analysis.
- `confusion_matrix.png`: a visual version of the test confusion matrix.

The test set is evaluated after model selection. Do not tune settings against test scores.
Accuracy values are fractions, not percentages. Saved neural checkpoints include class
order and architecture identifier. Do not load untrusted pickle/joblib model files.

## Legacy artifacts

The existing `models/*.pth` and sigmoid curve come from the original training scripts.
They are retained for reference and are incompatible with the new MLP input dimensions.
M1 originally trained on every image, so those weights must not be evaluated as an
independent held-out baseline. Original M2/M3 splits were not saved. Retrain before
reporting a fair comparison. No new performance claims are made by these code changes.

## Predicting new paintings

`predict.py` is the guide's recognition step: it loads a trained run, applies that model's own preprocessing to
any image (a photo of any size works) and returns the most likely styles with their probability.

```powershell
python predict.py photo.jpg                                    # best run whose weights are on this computer
python predict.py a.jpg b.png --run runs/<run folder> --top 3 --min-confidence .5 --json
```

Without `--run` it picks the latest run whose weights (`best.pt` / `best.joblib`, not in Git) and `config.json`
are present, preferring M3, then M3 frozen, M1, M0 and M2. With `--min-confidence`, a top probability below the
threshold is answered `uncertain`, so a kiosk can say "I am not sure" rather than give a wrong style. Example with
the fine-tuned M3 on one test painting per style: Ukiyo_e 99 %, Impressionism 97 %, Pop_Art 88 %, Cubism 65 %,
Baroque 54 %, all correct. Only load run files you produced: `best.joblib` is a pickle and can execute code.

## Dataset reproduction

```powershell
python make_wikiart_subset.py --label style --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art --per-class 300 --size 128 --out data/wikiart_style5
```

See `data/wikiart_style5/subset_info.json` for the pinned source revision and settings.
The dataset builder documents image rights and coursework usage restrictions.

## Remaining delivery work

Run and inspect corrected experiments, investigate M2 gradients, benchmark CPU inference
latency/RAM/model size, and write a measured comparison. Frozen ResNet feature caching,
inference for uploaded images, visual/text retrieval, and a Streamlit guide remain future
work. No museum interface or deployment benchmark is implemented yet.

## Saving completed work

See [RESULTS.md](RESULTS.md) for measured results and run completion status.
Small reports in `runs/` are tracked by Git; checkpoint files remain excluded.
Run `python backup_progress.py` after training to create and verify a local ZIP with
source files, settings, splits, reports and trained run weights. Copy the resulting ZIP
and checksum from `backups/` to independent storage. Neither `.venv/` nor dataset images
are included in that ZIP; dependency versions are in `requirements-lock.txt`.
