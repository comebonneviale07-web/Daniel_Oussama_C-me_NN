# Saved experiment progress

These runs share the saved stratified split: 1,050 training, 225 validation and
225 test paintings. Each class has 45 test paintings. Checkpoints were selected
using validation loss; these are test results, not validation results.

| Model | Run folder under `runs/` | Test accuracy | Macro F1 | Total experiment time |
| --- | --- | ---: | ---: | ---: |
| M0 | `m0_1793355775110682900` | 37.78% | 0.3722 | 140 seconds (C = 0.01 chosen on validation) |
| M1 | `m1_1793299518786019100` | 30.22% | 0.2565 | 35.3 seconds |
| M2 | `m2_1793300452973140000` | 20.00% | 0.0667 | 92.6 seconds |
| M2 corrected (ReLU) | `m2_fixed_1793368290720728000` | 41.78% | 0.4060 | 74.1 seconds |
| Frozen ResNet | `m3_frozen_1793300573210941500` | 75.11% | 0.7485 | 961.8 seconds |
| Fine-tuned ResNet (M3) | `m3_1793355952561843000` | 83.11% | 0.8304 | 2,375 seconds (8 threads) |

Times include experiment overhead and are not inference latency benchmarks.

## All versions, 30 October (benchmark `results/benchmarks/benchmark_1793383165870502500`)

Same 225 test paintings for every run, 95 % bootstrap intervals, one painting at a time on one CPU thread.
The latest run of each ID is used (M0 v1: the group's re-run, 38.2 %; the McNemar comparison below used the
37.8 % run). \* Measured on battery power: absolute times are about 1.7 times those of the mains-powered
benchmark `benchmark_1793358411979475300` (M3: 259 ms vs 151 ms); the ratios between models are comparable.
Run `python main.py benchmark` on mains power for absolute kiosk figures.

| Model | ID | Test accuracy [95 % CI] | Macro F1 | CPU time / painting * | Size | Run |
| --- | --- | --- | ---: | ---: | ---: | --- |
| M2 MLP, 5 sigmoid layers | `m2` | 20.0 % [15.1, 25.3] | 0.067 | 9.0 ms | 25.9 MB | `m2_1793300452973140000` |
| M1 MLP, 1 ReLU layer | `m1` | 30.2 % [24.4, 36.4] | 0.257 | 4.1 ms | 12.6 MB | `m1_1793299518786019100` |
| M2 regularised (BatchNorm, flip) | `m2_regularised` | 36.4 % [30.2, 42.7] | 0.358 | 8.2 ms | 25.9 MB | `m2_regularised_1793377157734286200` |
| M1 tuned (standardised, flip) | `m1_tuned` | 37.3 % [31.1, 43.6] | 0.365 | 7.5 ms | 12.6 MB | `m1_tuned_1793372169401909300` |
| M0 logistic regression, pixels | `m0` | 38.2 % [32.4, 44.9] | 0.376 | 0.9 ms | 0.5 MB | `m0_1793367969029599200` |
| M2 corrected, 5 ReLU layers | `m2_fixed` | 41.8 % [35.5, 48.0] | 0.406 | 9.1 ms | 25.9 MB | `m2_fixed_1793368666087080100` |
| **M0 logistic regression, hand-crafted features** | `m0_features` | **50.7 % [44.4, 57.3]** | 0.493 | 6.6 ms | 0.01 MB | `m0_features_1793372135499282000` |
| M3 ResNet-18 frozen + augmentation | `m3_frozen_augmented` | 70.7 % [64.4, 76.4] | 0.702 | 261 ms | 44.8 MB | `m3_frozen_augmented_1793377409306562700` |
| M3 ResNet-18, frozen | `m3_frozen` | 75.1 % [69.8, 80.9] | 0.749 | 260 ms | 44.8 MB | `m3_frozen_1793300573210941500` |
| **M3 ResNet-18 fine-tuned + augmentation** | `m3_augmented` | 79.1 % [73.8, 84.4] | 0.789 | 240 ms | 44.8 MB | `m3_augmented_1793378930229182800` |
| **M3 ResNet-18, fine-tuned** | `m3` | **83.1 % [77.8, 88.0]** | 0.830 | 259 ms | 44.8 MB | `m3_1793355952561843000` |

Paired exact McNemar tests (`python main.py compare`, files in `results/comparisons/`):

| New version vs reference | Test accuracy | p-value | Conclusion |
| --- | --- | ---: | --- |
| `m0_features` vs `m0` | 37.8 % -> 50.7 % | 0.004 | significant improvement |
| `m1_tuned` vs `m1` | 30.2 % -> 37.3 % | 0.081 | likely, not significant |
| `m2_regularised` vs `m2_fixed` | 41.8 % -> 36.4 % | 0.16 | no evidence; `m2_fixed` stays the M2 reference |
| `m3_frozen_augmented` vs `m3_frozen` | 75.1 % -> 70.7 % | 0.053 | not significant |
| `m3_augmented` vs `m3` | 83.1 % -> 79.1 % | 0.12 | not significant |
| `m3` vs `m3_frozen` | 75.1 % -> 83.1 % | 0.006 | fine-tuning gain significant |

Robustness to simulated visitor photos (`python main.py robustness`, run `results/robustness/robustness_1793381065190766500`):
with tilt, oblique view, close framing, dim light, blur and JPEG compression all together, `m3_augmented`
keeps 38.7 % against 30.7 % for `m3` (p = 0.001, significant); each degradation alone changes little, and blur
alone drops both to 41-44 %. Details in `configs/M3/README.md`.

**Recommendation.** A pretrained, fine-tuned ResNet-18 remains the only model that recognises most paintings
(79-83 % against at most 51 % without pretrained convolutional features), for about 0.15-0.26 s per painting on
one CPU thread and 45 MB. For the kiosk and phones, where pictures are real photos, deploy `m3_augmented`: on
clean scans it is not significantly below `m3`, and it is significantly more robust when the defects of a
photo add up. Without a neural network, the best option is `m0_features` (50.7 %, 0.01 MB), far behind.
Limits: one training per version, 225 test images (+-6 points), simulated rather than real gallery photos,
image-level split (89 % of the test paintings by "Unknown Artist").
The run folders contain settings, split snapshots, training histories, curves,
confusion matrices and individual test predictions. M2 also contains gradient logs
and plots. M2 predicted Cubism for every test image; stagnant performance alone
does not establish vanishing gradients as the cause.

M0's earlier folder `m0_1793300405327349300` is an incomplete run (configuration and split
only); it is superseded by `m0_1793355775110682900`. M0 and M3 were run with
`experiments.py` on the same shared split. Confidence intervals, CPU latency and model
sizes are in the README section "CPU benchmark and recommendation to the director".

## M2 repair

`m2_fixed` retains hidden widths 512-256-128-64-32 but uses ReLU instead of sigmoid,
no dropout, Adam learning rate 0.0001 instead of 0.01, and channel standardization
computed on the 1,050 training images only. These settings were taken from the
group's existing validation ablations in `DIAGNOSIS.md`, not chosen from test scores.
Run it with `python main.py train --model m2_fixed`; `train_m2.py` remains the original failure baseline.

Training ran for 20 epochs. The checkpoint selected by minimum validation loss was
epoch 6 (39.11% validation accuracy, loss 1.4411). After selection, its single test
evaluation gave 94/225 correct predictions (41.78%, macro F1 0.4060). It predicts all
five test classes: Baroque 58, Cubism 21, Impressionism 53, Pop Art 46, Ukiyo-e 47.
The original M2 predicted Cubism for all 225 test images.

Gradient L2 norms now reach the input layers: first/output weight norm ratio was
0.91 at epoch 1 and 0.78 at epoch 20, versus approximately 3.4e-7 in the original
M2 at epoch 15. Layer sizes affect raw norms; these measurements are supporting
evidence, not a gradient equality requirement. ReLU does not have sigmoid saturation.

The remaining limitation is overfitting: training accuracy reached 97.33% at epoch 20,
while validation loss worsened after epoch 6. The saved model is the earlier best
checkpoint, not the last epoch. This is one seed and the same image-level split;
it does not establish generalization to unseen artists. No improvement over ResNet
is claimed, and previously published benchmark numbers do not include this new run.

## What is saved where

- Git tracks code, dependencies, shared splits and small JSON/CSV/PNG run reports.
- `best.pt` and any `best.joblib` inside run folders stay local and are ignored by Git.
- Existing tracked legacy models in `models/` are unchanged.
- `python main.py backup` creates a verified ZIP in `backups/`, containing the
  current source files, dependency lists, shared splits and all run files including
  trained weights. It includes a SHA-256 manifest and an external ZIP checksum.
- Backups omit `.venv/`, Git history, the dataset and the legacy `models/` directory.
  Restore source/results by extracting into a new folder; recreate the Python 3.13
  environment with `requirements-lock.txt` and obtain the dataset from the repository
  or dataset builder. Move copied absolute paths in old configs as needed.

A ZIP on the same disk is a local recovery copy. Copy it and its `.sha256` file to
independent storage to protect against laptop or disk loss. `backups/` is ignored by Git.
