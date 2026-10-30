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
