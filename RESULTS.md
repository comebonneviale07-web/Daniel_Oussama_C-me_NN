# Saved experiment progress

These runs share the saved stratified split: 1,050 training, 225 validation and
225 test paintings. Each class has 45 test paintings. Checkpoints were selected
using validation loss; these are test results, not validation results.

| Model | Run folder under `runs/` | Test accuracy | Macro F1 | Total experiment time |
| --- | --- | ---: | ---: | ---: |
| M1 | `m1_1793299518786019100` | 30.22% | 0.2565 | 35.3 seconds |
| M2 | `m2_1793300452973140000` | 20.00% | 0.0667 | 92.6 seconds |
| Frozen ResNet | `m3_frozen_1793300573210941500` | 75.11% | 0.7485 | 961.8 seconds |

Times include experiment overhead and are not inference latency benchmarks.
The run folders contain settings, split snapshots, training histories, curves,
confusion matrices and individual test predictions. M2 also contains gradient logs
and plots. M2 predicted Cubism for every test image; stagnant performance alone
does not establish vanishing gradients as the cause.

M0's folder `m0_1793300405327349300` contains configuration and split files only.
It is an incomplete run and has no reported performance. Full ResNet fine-tuning
has not been rerun with the corrected shared split.

## What is saved where

- Git tracks code, dependencies, shared splits and small JSON/CSV/PNG run reports.
- `best.pt` and any `best.joblib` inside run folders stay local and are ignored by Git.
- Existing tracked legacy models in `models/` are unchanged.
- `python backup_progress.py` creates a verified ZIP in `backups/`, containing the
  current source files, dependency lists, shared splits and all run files including
  trained weights. It includes a SHA-256 manifest and an external ZIP checksum.
- Backups omit `.venv/`, Git history, the dataset and the legacy `models/` directory.
  Restore source/results by extracting into a new folder; recreate the Python 3.13
  environment with `requirements-lock.txt` and obtain the dataset from the repository
  or dataset builder. Move copied absolute paths in old configs as needed.

A ZIP on the same disk is a local recovery copy. Copy it and its `.sha256` file to
independent storage to protect against laptop or disk loss. `backups/` is ignored by Git.
