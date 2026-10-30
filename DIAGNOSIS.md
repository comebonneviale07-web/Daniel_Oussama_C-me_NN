# Training monitoring and failure diagnosis (deliverable 2)

`diagnose.py` breaks or repairs one thing at a time on the shared split (`splits/style5.json`, 1,050 train /
225 validation paintings, 5 styles) and records what a loss curve alone cannot show: the mean gradient L2 norm of
every weight matrix per epoch, the share of saturated sigmoid outputs (< .01 or > .99), of ReLU outputs exactly 0,
of hidden units that output the same value for every painting ("constant units"), and the number of distinct
classes the network still predicts. All numbers below are **validation** scores; the test set is not used here.

```powershell
python diagnose.py --experiment all --threads 8          # ~36 min on a laptop CPU
python -m unittest -v test_diagnose
```

Reference value: **ln 5 = 1.609** is the cross-entropy of a network that gives 1/5 to every style, i.e. that has
learned nothing. Chance accuracy is 0.20; a network that always answers the same style has macro F1 0.067.

| Experiment | Run folder under `runs/` |
| --- | --- |
| M2 prototype and ablation | `diagnosis_prototype_1793353811085666600` |
| Learning rate | `diagnosis_lr_1793353972337064700` |
| Initialisation | `diagnosis_init_1793354031511055000` |
| Input normalisation | `diagnosis_normalisation_1793354179974204200` |
| Regularisation (3 seeds) | `diagnosis_regularisation_1793355497162975600` |

Each folder holds `summary.json` (one row per run, with activation statistics), `histories.json` (every epoch,
every layer's gradient) and a figure (training loss, validation macro F1, gradients of the first run).

## 1. Why M2 predicts Cubism for every painting

M2 as in `experiments.py` (5 sigmoid layers 512-256-128-64-32, dropout .3, Adam lr 1e-2, inputs in [0, 1]):

- **Symptom:** the training loss stays at **1.61 = ln 5** for 20 epochs, the network predicts **one class**
  (macro F1 0.067), as in the original run (`m2_1793300452973140000`, all 225 test paintings predicted Cubism).
- **Gradients:** the first layer receives a gradient **3 x 10^-8 times** the last layer's. In the original run's own
  `gradients.json` the ratio is already 1.4 x 10^-4 at epoch 1 (5.4e-05 vs 0.39) and 3.4 x 10^-7 at epoch 15
  (4.3e-09 vs 1.3e-02): the signal fades layer after layer, the first layers stop learning.
- **Activations:** **97-100 %** of the sigmoid outputs of layers 1-3 are saturated (45 % of layer 1 is exactly 0),
  and **97-100 %** of the units of every layer are constant: they output the same value for every painting, so
  nothing about the image reaches the classifier. A saturated sigmoid has a derivative close to 0, which is what
  kills the gradient.

**Cause, then fix, one change at a time (ablation, 20 epochs, same seed):**

| Variant | Final train loss | Best val macro F1 | Predicted classes | Constant units (layers 1-5) |
| --- | ---: | ---: | ---: | --- |
| M2 as trained (sigmoid, dropout .3, lr 1e-2, inputs 0-1) | 1.61 | 0.067 | 1 | 97-100 % |
| lr 1e-4 | 1.60 | 0.186 | 2 | 0-3 % |
| lr 1e-4, no dropout | 1.52 | 0.200 | 4 | 0 % |
| ReLU (lr 1e-2) | 1.61 | 0.189 | 1 | 87-100 % (88-100 % of ReLU outputs are 0: dead units) |
| lr 1e-4, ReLU | 1.56 | 0.238 | 4 | 22-89 % |
| lr 1e-4, ReLU, no dropout | 1.37 | 0.326 | 5 | 16-66 % |
| **lr 1e-4, ReLU, no dropout, standardised inputs** | **0.16** | **0.409** | 5 | **0-14 %** |

- **The learning rate of 1e-2 is the trigger:** the first updates push the units into saturation (sigmoid) or
  kill them (ReLU: 88-100 % dead). Lowering it alone removes the constant units, but a 5-layer sigmoid network then
  learns far too slowly (0.19).
- **No single change is enough; four are needed together:** lr 1e-4, ReLU, no dropout on layers as narrow as 32
  units, and standardised inputs. Only then does the loss leave ln 5 and the network reach 0.41, the level of the
  one-layer MLP (section 5). Five layers bring nothing over one: depth is not what limits an MLP on pixels.
- The stagnant loss was correctly treated as a hypothesis in the README; the per-layer gradients and the
  activation statistics now confirm **vanishing gradients through saturated sigmoids, triggered by a too-large
  learning rate**.

## 2. Learning rate (M1: one ReLU layer of 256, standardised inputs, Adam, 20 epochs)

| Learning rate | First-epoch loss | Final train loss | Best val macro F1 (epoch) |
| --- | ---: | ---: | ---: |
| 1e-5 | 1.54 | 0.82 | 0.403 (19) |
| **1e-4** | 1.65 | 0.14 | **0.421 (3)** |
| 1e-3 (M1's default) | 4.48 | 0.27 | 0.375 (12) |
| 1e-2 | 55.9 | 12.7 | 0.371 (4) |
| 1e-1 | 3,290 | 1,780 | 0.335 (9) |
| SGD, lr 1 | `NaN` after 1 epoch | - | diverged |

- 1e-5 is too slow (still improving at epoch 19); 1e-4 is best; from 1e-3 the first epoch already overshoots
  (loss 4.5 instead of 1.6) and at 1e-2 / 1e-1 the loss is huge and erratic. M1's default of 1e-3 costs about
  5 points of macro F1 compared with 1e-4.
- With one hidden layer a large learning rate makes training unstable but does **not** collapse to one class;
  with M2's five layers it does (section 1). Depth amplifies the damage.

## 3. Initialisation (5 ReLU layers of 256, plain SGD lr 1e-2)

SGD is used on purpose: Adam rescales each update by the recent gradient size and partly hides a vanishing gradient.

| Initialisation | Gradient norm at initialisation (first -> last layer) | Result |
| --- | --- | --- |
| He, std sqrt(2 / fan_in) | 11 -> 4 | learns, best val macro F1 0.390 |
| std 0.01 | 1.5e-4 everywhere (about 10^4 times smaller) | loss stuck at 1.61, one class predicted |
| std 1 | 3.2e6 -> 1.1e7 (growing towards the output) | `NaN` after one epoch |

Too small weights shrink the signal at every layer (vanishing), too large ones multiply it (exploding). The
initialisation has to be scaled to the layer size and activation; PyTorch's default does this for `nn.Linear`.

## 4. Input normalisation (M1, 20 epochs)

| Inputs | Adam lr 1e-4: first-epoch loss / best val macro F1 | SGD lr 1e-2: first-epoch loss / best val macro F1 |
| --- | --- | --- |
| raw 0-255 | 43.7 / 0.265 | 171,000 / 0.067 (collapse to one class) |
| 0-1 (as in `experiments.py`) | 1.62 / 0.374 | 1.62 / 0.259 |
| **standardised per channel (training statistics)** | **1.65 / 0.421** | 1.81 / 0.306 (unstable: final loss 2 x 10^4) |

Forgetting to scale the pixels (0-255) breaks training; scaling to 0-1, as `experiments.py` does, works, but
standardising each colour channel with the training mean and standard deviation adds about 5 points with Adam.
SGD with lr 1e-2 is too aggressive for standardised inputs (larger input variance): the optimiser and the input
scale have to be tuned together.

## 5. Overfitting (M1 standardised, Adam lr 1e-4, 40 epochs, 3 seeds)

| Regularisation | Best val macro F1 (mean +- std over 3 seeds) | Final train accuracy |
| --- | ---: | ---: |
| none | 0.415 +- 0.016 | 0.99 |
| dropout .5 | 0.393 +- 0.013 | 0.95 |
| weight decay 1 (AdamW) | 0.409 +- 0.008 | 0.99 |
| **mirror (flip) augmentation** | **0.438 +- 0.014** | 0.99 |
| dropout .5 + flip augmentation | 0.435 +- 0.012 | 0.92 |

- **The MLP memorises the training set** (99 % train accuracy) and its validation score peaks after 2-5 epochs:
  keeping the best validation epoch (early stopping) is the main protection, which `experiments.py` already does
  by selecting `best.pt` on validation loss.
- **Only augmentation helps** (+2.3 points, about 1.5 standard deviations): mirrored paintings are new examples.
  Dropout does not help here, and AdamW's weight decay (lr x decay = 1e-4 per step) is too weak to matter.
- **Regularisation does not change the verdict:** the MLP on 64 px pixels stays around 0.42-0.44 validation macro
  F1. Its limit is the representation (flattened pixels carry no notion of shape), which is why the pretrained
  ResNet (75 % test accuracy, `m3_frozen_1793300573210941500`) is far ahead.

## Test score of the repaired M2 (`m2_fixed`)

The four fixes are now a model of `experiments.py` (`python train_m2_fixed.py`): M2's widths with ReLU, no dropout,
inputs standardised with the training-set channel statistics, Adam lr 1e-4, 20 epochs, checkpoint chosen on
validation loss (epoch 6). M2 itself is kept unchanged as the failure case. Run `m2_fixed_1793368618444914200`:

| | M2 | M2 fixed |
| --- | ---: | ---: |
| Test accuracy [95 % CI] | 0.200 [0.151, 0.253] | **0.404 [0.342, 0.471]** |
| Test macro F1 | 0.067 | **0.390** |
| Styles predicted on the test set | 1 (Cubism) | **5** |
| First / last layer gradient, epoch 1 | 5.4e-05 / 0.39 | 0.19 / 0.20 |

Test confusion matrix of M2 fixed (rows = true style, columns = predicted; Baroque, Cubism, Impressionism, Pop_Art,
Ukiyo_e): `[30, 0, 10, 3, 2]`, `[10, 8, 5, 15, 7]`, `[14, 5, 11, 10, 5]`, `[6, 7, 6, 22, 4]`, `[4, 5, 11, 5, 20]`.
The gradient now reaches the first layer as strongly as the last, and every style is predicted. Baroque (30 / 45)
and Pop_Art (22 / 45) are recognised best; Cubism is the weakest (8 / 45, often taken for Pop_Art). The repaired
network is at the level of the logistic regression M0 (0.378, overlapping intervals) and still far below the
pretrained ResNet (0.75-0.83): the repair removes the bug, not the limit of an MLP on flattened pixels.

## Recommended settings for the final MLP (M1)

Standardised inputs, one ReLU layer, Adam lr 1e-4, mirror augmentation, model selected on validation.
Compared with the current M1 (inputs 0-1, lr 1e-3, 3 epochs, test accuracy 30.2 %), each change above is backed by
a measurement; the test score of the new setting still has to be measured once with `experiments.py`.

**Limits:** one seed for sections 1-4 (differences below ~3 points are within seed noise, see section 5), 225
validation paintings, and an image-level split: 23 of the 225 test paintings are by a named artist who also appears
in the training set, and 93 % of the paintings (89 % of the test set) are by "Unknown Artist", for whom overlap
cannot be checked.
