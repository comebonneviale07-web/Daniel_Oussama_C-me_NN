"""Training-failure diagnosis for the MLPs (deliverable 2). See --help and DIAGNOSIS.md.

Each experiment breaks or repairs one thing and records what the curves alone cannot show:
per-layer gradient norms, saturated / dead / constant hidden units, and whether the network
collapsed to a single predicted class. Results go to runs/diagnosis_<experiment>_<timestamp>/.
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from experiments import ROOT, preprocessing, shared_split, write_json

ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}
M2_WIDTHS = [3 * 64 * 64, 512, 256, 128, 64, 32]
M1_WIDTHS = [3 * 64 * 64, 256]


def build_mlp(widths, classes, activation="relu", dropout=0., init="default"):
    """Flatten -> [Linear -> activation -> Dropout] per hidden layer -> Linear.

    With M2_WIDTHS, sigmoid and dropout .3 this is exactly experiments.build_model("m2"), with the same
    parameter names. init: "default" (PyTorch), "he" (std sqrt(2/fan_in)), "small" (std .01), "large" (std 1).
    """
    layers = [nn.Flatten()]
    for incoming, outgoing in zip(widths, widths[1:]):
        layers.extend([nn.Linear(incoming, outgoing), ACTIVATIONS[activation]()])
        if dropout:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(widths[-1], classes))
    model = nn.Sequential(*layers)
    if init != "default":
        for layer in model:
            if isinstance(layer, nn.Linear):
                std = {"he": math.sqrt(2 / layer.in_features), "small": .01, "large": 1.}[init]
                nn.init.normal_(layer.weight, 0., std)
                nn.init.zeros_(layer.bias)
    return model


def load_arrays(data, split_path, seed):
    """The shared split as in-memory 64 px tensors in [0, 1] (the M1/M2 preprocessing)."""
    dataset = datasets.ImageFolder(data, transform=preprocessing("m1"))
    _, ids = shared_split(dataset, split_path, seed)
    arrays = {}
    for key, indices in ids.items():
        samples = [dataset[i] for i in indices]
        arrays[key] = (torch.stack([x for x, _ in samples]), torch.tensor([y for _, y in samples]))
    return arrays, dataset.classes


def scale(arrays, mode):
    """"unit": pixels in [0, 1] (as in experiments.py); "raw": 0-255; "standard": per-channel
    standardisation with the training statistics only."""
    if mode == "unit":
        return arrays
    if mode == "raw":
        return {k: (x * 255, y) for k, (x, y) in arrays.items()}
    mean = arrays["train"][0].mean(dim=(0, 2, 3), keepdim=True)
    std = arrays["train"][0].std(dim=(0, 2, 3), keepdim=True)
    return {k: ((x - mean) / std, y) for k, (x, y) in arrays.items()}


def activation_stats(model, x):
    """Per hidden layer: share of saturated sigmoid outputs (<.01 or >.99), of ReLU outputs exactly 0,
    and of units whose output is (almost) the same for every painting, so they carry no information."""
    outputs = []
    hooks = [m.register_forward_hook(lambda _, __, out: outputs.append(out.detach()))
             for m in model if isinstance(m, tuple(ACTIVATIONS.values()))]
    model.eval()
    with torch.inference_mode():
        model(x)
    for hook in hooks:
        hook.remove()
    return [{"layer": i, "saturated": ((a < .01) | (a > .99)).float().mean().item(),
             "zero": (a == 0).float().mean().item(), "constant_units": (a.std(0) < 1e-3).float().mean().item()}
            for i, a in enumerate(outputs, 1)]


def evaluate(model, x, y):
    model.eval()
    with torch.inference_mode():
        output = model(x)
    predicted = output.argmax(1)
    return {"loss": nn.functional.cross_entropy(output, y).item(), "accuracy": (predicted == y).float().mean().item(),
            "macro_f1": f1_score(y, predicted, average="macro", zero_division=0),
            "predicted_classes": len(predicted.unique())}


def train(model, arrays, epochs, lr, optimizer="adam", weight_decay=0., seed=42, augment=False, batch_size=32):
    """Mini-batch training. Records per epoch: losses, accuracies, validation macro-F1, the number of
    distinct predicted classes and the mean gradient L2 norm of every weight matrix. Stops on NaN/inf."""
    x, y = arrays["train"]
    if augment:  # a painting mirrored left-right keeps its style
        x, y = torch.cat([x, x.flip(-1)]), torch.cat([y, y])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, generator=generator)
    params = model.parameters()
    opt = (torch.optim.SGD(params, lr=lr, momentum=.9) if optimizer == "sgd"
           else torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay) if weight_decay
           else torch.optim.Adam(params, lr=lr))
    weights = {n: p for n, p in model.named_parameters() if n.endswith("weight")}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, grads, batches = 0., 0, dict.fromkeys(weights, 0.), 0
        for xb, yb in loader:
            opt.zero_grad()
            output = model(xb)
            loss = nn.functional.cross_entropy(output, yb)
            loss.backward()
            for n, p in weights.items():
                grads[n] += p.grad.norm().item()
            opt.step()
            batches += 1
            loss_sum += loss.item() * len(yb)
            correct += (output.argmax(1) == yb).sum().item()
        val = evaluate(model, *arrays["val"])
        history.append({"epoch": epoch, "train_loss": loss_sum / len(y), "train_accuracy": correct / len(y),
                        "val_loss": val["loss"], "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                        "val_predicted_classes": val["predicted_classes"],
                        "gradients": {n: g / batches for n, g in grads.items()}})
        if not math.isfinite(history[-1]["train_loss"]):
            break
    return history


def summary(name, history, model=None, arrays=None):
    best = max(history, key=lambda r: r["val_macro_f1"])
    last = history[-1]
    grads = list(last["gradients"].values())  # in layer order, input side first
    row = {"run": name, "epochs": last["epoch"], "diverged": not math.isfinite(last["train_loss"]),
           "first_epoch_loss": history[0]["train_loss"], "final_train_loss": last["train_loss"],
           "final_train_accuracy": last["train_accuracy"], "best_val_macro_f1": best["val_macro_f1"],
           "best_val_accuracy": best["val_accuracy"], "best_epoch": best["epoch"],
           "final_val_predicted_classes": last["val_predicted_classes"],
           "first_to_last_layer_gradient_ratio": grads[0] / grads[-1] if grads[-1] else None}
    if model is not None and row["diverged"] is False:
        row["activations"] = activation_stats(model, arrays["val"][0])
    return row


def seeded(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------------------------------
# Experiments: each returns (rows, histories)
# --------------------------------------------------------------------------------------------------
def run(name, arrays, classes, epochs, widths=M1_WIDTHS, activation="relu", dropout=0., init="default", seed=42, **fit):
    seeded(seed)
    model = build_mlp(widths, len(classes), activation, dropout, init)
    history = train(model, arrays, epochs, seed=seed, **fit)
    return summary(name, history, model, arrays), history


def exp_prototype(arrays, classes, epochs):
    """M2 exactly as in experiments.py, then one change at a time (ablation)."""
    m2 = dict(widths=M2_WIDTHS, activation="sigmoid", dropout=.3, lr=.01)
    variants = [("M2 as trained (sigmoid, dropout .3, lr 1e-2, inputs 0-1)", "unit", m2),
                ("lr 1e-4", "unit", {**m2, "lr": 1e-4}),
                ("lr 1e-4, no dropout", "unit", {**m2, "lr": 1e-4, "dropout": 0.}),
                ("ReLU (lr 1e-2)", "unit", {**m2, "activation": "relu"}),
                ("lr 1e-4, ReLU", "unit", {**m2, "lr": 1e-4, "activation": "relu"}),
                ("lr 1e-4, ReLU, no dropout", "unit", {**m2, "lr": 1e-4, "activation": "relu", "dropout": 0.}),
                ("lr 1e-4, ReLU, no dropout, standardised inputs", "standard",
                 {**m2, "lr": 1e-4, "activation": "relu", "dropout": 0.})]
    rows, histories = [], {}
    for name, mode, kw in variants:
        row, histories[name] = run(name, scale(arrays, mode), classes, epochs, **kw)
        rows.append(row)
    return rows, histories


def exp_lr(arrays, classes, epochs):
    data = scale(arrays, "standard")
    rows, histories = [], {}
    for lr in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        row, histories[f"lr={lr:g}"] = run(f"Adam lr={lr:g}", data, classes, epochs, lr=lr)
        rows.append(row)
    row, histories["SGD lr=1"] = run("SGD lr=1", data, classes, epochs, lr=1., optimizer="sgd")
    rows.append(row)
    return rows, histories


def exp_init(arrays, classes, epochs):
    """5 ReLU layers of 256 units, plain SGD (Adam rescales updates and partly hides vanishing gradients)."""
    data = scale(arrays, "standard")
    widths = [3 * 64 * 64] + [256] * 5
    rows, histories = [], {}
    xb, yb = data["train"][0][:256], data["train"][1][:256]
    for init in ("he", "small", "large"):
        seeded(42)
        model = build_mlp(widths, len(classes), init=init)
        nn.functional.cross_entropy(model(xb), yb).backward()
        at_init = {n: p.grad.norm().item() for n, p in model.named_parameters() if n.endswith("weight")}
        row, histories[init] = run(f"init={init}", data, classes, epochs, widths=widths, init=init,
                                   lr=.01, optimizer="sgd")
        rows.append({**row, "gradients_at_init": at_init})
    return rows, histories


def exp_normalisation(arrays, classes, epochs):
    rows, histories = [], {}
    for mode in ("raw", "unit", "standard"):
        for optimizer, lr in (("adam", 1e-4), ("sgd", .01)):
            name = f"inputs {mode}, {optimizer}"
            row, histories[name] = run(name, scale(arrays, mode), classes, epochs, lr=lr, optimizer=optimizer)
            rows.append(row)
    return rows, histories


def exp_regularisation(arrays, classes, epochs, seeds=(42, 43, 44)):
    data = scale(arrays, "standard")
    configs = {"none": {}, "dropout .5": {"dropout": .5}, "weight decay 1": {"weight_decay": 1.},
               "flip augmentation": {"augment": True}, "dropout .5 + flip augmentation": {"dropout": .5, "augment": True}}
    rows, histories = [], {}
    for name, kw in configs.items():
        for seed in seeds:
            row, histories[f"{name} seed {seed}"] = run(name, data, classes, epochs, lr=1e-4, seed=seed, **kw)
            rows.append({**row, "seed": seed})
    return rows, histories


EXPERIMENTS = {"prototype": (exp_prototype, 20), "lr": (exp_lr, 20), "init": (exp_init, 20),
               "normalisation": (exp_normalisation, 20), "regularisation": (exp_regularisation, 40)}


def plot(name, histories, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for label, history in histories.items():
        epochs = [r["epoch"] for r in history]
        axes[0].plot(epochs, [r["train_loss"] for r in history], label=label)
        axes[1].plot(epochs, [r["val_macro_f1"] for r in history], label=label)
    axes[0].axhline(math.log(5), color="black", linestyle=":", linewidth=1, label="ln 5 (uniform guess)")
    axes[0].set(xlabel="Epoch", ylabel="Train loss (log)", yscale="log", title="Training loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation macro F1", title="Validation macro F1")
    first = next(iter(histories.values()))
    for layer in first[0]["gradients"]:
        axes[2].plot([r["epoch"] for r in first], [r["gradients"][layer] for r in first], label=layer)
    axes[2].set(xlabel="Epoch", ylabel="Mean gradient L2 norm", yscale="log", title=f"Gradients: {next(iter(histories))}")
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=[*EXPERIMENTS, "all"], default="all")
    parser.add_argument("--epochs", type=int, default=None, help="override the per-experiment default")
    parser.add_argument("--data", type=Path, default=ROOT / "data/wikiart_style5")
    parser.add_argument("--split", type=Path, default=ROOT / "splits/style5.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    parser.add_argument("--seed", type=int, default=42, help="split seed (an existing split file is reused)")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    arrays, classes = load_arrays(args.data, args.split, args.seed)
    for name in EXPERIMENTS if args.experiment == "all" else [args.experiment]:
        function, epochs = EXPERIMENTS[name]
        started = time.perf_counter()
        rows, histories = function(arrays, classes, args.epochs or epochs)
        out = args.output / f"diagnosis_{name}_{time.time_ns()}"
        write_json(out / "summary.json", {"experiment": name, "classes": classes,
                                          "elapsed_seconds": time.perf_counter() - started, "runs": rows})
        write_json(out / "histories.json", histories)
        plot(name, histories, out)
        print(f"{name}: {out}", flush=True)


if __name__ == "__main__":
    main()
