"""Reproducible, CPU-first Museum AI experiments. See --help and README.md."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.versions import MODEL_VERSIONS, get_version, write_run_readme
from src.models import build_model, preprocessing

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def shared_split(dataset, path, seed):
    paths = [Path(p).relative_to(dataset.root).as_posix() for p, _ in dataset.samples]
    fingerprint = hashlib.sha256("\n".join(paths).encode()).hexdigest()
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        if split["fingerprint"] != fingerprint or split["classes"] != dataset.classes:
            raise ValueError("Dataset changed; use a new split file for the new experiment.")
    else:
        indices = np.arange(len(paths))
        train, rest = train_test_split(indices, test_size=.30, stratify=dataset.targets, random_state=seed)
        val, test = train_test_split(rest, test_size=.5,
                                     stratify=np.array(dataset.targets)[rest], random_state=seed)
        split = {"seed": seed, "fingerprint": fingerprint, "classes": dataset.classes,
                 **{key: [paths[i] for i in ids] for key, ids in
                    (("train", train), ("val", val), ("test", test))}}
        write_json(path, split)
    groups = [split[key] for key in ("train", "val", "test")]
    flat = [p for group in groups for p in group]
    if len(flat) != len(paths) or len(set(flat)) != len(paths) or set(flat) != set(paths):
        raise ValueError("Split must partition the dataset exactly once.")
    lookup = {p: i for i, p in enumerate(paths)}
    return split, {key: [lookup[p] for p in split[key]] for key in ("train", "val", "test")}


def training_normalization(dataset, train_indices, batch_size=32):
    """Channel statistics from training pixels only, before applying normalization."""
    total = torch.zeros(3, dtype=torch.float64)
    squared = torch.zeros(3, dtype=torch.float64)
    pixels = 0
    for images, _ in DataLoader(Subset(dataset, train_indices), batch_size=batch_size):
        images = images.double()
        total += images.sum(dim=(0, 2, 3))
        squared += images.square().sum(dim=(0, 2, 3))
        pixels += images.shape[0] * images.shape[2] * images.shape[3]
    mean = total / pixels
    std = (squared / pixels - mean.square()).clamp_min(1e-12).sqrt()
    return {"mean": mean.tolist(), "std": std.tolist(), "source": "training split only"}


def evaluate(model, loader, device):
    model.eval()
    total_loss, truth, predicted = 0., [], []
    with torch.inference_mode():
        for x, y in loader:
            output = model(x.to(device))
            total_loss += nn.functional.cross_entropy(output, y.to(device), reduction="sum").item()
            truth.extend(y.tolist())
            predicted.extend(output.argmax(1).cpu().tolist())
    return total_loss / len(truth), accuracy_score(truth, predicted), truth, predicted


def main(default_model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODEL_VERSIONS), default=default_model or "m1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/wikiart_style5")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/style5.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads; avoids oversubscribing a laptop")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--split-only", action="store_true")
    args = parser.parse_args()
    version = get_version(args.model)
    args.output = args.output or ROOT / "results" / MODEL_VERSIONS[args.model]
    args.epochs = args.epochs if args.epochs is not None else version["epochs"]
    args.lr = args.lr if args.lr is not None else version["lr"]
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.threads < 1:
        parser.error("epochs, batch size, threads, and learning rate must be positive")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    needs_normalization = version["normalization"] == "training_channel"
    dataset = datasets.ImageFolder(args.data, transform=preprocessing("m1" if needs_normalization else args.model))
    split, ids = shared_split(dataset, args.split, args.seed)
    print({key: len(value) for key, value in ids.items()})
    if args.split_only:
        return
    normalization = None
    if needs_normalization:
        normalization = training_normalization(dataset, ids["train"], args.batch_size)
        dataset.transform = preprocessing(args.model, normalization)
    # A unique directory prevents accidental replacement of previous experiments.
    out = args.output / f"{args.model}_{time.time_ns()}"
    out.mkdir(parents=True)
    run_config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                    "model_version": version["version"], "version_settings": version,
                                    "classes": dataset.classes, "input_normalization": normalization,
                                    "activation": version.get("activation"),
                                    "dropout": version.get("dropout"),
                                    "torch_version": torch.__version__,
                                    "numpy_version": np.__version__, "python_version": platform.python_version(),
                                    "package_versions": {name: importlib.metadata.version(name) for name in
                                                         ("torch", "torchvision", "scikit-learn", "numpy", "pillow", "matplotlib")},
                                    "input_size": version["input_size"]}
    write_json(out / "config.json", run_config)
    write_run_readme(out, run_config)
    write_json(out / "split.json", split)
    loaders = {key: DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                               shuffle=key == "train", num_workers=0) for key, indices in ids.items()}
    started = time.perf_counter()
    selected_epoch = None
    if args.model == "m0":
        import joblib
        arrays = {}
        for key, indices in ids.items():
            samples = [dataset[i] for i in indices]
            arrays[key] = (np.stack([x.numpy().ravel() for x, _ in samples]), np.array([y for _, y in samples]))
        # C is selected on validation only; test remains untouched until selection ends.
        best_score, model = -1, None
        candidates = []
        for c in version["c_candidates"]:
            candidate = LogisticRegression(C=c, max_iter=version["max_iter"], random_state=args.seed)
            with threadpool_limits(limits=args.threads):
                candidate.fit(*arrays["train"])
            score = candidate.score(*arrays["val"])
            candidates.append({"C": c, "val_accuracy": score, "iterations": candidate.n_iter_.tolist()})
            if score > best_score:
                best_score, model = score, candidate
        write_json(out / "selection.json", candidates)
        joblib.dump(model, out / "best.joblib")
        truth = arrays["test"][1].tolist()
        predicted = model.predict(arrays["test"][0]).tolist()
    else:
        device = torch.device(args.device)
        model = build_model(args.model, len(dataset.classes)).to(device)
        model.input_normalization = normalization
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        history, gradients, best_loss = [], [], float("inf")
        for epoch in range(1, args.epochs + 1):
            model.train()
            if args.model == "m3_frozen":
                model.eval()  # Freeze batch-normalization statistics as well as weights.
                model.fc.train()
            loss_sum, correct, count = 0., 0, 0
            grad_sum, batches = {}, 0
            for x, y in loaders["train"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = nn.functional.cross_entropy(output, y)
                loss.backward()
                if args.model in ("m2", "m2_fixed"):
                    for name, param in model.named_parameters():
                        if name.endswith("weight") and param.grad is not None:
                            grad_sum[name] = grad_sum.get(name, 0.) + param.grad.norm().item()
                optimizer.step()
                batches += 1
                loss_sum += loss.item() * len(y)
                correct += (output.argmax(1) == y).sum().item()
                count += len(y)
            val_loss, val_acc, val_truth, val_predicted = evaluate(model, loaders["val"], device)
            row = dict(epoch=epoch, train_loss=loss_sum/count, train_accuracy=correct/count,
                       val_loss=val_loss, val_accuracy=val_acc,
                       val_macro_f1=f1_score(val_truth, val_predicted, labels=list(range(len(dataset.classes))),
                                             average="macro", zero_division=0),
                       val_predicted_classes=len(set(val_predicted)))
            history.append(row)
            gradients.append({"epoch": epoch, **{key: value/batches for key, value in grad_sum.items()}})
            write_json(out / "history.json", history)
            if args.model in ("m2", "m2_fixed"):
                write_json(out / "gradients.json", gradients)
            print(row, flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"state_dict": model.state_dict(), "model": args.model,
                            "classes": dataset.classes, "epoch": epoch,
                            "input_normalization": normalization}, out / "best.pt")
        checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
        selected_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        _, _, truth, predicted = evaluate(model, loaders["test"], device)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, metric in zip(axes, ("loss", "accuracy")):
            for phase in ("train", "val"):
                axis.plot([r["epoch"] for r in history], [r[f"{phase}_{metric}"] for r in history], label=phase)
            axis.set(xlabel="Epoch", ylabel=metric)
            axis.legend()
        fig.tight_layout()
        fig.savefig(out / "curves.png")
        plt.close(fig)
        if args.model in ("m2", "m2_fixed"):
            fig, axis = plt.subplots(figsize=(8, 4))
            for layer in grad_sum:
                axis.plot([r["epoch"] for r in gradients], [r[layer] for r in gradients], label=layer)
            axis.set(xlabel="Epoch", ylabel="Mean gradient L2 norm", yscale="log")
            axis.legend()
            fig.tight_layout()
            fig.savefig(out / "gradients.png")
            plt.close(fig)
    labels = list(range(len(dataset.classes)))
    write_json(out / "test_metrics.json", {"accuracy": accuracy_score(truth, predicted),
               "macro_f1": f1_score(truth, predicted, labels=labels, average="macro", zero_division=0),
               "classes": dataset.classes, "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
               "elapsed_seconds": time.perf_counter()-started})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    display = ConfusionMatrixDisplay.from_predictions(truth, predicted, labels=labels,
               display_labels=dataset.classes, xticks_rotation=45, colorbar=False)
    display.figure_.tight_layout()
    display.figure_.savefig(out / "confusion_matrix.png")
    plt.close(display.figure_)
    with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "true_class", "predicted_class"])
        writer.writerows((path, dataset.classes[y], dataset.classes[p]) for path, y, p in zip(split["test"], truth, predicted))
    print(f"Results: {out}")
    write_run_readme(out, run_config, json.loads((out / "test_metrics.json").read_text(encoding="utf-8")), selected_epoch)


if __name__ == "__main__":
    main()
