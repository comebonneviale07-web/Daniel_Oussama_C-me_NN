"""Kiosk CPU benchmark and final comparison of the latest completed run of every model. See --help and README.md.

Latency is measured the way a kiosk works: one painting at a time, on one CPU thread, from the decoded image to
the predicted style (the model's own preprocessing included). Latency and size depend on the architecture, not on
the trained values, so a run whose weights are not on this computer (best.pt is not in Git) is timed with its
exact architecture. Accuracy and macro F1 come from the run's test_predictions.csv, with 95 % bootstrap intervals.
"""
import argparse
import csv
import io
import json
import os
import pickle
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from experiments import ROOT, build_model, preprocessing, write_json

FAMILIES = {"m0": "Classical", "m1": "MLP on pixels", "m2": "MLP on pixels",
            "m3": "Pretrained ResNet-18", "m3_frozen": "Pretrained ResNet-18"}
LABELS = {"m0": "M0 logistic regression", "m1": "M1 MLP, 1 ReLU layer", "m2": "M2 MLP, 5 sigmoid layers",
          "m3": "M3 ResNet-18, fine-tuned", "m3_frozen": "M3 ResNet-18, frozen"}
COLORS = {"Classical": "#eb6834", "MLP on pixels": "#2a78d6", "Pretrained ResNet-18": "#1baf7a"}


def latest_runs(runs_dir):
    """{model: run folder} for the most recent run of each model that reached its test evaluation."""
    found = {}
    for metrics in sorted(runs_dir.glob("*/test_metrics.json")):  # folder names end with time_ns
        config = json.loads((metrics.parent / "config.json").read_text(encoding="utf-8"))
        found[config["model"]] = metrics.parent
    return found


def bootstrap_ci(truth, predicted, n=2000, seed=0):
    """95 % intervals of accuracy and macro F1, resampling the test paintings with replacement."""
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    rng = np.random.default_rng(seed)
    accuracy, macro_f1 = [], []
    for _ in range(n):
        i = rng.integers(0, len(truth), len(truth))
        accuracy.append((truth[i] == predicted[i]).mean())
        macro_f1.append(f1_score(truth[i], predicted[i], average="macro", zero_division=0))
    return np.percentile(accuracy, [2.5, 97.5]).tolist(), np.percentile(macro_f1, [2.5, 97.5]).tolist()


def size_mb(model):
    buffer = io.BytesIO()
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), buffer)
    else:
        pickle.dump(model, buffer)
    return buffer.tell() / 1e6


def load_model(name, run, classes):
    """The run's trained model if its weights are on this computer, else the same architecture (trained=False)."""
    if name == "m0":
        import joblib  # our own run output, not an untrusted file
        return joblib.load(run / "best.joblib"), True
    model = build_model(name, len(classes), pretrained=False)
    checkpoint = run / "best.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    return model.eval(), checkpoint.exists()


def latency_ms(name, model, images, warmup=3):
    transform = preprocessing(name)
    if name == "m0":
        predict = lambda image: model.predict(transform(image).numpy().reshape(1, -1))
    else:
        predict = lambda image: model(transform(image).unsqueeze(0)).argmax(1)
    times = []
    with torch.inference_mode(), threadpool_limits(limits=1):
        for image in images[:warmup]:
            predict(image)
        for image in images:
            started = time.perf_counter()
            predict(image)
            times.append((time.perf_counter() - started) * 1000)
    times.sort()
    return {"median_ms": statistics.median(times), "p90_ms": times[int(.9 * len(times)) - 1]}


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(8.5, 5))
    for family, color in COLORS.items():
        subset = [r for r in rows if r["family"] == family]
        if not subset:
            continue
        axis.errorbar([r["latency_median_ms"] for r in subset], [r["test_accuracy"] for r in subset],
                      yerr=np.array([[r["test_accuracy"] - r["accuracy_ci95"][0], r["accuracy_ci95"][1] - r["test_accuracy"]]
                                     for r in subset]).T,
                      fmt="o", ms=8, color=color, elinewidth=1.5, markeredgecolor="white", markeredgewidth=1.5,
                      label=family)
    for r in rows:
        axis.annotate(r["label"], (r["latency_median_ms"], r["test_accuracy"]), xytext=(8, -3),
                      textcoords="offset points", fontsize=8, color="#444444")
    axis.set_xscale("log")
    axis.axhline(.2, color="#888888", linestyle=":", linewidth=1)
    axis.set(xlabel="CPU time per painting, one thread (ms, log scale)", ylabel="Test accuracy (95 % CI)",
             title="Accuracy vs kiosk CPU cost (same 225 test paintings)", ylim=(0, 1),
             xlim=(min(r["latency_median_ms"] for r in rows) / 2, max(r["latency_median_ms"] for r in rows) * 8))
    axis.text(axis.get_xlim()[0] * 1.1, .21, "chance (1/5)", fontsize=8, color="#666666")
    axis.grid(alpha=.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--data", type=Path, default=ROOT / "data/wikiart_style5")
    parser.add_argument("--split", type=Path, default=ROOT / "splits/style5.json")
    parser.add_argument("--images", type=int, default=60, help="test paintings timed one by one")
    args = parser.parse_args()
    torch.set_num_threads(1)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    images = [Image.open(args.data / p).convert("RGB") for p in split["test"][:args.images]]
    rows = []
    for name, run in latest_runs(args.runs).items():
        with (run / "test_predictions.csv").open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        truth = [p["true_class"] for p in predictions]
        predicted = [p["predicted_class"] for p in predictions]
        accuracy_ci, f1_ci = bootstrap_ci(truth, predicted)
        model, trained = load_model(name, run, split["classes"])
        rows.append({"model": name, "label": LABELS[name], "family": FAMILIES[name], "run": run.name,
                     "test_paintings": len(truth),
                     "test_accuracy": float(np.mean(np.array(truth) == np.array(predicted))), "accuracy_ci95": accuracy_ci,
                     "test_macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)), "macro_f1_ci95": f1_ci,
                     **{f"latency_{k}": v for k, v in latency_ms(name, model, images).items()},
                     "size_mb": size_mb(model), "timed_with_trained_weights": trained})
        print(f"{name:10s} acc {rows[-1]['test_accuracy']:.3f} latency {rows[-1]['latency_median_ms']:.1f} ms "
              f"size {rows[-1]['size_mb']:.1f} MB", flush=True)
    rows.sort(key=lambda r: r["test_accuracy"])
    out = args.runs / f"benchmark_{time.time_ns()}"
    write_json(out / "benchmark.json", {"cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count(),
                                        "threads": 1, "timed_images": len(images), "torch_version": torch.__version__,
                                        "models": rows})
    plot(rows, out / "accuracy_vs_latency.png")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
