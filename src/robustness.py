"""How do trained runs cope with a visitor's photo instead of a clean scan?

    python main.py robustness --runs results/M3/v2_finetuned/<run> results/M3/v3_augmented/<run>

The test paintings are degraded in fixed, reproducible ways that a phone or kiosk camera produces (tilt, oblique
view, close framing, dim light, blur, compression, and all of them together), then every run predicts them
with its own preprocessing. Same images and same degradations for every run, so the accuracies are comparable;
the McNemar test of `main.py compare` applies to each condition through the saved predictions.
"""
import argparse
import csv
import io
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import functional as F

from src.predict import load_run, predict
from src.training import ROOT, write_json


def tilt(image, index):
    return F.rotate(image, 10 if index % 2 else -10, fill=(128, 128, 128))


def oblique(image, index):
    w, h = image.size
    d = int(.15 * w)
    start = [[0, 0], [w, 0], [w, h], [0, h]]
    end = [[d, d // 2], [w, 0], [w, h], [d, h - d // 2]] if index % 2 else [[0, 0], [w - d, d // 2], [w - d, h - d // 2], [0, h]]
    return F.perspective(image, start, end, fill=(128, 128, 128))


def close_framing(image, index):
    w, h = image.size
    return F.resized_crop(image, int(.1 * h), int(.1 * w), int(.75 * h), int(.75 * w), [h, w])


def dim_light(image, index):
    return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(.65)).enhance(.75)


def blur(image, index):
    return image.filter(ImageFilter.GaussianBlur(radius=1.5))


def compression(image, index):
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=25)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def visitor_photo(image, index):
    for degrade in (close_framing, tilt, oblique, dim_light, blur, compression):
        image = degrade(image, index)
    return image


CONDITIONS = {"clean": lambda image, index: image, "tilt 10 degrees": tilt, "oblique view": oblique,
              "close framing": close_framing, "dim light": dim_light, "blur": blur, "jpeg compression": compression,
              "all together": visitor_photo}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="run folders with their weights")
    parser.add_argument("--data", type=Path, default=ROOT / "data/wikiart_style5")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/style5.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/robustness")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    paths = split["test"]
    images = [Image.open(args.data / p).convert("RGB") for p in paths]
    truth = [p.split("/")[0] for p in paths]
    out = args.output / f"robustness_{time.time_ns()}"
    table = {}
    for run in args.runs:
        model, name, classes = load_run(run)
        table[run.name] = {"model": name}
        for condition, degrade in CONDITIONS.items():
            predicted = [predict(model, name, classes, degrade(image, i), top=1)[0][0] for i, image in enumerate(images)]
            accuracy = sum(p == t for p, t in zip(predicted, truth)) / len(truth)
            table[run.name][condition] = accuracy
            folder = out / run.name / condition.replace(" ", "_")
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["path", "true_class", "predicted_class"])
                writer.writerows(zip(paths, truth, predicted))
            print(f"{run.name:45s} {condition:18s} {accuracy:.2%}", flush=True)
    write_json(out / "robustness.json", {"test_images": len(paths), "accuracy": table})
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
