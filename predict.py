"""Predict the style of new paintings with a trained run (deliverable: the guide's recognition step).

    python predict.py photo.jpg                     # latest run whose weights are on this computer, M3 first
    python predict.py a.jpg b.png --run runs/m3_1793355952561843000 --top 3 --min-confidence .5

Each image goes through the run's own preprocessing; the answer is the most likely styles with their
probability. With --min-confidence, a top probability below the threshold is reported as uncertain, so a kiosk
can say "I am not sure" instead of a wrong style.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from experiments import ROOT, build_model, preprocessing

PREFERENCE = ("m3", "m3_frozen", "m1", "m0", "m2")  # best test accuracy first, see RESULTS.md


def default_run(runs_dir):
    """Latest run of the preferred model whose trained weights are present locally (they are not in Git)."""
    for name in PREFERENCE:
        runs = sorted(p.parent for p in runs_dir.glob(f"{name}_*/best.*") if (p.parent / "config.json").exists()
                      and json.loads((p.parent / "config.json").read_text(encoding="utf-8"))["model"] == name)
        if runs:
            return runs[-1]
    raise FileNotFoundError(f"No trained run in {runs_dir}: train one with experiments.py first.")


def load_run(run):
    """(model, model name, class names) from a run folder of experiments.py."""
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    name = config["model"]
    if name == "m0":
        import joblib  # joblib files can execute code: only load runs you produced yourself
        return joblib.load(run / "best.joblib"), name, config["classes"]
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["model"] != name:
        raise ValueError(f"{run}: checkpoint is for {checkpoint['model']}, config says {name}")
    model = build_model(name, len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), name, checkpoint["classes"]


def predict(model, name, classes, image, top=3):
    """[(style, probability)] for one PIL image, most likely first."""
    x = preprocessing(name)(image.convert("RGB")).unsqueeze(0)
    if name == "m0":
        probabilities = torch.tensor(model.predict_proba(x.numpy().reshape(1, -1))[0])
    else:
        with torch.inference_mode():
            probabilities = torch.softmax(model(x), dim=1)[0]
    values, indices = probabilities.topk(min(top, len(classes)))
    return [(classes[i], float(v)) for v, i in zip(values, indices)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--run", type=Path, default=None, help="run folder (default: best available, see above)")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0., help="below this top probability: uncertain")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="print one JSON object per image")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run = args.run or default_run(args.runs)
    model, name, classes = load_run(run)
    if not args.json:
        print(f"Model {name} from {run}")
    for path in args.images:
        with Image.open(path) as image:
            ranking = predict(model, name, classes, image, args.top)
        uncertain = ranking[0][1] < args.min_confidence
        if args.json:
            print(json.dumps({"image": str(path), "model": name, "uncertain": uncertain,
                              "predictions": [{"style": s, "probability": p} for s, p in ranking]}))
        else:
            answer = "uncertain" if uncertain else ranking[0][0]
            print(f"{path}: {answer}  (" + ", ".join(f"{s} {p:.0%}" for s, p in ranking) + ")")


if __name__ == "__main__":
    main()
