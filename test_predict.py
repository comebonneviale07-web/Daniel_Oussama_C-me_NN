"""Offline checks of prediction on new images: loading runs of experiments.py and ranking styles."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

from experiments import build_model
from predict import default_run, load_run, predict

CLASSES = ["Baroque", "Cubism", "Impressionism", "Pop_Art", "Ukiyo_e"]


def save_run(runs, folder, name):
    """A run folder in the format written by experiments.py, with untrained weights."""
    run = runs / folder
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"model": name, "classes": CLASSES}))
    if name == "m0":
        rng = np.random.default_rng(0)
        model = LogisticRegression(max_iter=50).fit(rng.random((10, 3 * 64 * 64)), np.arange(10) % 5)
        joblib.dump(model, run / "best.joblib")
    else:
        torch.manual_seed(0)
        torch.save({"state_dict": build_model(name, 5).state_dict(), "model": name, "classes": CLASSES, "epoch": 1},
                   run / "best.pt")
    return run


class PredictTests(unittest.TestCase):
    def test_neural_and_logistic_runs_rank_every_style(self):
        image = Image.new("RGB", (300, 200), (200, 120, 40))
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("m1", "m0"):
                ranking = predict(*load_run(save_run(Path(temporary), f"{name}_1", name)), image, top=3)
                self.assertEqual(len(ranking), 3)
                self.assertTrue(set(s for s, _ in ranking) <= set(CLASSES))
                probabilities = [p for _, p in ranking]
                self.assertEqual(probabilities, sorted(probabilities, reverse=True))
                self.assertLessEqual(sum(probabilities), 1 + 1e-6)

    def test_default_run_prefers_resnet_and_skips_runs_without_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary)
            save_run(runs, "m1_100", "m1")
            save_run(runs, "m0_100", "m0")
            (runs / "m3_200").mkdir()  # reports only, weights not on this computer
            (runs / "m3_200" / "config.json").write_text(json.dumps({"model": "m3", "classes": CLASSES}))
            (runs / "m3_frozen_300").mkdir()  # weights without their config.json (e.g. another branch)
            torch.save({}, runs / "m3_frozen_300" / "best.pt")
            self.assertEqual(default_run(runs).name, "m1_100")

    def test_command_line_reports_uncertain_answers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_run(root / "runs", "m1_100", "m1")
            Image.new("RGB", (64, 64)).save(root / "painting.png")
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("predict.py")), str(root / "painting.png"),
                                     "--runs", str(root / "runs"), "--min-confidence", "1", "--json"],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            answer = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertTrue(answer["uncertain"])
            self.assertEqual(answer["model"], "m1")


if __name__ == "__main__":
    unittest.main()
