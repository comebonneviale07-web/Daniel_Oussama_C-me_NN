"""Offline checks of split integrity, learning and frozen-backbone behavior."""
import tempfile
import unittest
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets

from experiments import build_model, preprocessing, shared_split


class ExperimentTests(unittest.TestCase):
    def test_training_cli_saves_held_out_predictions_and_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in range(5):
                folder = root / "images" / str(label)
                folder.mkdir(parents=True)
                for index in range(20):
                    Image.new("RGB", (16, 16), (label * 40, index * 10, 0)).save(folder / f"{index}.png")
            for name in ("m0", "m1", "m2", "m2_fixed"):
                result = subprocess.run([sys.executable, str(Path(__file__).with_name("experiments.py")),
                                "--model", name, "--data", str(root / "images"),
                                "--split", str(root / "split.json"), "--output", str(root / "runs"),
                                "--epochs", "1"], capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                run = next((root / "runs").glob(f"{name}_*"))
                metrics = json.loads((run / "test_metrics.json").read_text())
                self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 15)
                split = json.loads((run / "split.json").read_text())
                with (run / "test_predictions.csv").open(newline="", encoding="utf-8") as stream:
                    predictions = list(csv.DictReader(stream))
                self.assertEqual({row["path"] for row in predictions}, set(split["test"]))
                self.assertFalse({row["path"] for row in predictions} & set(split["train"]))
                self.assertTrue((run / "confusion_matrix.png").exists())
                checkpoint = "best.joblib" if name == "m0" else "best.pt"
                self.assertTrue((run / checkpoint).exists())
                if name.startswith("m2"):
                    gradients = json.loads((run / "gradients.json").read_text())
                    self.assertEqual(len(gradients[0]), 7)  # epoch and six linear weight matrices

    def test_saved_split_is_stratified_disjoint_and_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in range(5):
                folder = root / "images" / str(label)
                folder.mkdir(parents=True)
                for index in range(20):
                    Image.new("RGB", (8, 8), (label * 40, index, 0)).save(folder / f"{index}.png")
            dataset = datasets.ImageFolder(root / "images")
            split, groups = shared_split(dataset, root / "split.json", 42)
            self.assertEqual([len(groups[k]) for k in ("train", "val", "test")], [70, 15, 15])
            self.assertEqual(len(set(sum(groups.values(), []))), 100)
            for indices in groups.values():
                self.assertEqual(set(dataset.targets[i] for i in indices), set(range(5)))
            reused, _ = shared_split(dataset, root / "split.json", 123)
            self.assertEqual(split, reused)
            Image.new("RGB", (8, 8)).save(root / "images/0/extra.png")
            with self.assertRaisesRegex(ValueError, "Dataset changed"):
                shared_split(datasets.ImageFolder(root / "images"), root / "split.json", 42)

    def test_mlps_backpropagate_and_checkpoint_round_trip(self):
        torch.set_num_threads(2)
        for name in ("m1", "m2", "m2_fixed"):
            torch.manual_seed(42)
            model = build_model(name, 5)
            inputs = torch.rand(2, 3, 64, 64)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            before = [p.detach().clone() for p in model.parameters()]
            loss = torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1]))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
            model.eval()
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "checkpoint.pt"
                torch.save(model.state_dict(), path)
                restored = build_model(name, 5).eval()
                restored.load_state_dict(torch.load(path, weights_only=True))
                with torch.no_grad():
                    torch.testing.assert_close(model(inputs), restored(inputs))

    def test_m2_fixed_keeps_m2_widths_with_relu_no_dropout_and_standardised_inputs(self):
        fixed, original = build_model("m2_fixed", 5), build_model("m2", 5)
        self.assertEqual([p.shape for p in fixed.parameters()], [p.shape for p in original.parameters()])
        self.assertFalse(any(isinstance(m, (torch.nn.Sigmoid, torch.nn.Dropout)) for m in fixed))
        self.assertEqual(sum(isinstance(m, torch.nn.ReLU) for m in fixed), 5)
        image = Image.new("RGB", (128, 128), (255, 255, 255))
        self.assertTrue(torch.all(preprocessing("m2")(image) == 1))
        self.assertTrue(torch.all(preprocessing("m2_fixed")(image) > 1))  # standardised: white is above the mean

    def test_resnet_freezes_only_backbone_and_preprocessing_shape(self):
        model = build_model("m3_frozen", 5, pretrained=False)
        self.assertTrue(all(p.requires_grad == name.startswith("fc.") for name, p in model.named_parameters()))
        image = Image.new("RGB", (128, 128))
        self.assertEqual(tuple(preprocessing("m3")(image).shape), (3, 224, 224))


if __name__ == "__main__":
    unittest.main()
