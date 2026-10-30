"""Offline checks of the benchmark helpers: run discovery, confidence intervals, model size and latency."""
import json
import tempfile
import unittest
import torch
from pathlib import Path

from PIL import Image

from src.benchmark import bootstrap_ci, latency_ms, latest_runs, size_mb, load_model
from src.training import build_model


class BenchmarkTests(unittest.TestCase):
    def test_corrected_m2_loads_normalization_with_or_without_weights(self):
        torch.set_num_threads(2)
        statistics = {"mean": [.2, .4, .6], "std": [.3, .3, .3]}
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "config.json").write_text(json.dumps({"input_normalization": statistics}))
            for has_weights in (False, True):
                if has_weights:
                    torch.save({"state_dict": build_model("m2_fixed", 5).state_dict(),
                                "input_normalization": statistics}, run / "best.pt")
                model, trained = load_model("m2_fixed", run, list("abcde"))
                self.assertEqual(trained, has_weights)
                self.assertEqual(model.input_normalization, statistics)
                timing = latency_ms("m2_fixed", model, [Image.new("RGB", (128, 128))] * 2, warmup=1)
                self.assertGreater(timing["median_ms"], 0)

    def test_latest_completed_run_of_each_model_is_found(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary)
            for folder, model, complete in (("m1_100", "m1", True), ("m1_200", "m1", True), ("m1_300", "m1", False),
                                            ("m2_150", "m2", True), ("diagnosis_lr_1", None, False)):
                (runs / folder).mkdir()
                if model:
                    (runs / folder / "config.json").write_text(json.dumps({"model": model}))
                if complete:
                    (runs / folder / "test_metrics.json").write_text("{}")
            self.assertEqual({k: v.name for k, v in latest_runs(runs).items()}, {"m1": "m1_200", "m2": "m2_150"})

    def test_bootstrap_interval_contains_the_observed_accuracy(self):
        truth = ["a", "b"] * 50
        predicted = ["a", "b"] * 35 + ["b", "a"] * 15
        (low, high), (f1_low, f1_high) = bootstrap_ci(truth, predicted)
        self.assertLess(low, .7)
        self.assertGreater(high, .7)
        self.assertLess(f1_low, f1_high)

    def test_size_and_latency_of_a_small_model(self):
        model = build_model("m1", 5).eval()
        self.assertAlmostEqual(size_mb(model), (3 * 64 * 64 * 256 + 256 + 256 * 5 + 5) * 4 / 1e6, delta=.05)
        timing = latency_ms("m1", model, [Image.new("RGB", (128, 128))] * 5, warmup=1)
        self.assertGreater(timing["median_ms"], 0)
        self.assertGreaterEqual(timing["p90_ms"], timing["median_ms"] * .5)


if __name__ == "__main__":
    unittest.main()
