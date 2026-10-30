"""Offline checks of the diagnosis tools: M2 equivalence, initialisations, activation statistics, training."""
import math
import unittest

import torch

from src.diagnose import M2_WIDTHS, activation_stats, build_mlp, scale, summary, train
from src.training import build_model


def toy_arrays(n=60, classes=3, seed=0):
    """Tiny separable problem: the class is encoded in the mean brightness of the first channel."""
    generator = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % classes
    x = torch.rand(n, 3, 4, 4, generator=generator) * .1
    x[:, 0] += y.view(-1, 1, 1) / classes
    return {"train": (x, y), "val": (x.clone(), y.clone()), "test": (x.clone(), y.clone())}


class DiagnoseTests(unittest.TestCase):
    def test_sigmoid_dropout_mlp_matches_experiments_m2(self):
        ours, theirs = build_mlp(M2_WIDTHS, 5, "sigmoid", .3), build_model("m2", 5)
        self.assertEqual([(n, p.shape) for n, p in ours.named_parameters()],
                         [(n, p.shape) for n, p in theirs.named_parameters()])
        self.assertEqual([type(m) for m in ours], [type(m) for m in theirs])

    def test_initialisations_have_the_requested_scale(self):
        torch.manual_seed(0)
        for init, expected in (("he", math.sqrt(2 / 1000)), ("small", .01), ("large", 1.)):
            weight = build_mlp([1000, 500], 2, init=init)[1].weight
            self.assertAlmostEqual(weight.std().item(), expected, delta=expected * .05)

    def test_activation_stats_detect_saturated_and_constant_units(self):
        model = build_mlp([3 * 4 * 4, 8], 2, "sigmoid", init="large")
        with torch.no_grad():
            model[1].weight.mul_(1000)  # pushes every sigmoid to 0 or 1
            model[1].weight[0].zero_()  # unit 0 outputs sigmoid(0) = .5 for every input
        stats = activation_stats(model, torch.rand(32, 3, 4, 4))
        self.assertGreater(stats[0]["saturated"], .8)
        self.assertGreaterEqual(stats[0]["constant_units"], 1 / 8)

    def test_training_learns_records_gradients_and_detects_divergence(self):
        arrays = scale(toy_arrays(), "standard")
        torch.manual_seed(0)
        model = build_mlp([3 * 4 * 4, 16], 3)
        history = train(model, arrays, epochs=30, lr=1e-2)
        self.assertGreater(history[-1]["val_accuracy"], .9)
        self.assertEqual(set(history[0]["gradients"]), {"1.weight", "3.weight"})
        row = summary("toy", history, model, arrays)
        self.assertFalse(row["diverged"])
        self.assertEqual(len(row["activations"]), 1)

        torch.manual_seed(0)
        exploded = train(build_mlp([3 * 4 * 4, 16], 3, init="large"), scale(toy_arrays(), "raw"),
                         epochs=5, lr=1e8, optimizer="sgd")
        self.assertTrue(summary("exploded", exploded)["diverged"])
        self.assertLess(len(exploded), 5)


if __name__ == "__main__":
    unittest.main()
