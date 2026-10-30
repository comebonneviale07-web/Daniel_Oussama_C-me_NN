"""The improved versions add options without changing what the historical versions do."""
import unittest

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.features import handcrafted_features, logistic_inputs, rgb_to_hsv
from src.models import build_model, preprocessing
from src.training import CheckpointSelector
from src.versions import get_version

NORMALIZATION = {"mean": [.5, .5, .5], "std": [.25, .25, .25]}


def painting(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (128, 96, 3), dtype=np.uint8))


class ImprovementTests(unittest.TestCase):
    def test_historical_versions_keep_their_architecture_and_preprocessing(self):
        for name in ("m1", "m2", "m2_fixed"):
            self.assertFalse(any(isinstance(m, torch.nn.BatchNorm1d) for m in build_model(name, 5)))
        image = painting()
        for name, normalization in (("m0", None), ("m1", None), ("m2_fixed", NORMALIZATION), ("m3", None)):
            torch.manual_seed(0)
            first = preprocessing(name, normalization, train=True)(image)
            second = preprocessing(name, normalization)(image)
            torch.testing.assert_close(first, second)  # no augmentation for versions without the setting

    def test_augmentation_changes_training_images_only(self):
        image = painting()
        for name, normalization in (("m1_tuned", NORMALIZATION), ("m3_augmented", None)):
            evaluation = preprocessing(name, normalization)
            torch.testing.assert_close(evaluation(image), evaluation(image))
            training = preprocessing(name, normalization, train=True)
            torch.manual_seed(1)
            samples = [training(image) for _ in range(6)]
            self.assertTrue(any(not torch.equal(samples[0], s) for s in samples[1:]))
            self.assertEqual(samples[0].shape, evaluation(image).shape)
        flips = [t for t in preprocessing("m1_tuned", NORMALIZATION, train=True).transforms
                 if isinstance(t, transforms.RandomVerticalFlip)]
        self.assertEqual(flips, [])

    def test_m3_augmented_versions_differ_from_their_reference_only_by_training_settings(self):
        for new, old in (("m3_augmented", "m3"), ("m3_frozen_augmented", "m3_frozen")):
            self.assertEqual(get_version(new)["frozen"], get_version(old)["frozen"])
            new_model, old_model = build_model(new, 5, pretrained=False), build_model(old, 5, pretrained=False)
            self.assertEqual([(n, p.shape, p.requires_grad) for n, p in new_model.named_parameters()],
                             [(n, p.shape, p.requires_grad) for n, p in old_model.named_parameters()])

    def test_regularised_m2_adds_batch_norm_after_each_hidden_layer(self):
        model = build_model("m2_regularised", 5)
        self.assertEqual(sum(isinstance(m, torch.nn.BatchNorm1d) for m in model), 5)
        self.assertEqual(sum(isinstance(m, torch.nn.ReLU) for m in model), 5)
        model.train()
        output = model(torch.rand(4, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (4, 5))

    def test_handcrafted_features_have_a_fixed_finite_size(self):
        images = torch.rand(3, 3, 64, 64)
        images[1] = 0  # a black image must not produce NaN
        features = handcrafted_features(images)
        self.assertEqual(features.shape, (3, 190))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[:, :128].sum(1), 1, rtol=1e-5)  # normalised colour histogram
        self.assertEqual(logistic_inputs(get_version("m0"), images).shape, (3, 3 * 64 * 64))
        self.assertEqual(logistic_inputs(get_version("m0_features"), images).shape, (3, 190))

    def test_hsv_conversion_of_primary_colours(self):
        x = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.5, .5, .5]]).view(4, 3, 1, 1)
        hsv = rgb_to_hsv(x).view(4, 3)
        torch.testing.assert_close(hsv[:, 0], torch.tensor([0., 1 / 3, 2 / 3, 0.]))
        torch.testing.assert_close(hsv[:, 1], torch.tensor([1., 1., 1., 0.]))
        torch.testing.assert_close(hsv[:, 2], torch.tensor([1., 1., 1., .5]))

    def test_checkpoint_selector_metric_direction_and_patience(self):
        loss = CheckpointSelector("val_loss")
        self.assertEqual([loss.update({"epoch": e, "val_loss": v})[0] for e, v in enumerate((1., .8, .9), 1)],
                         [True, True, False])
        f1 = CheckpointSelector("val_macro_f1", patience=2)
        steps = [f1.update({"epoch": e, "val_macro_f1": v}) for e, v in enumerate((.3, .4, .35, .38), 1)]
        self.assertEqual([improved for improved, _ in steps], [True, True, False, False])
        self.assertEqual([stop for _, stop in steps], [False, False, False, True])
        self.assertEqual(f1.best_epoch, 2)
        with self.assertRaises(ValueError):
            CheckpointSelector("test_accuracy")


if __name__ == "__main__":
    unittest.main()
