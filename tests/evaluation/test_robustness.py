"""The simulated visitor-photo degradations are reproducible and keep the image format."""
import unittest

import numpy as np
from PIL import Image

from src.robustness import CONDITIONS


class RobustnessTests(unittest.TestCase):
    def test_conditions_are_deterministic_and_keep_size_and_mode(self):
        rng = np.random.default_rng(0)
        image = Image.fromarray(rng.integers(0, 255, (128, 96, 3), dtype=np.uint8))
        for name, degrade in CONDITIONS.items():
            for index in (0, 1):
                first, second = degrade(image, index), degrade(image, index)
                self.assertEqual((first.size, first.mode), (image.size, "RGB"), name)
                self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)), name)
        self.assertTrue(np.array_equal(np.asarray(CONDITIONS["clean"](image, 0)), np.asarray(image)))
        for name in ("tilt 10 degrees", "blur", "all together"):
            self.assertFalse(np.array_equal(np.asarray(CONDITIONS[name](image, 0)), np.asarray(image)), name)


if __name__ == "__main__":
    unittest.main()
