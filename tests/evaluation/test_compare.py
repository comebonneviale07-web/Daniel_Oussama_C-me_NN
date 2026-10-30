"""Exact McNemar comparison of two runs on the same test images."""
import csv
import tempfile
import unittest
from pathlib import Path

from src.compare import mcnemar


def write_run(folder, rows):
    folder.mkdir(parents=True)
    with (folder / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "true_class", "predicted_class"])
        writer.writerows(rows)
    return folder


class CompareTests(unittest.TestCase):
    def test_identical_runs_are_not_different(self):
        rows = [(f"{i}.jpg", "Baroque", "Baroque" if i % 2 else "Cubism") for i in range(20)]
        with tempfile.TemporaryDirectory() as temporary:
            a, b = write_run(Path(temporary) / "a", rows), write_run(Path(temporary) / "b", rows)
            result = mcnemar(a, b)
        self.assertEqual(result["mcnemar_exact_p_value"], 1.)
        self.assertEqual(result["accuracy_a"], .5)
        self.assertEqual(result["only_a_right"] + result["only_b_right"], 0)

    def test_one_sided_disagreements_are_counted_and_significant(self):
        truth = [(f"{i}.jpg", "Baroque") for i in range(30)]
        with tempfile.TemporaryDirectory() as temporary:
            a = write_run(Path(temporary) / "a", [(p, t, "Cubism") for p, t in truth])
            b = write_run(Path(temporary) / "b", [(p, t, t if i < 12 else "Cubism") for i, (p, t) in enumerate(truth)])
            result = mcnemar(a, b)
        self.assertEqual((result["only_a_right"], result["only_b_right"], result["both_wrong"]), (0, 12, 18))
        self.assertAlmostEqual(result["mcnemar_exact_p_value"], 2 * .5 ** 12)
        self.assertAlmostEqual(result["accuracy_b"], .4)

    def test_runs_on_different_test_images_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            a = write_run(Path(temporary) / "a", [("1.jpg", "Baroque", "Baroque")])
            b = write_run(Path(temporary) / "b", [("2.jpg", "Baroque", "Baroque")])
            with self.assertRaisesRegex(ValueError, "same test images"):
                mcnemar(a, b)


if __name__ == "__main__":
    unittest.main()
