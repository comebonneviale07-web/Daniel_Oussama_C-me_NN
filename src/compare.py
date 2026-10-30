"""Is one run really better than another on the same test paintings? Exact McNemar test.

    python main.py compare --run-a results/M1/v1_baseline/<run> --run-b results/M1/v2_tuned/<run>

Both runs must have predicted exactly the same test images. Only the paintings where the two runs disagree
(one right, the other wrong) carry information; under "no difference" each such painting is equally likely to
favour either run, so the p-value is a two-sided binomial test on those disagreements. With 225 test paintings
a gap of a few points is often not significant.
"""
import argparse
import csv
import json
from pathlib import Path

from scipy.stats import binomtest

from src.training import ROOT, write_json


def read_predictions(run):
    with (Path(run) / "test_predictions.csv").open(newline="", encoding="utf-8") as stream:
        return {row["path"]: (row["true_class"], row["predicted_class"]) for row in csv.DictReader(stream)}


def mcnemar(run_a, run_b):
    a, b = read_predictions(run_a), read_predictions(run_b)
    if set(a) != set(b):
        raise ValueError("The two runs were not evaluated on the same test images.")
    if any(a[p][0] != b[p][0] for p in a):
        raise ValueError("The two runs disagree on the true class of some test images.")
    table = {"both_right": 0, "only_a_right": 0, "only_b_right": 0, "both_wrong": 0}
    for path, (truth, predicted_a) in a.items():
        right_a, right_b = predicted_a == truth, b[path][1] == truth
        key = "both_right" if right_a and right_b else "only_a_right" if right_a else \
            "only_b_right" if right_b else "both_wrong"
        table[key] += 1
    disagreements = table["only_a_right"] + table["only_b_right"]
    p_value = 1. if disagreements == 0 else binomtest(table["only_a_right"], disagreements, .5).pvalue
    total = len(a)
    return {"run_a": Path(run_a).name, "run_b": Path(run_b).name, "test_images": total,
            "accuracy_a": (table["both_right"] + table["only_a_right"]) / total,
            "accuracy_b": (table["both_right"] + table["only_b_right"]) / total,
            **table, "mcnemar_exact_p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/comparisons")
    args = parser.parse_args()
    result = mcnemar(args.run_a, args.run_b)
    write_json(args.output / f"{result['run_a']}_vs_{result['run_b']}.json", result)
    print(json.dumps(result, indent=2))
    verdict = "significant" if result["mcnemar_exact_p_value"] < .05 else "not significant"
    print(f"Accuracy {result['accuracy_a']:.2%} vs {result['accuracy_b']:.2%}: difference {verdict} at 5 % "
          f"(p = {result['mcnemar_exact_p_value']:.3g}, {result['only_a_right']} vs {result['only_b_right']} "
          "disagreements).")


if __name__ == "__main__":
    main()
