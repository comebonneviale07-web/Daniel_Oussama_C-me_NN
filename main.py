"""Single project entry point: python main.py <command> [options]."""
import argparse
import importlib
import sys

COMMANDS = {
    "demo": "src.demo",
    "train": "src.training",
    "predict": "src.predict",
    "benchmark": "src.benchmark",
    "diagnose": "src.diagnose",
    "dataset": "src.dataset",
    "backup": "src.backup",
    "document-runs": "src.document_runs",
    "compare": "src.compare",
    "robustness": "src.robustness",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [f"main.py {args.command}", *sys.argv[2:]]
    importlib.import_module(COMMANDS[args.command]).main()


if __name__ == "__main__":
    main()
