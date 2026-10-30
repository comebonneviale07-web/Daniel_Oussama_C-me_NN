"""Add version explanations to historical classification runs without changing their results."""
import json

from src.versions import ROOT, MODEL_VERSIONS, write_run_readme


def main():
    documented = 0
    for path in sorted((ROOT / "results").rglob("config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("model") not in MODEL_VERSIONS or (path.parent / "README.md").exists():
            continue
        metrics_path = path.parent / "test_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
        # Metadata-only: never load or overwrite old checkpoint binaries.
        write_run_readme(path.parent, config, metrics)
        note = "\nDocumentation ajoutée rétrospectivement lors du rangement ; mesures originales inchangées.\n"
        with (path.parent / "README.md").open("a", encoding="utf-8") as output:
            output.write(note)
        documented += 1
    print(f"Documented {documented} classification runs; existing README files preserved.")


if __name__ == "__main__":
    main()
