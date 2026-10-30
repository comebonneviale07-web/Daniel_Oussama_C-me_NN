"""Documented model versions; IDs remain compatible with existing run checkpoints."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSIONS = {
    "m0": "M0/v1_logistic",
    "m1": "M1/v1_baseline",
    "m2": "M2/v1_sigmoid",
    "m2_fixed": "M2/v2_relu",
    "m3_frozen": "M3/v1_frozen",
    "m3": "M3/v2_finetuned",
}


def get_version(name):
    config_path = ROOT / "configs" / (MODEL_VERSIONS[name] + ".json")
    folder = config_path.parent
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    if not (folder / "README.md").read_text(encoding="utf-8").strip():
        raise ValueError(f"Model version {name} must have an explanatory README")
    if spec["id"] != name:
        raise ValueError(f"Wrong model ID in {folder}")
    return spec


def write_run_readme(run, config, metrics=None, selected_epoch=None):
    """Snapshot the version explanation and actual settings alongside each experiment."""
    name = config["model"]
    folder = (ROOT / "configs" / MODEL_VERSIONS[name]).parent
    explanation = (folder / "README.md").read_text(encoding="utf-8")
    lines = [f"# Entraînement {run.name}", "", f"Version : `{MODEL_VERSIONS[name]}` ; identifiant : `{name}`.",
             "", "## Réglages de ce run", "", "```json",
             json.dumps(config, indent=2, ensure_ascii=False), "```", ""]
    if metrics is None:
        lines += ["Statut : créé. Le fichier test_metrics.json confirme une évaluation terminée.", ""]
    else:
        lines += ["## Résultats", "", f"Accuracy test : {metrics['accuracy']:.2%}.",
                  f"Macro F1 test : {metrics['macro_f1']:.4f}."]
        if selected_epoch is not None:
            lines += [f"Checkpoint choisi sur la perte de validation : époque {selected_epoch}."]
        lines += ["", "Voir history.json, test_metrics.json et test_predictions.csv pour les mesures détaillées.", ""]
    lines += ["## Fiche de version (copie au moment de la documentation)", "", explanation]
    (run / "README.md").write_text("\n".join(lines), encoding="utf-8")
