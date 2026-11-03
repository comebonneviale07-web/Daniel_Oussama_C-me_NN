"""Create and verify a local backup of source, splits, reports and trained weights."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [p for p in ROOT.iterdir() if p.is_file() and
             (p.suffix in (".py", ".md", ".txt") or p.name == ".gitignore")]
    if (ROOT / "data/style_descriptions.json").is_file():
        files.append(ROOT / "data/style_descriptions.json")
    for folder in ("results", "data/splits", "docs", "configs", "tests", "src", ".streamlit"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".pyc")
    destination = ROOT / "archives/backups"
    destination.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = destination / f"museum-progress-{stamp}.zip"
    manifest = {}
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=1) as output:
        for path in sorted(files):
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            manifest[relative] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            output.writestr(relative, content)
        output.writestr("BACKUP_MANIFEST.json", json.dumps(manifest, indent=2))
    # Verify every archived byte against the source snapshot, including checkpoints.
    with ZipFile(archive) as saved:
        for relative, metadata in manifest.items():
            if hashlib.sha256(saved.read(relative)).hexdigest() != metadata["sha256"]:
                raise RuntimeError(f"Backup verification failed: {relative}")
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    print(f"Verified backup: {archive}\nFiles: {len(manifest)}; size: {archive.stat().st_size / 1024**2:.1f} MiB")
    print("Copy the ZIP and checksum to another device or cloud storage for an independent backup.")


if __name__ == "__main__":
    main()
