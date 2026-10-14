"""Build a clean zip of this project for uploading to a hosted Colab or Kaggle session.

The repository is not (yet) under version control, so there is nothing to clone. This
script exists so the upload step is repeatable and so the archive can be inspected
before it is sent anywhere: it prints every file it includes, and it excludes caches
and build artefacts that would otherwise be stale on the remote machine.

Usage:
    python scripts/make_colab_bundle.py [--output PATH]

Default output is TinyTurbo.zip in the project root, beside the rest of the repository,
so there is one predictable place to pick it up from. The archive never contains itself:
.zip files are excluded, as are caches and build artefacts.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directories that must never travel: caches are machine-specific and an editable
# install's .egg-info describes a path that does not exist on the remote machine.
EXCLUDED_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
}

EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}
EXCLUDED_NAMES = {"tinyturbo.egg-info"}


def included_files() -> list[Path]:
    """Every project file worth uploading, in a stable order."""
    files: list[Path] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in EXCLUDED_DIRECTORIES or part in EXCLUDED_NAMES for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(relative)
    return files


def build_bundle(output: Path) -> tuple[Path, list[Path]]:
    """Write the archive and return its path plus the manifest of what went in."""
    files = included_files()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            # Files are stored under a top-level folder so unzipping into /content
            # produces /content/TinyTurbo regardless of the local folder's name. The
            # local folder is called "TinyTurbo Project", and the space would have to be
            # quoted in every remote shell command.
            archive.write(PROJECT_ROOT / relative, Path("TinyTurbo") / relative)
    return output, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "TinyTurbo.zip",
        help="where to write the archive (default: the project root)",
    )
    args = parser.parse_args()

    output, files = build_bundle(args.output)
    size_mb = output.stat().st_size / (1024 * 1024)

    print(f"bundle: {output}")
    print(f"size  : {size_mb:.2f} MB")
    print(f"files : {len(files)}")
    for relative in files:
        print(f"  {relative.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
