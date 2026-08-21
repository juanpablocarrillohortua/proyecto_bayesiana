"""Remove byte code, tool caches and notebook checkpoints.

Called by `make clean`. Written in Python rather than as a shell recipe so
the Makefile works under cmd.exe, Git Bash, macOS and Linux alike.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories never walked into.
PRUNED = {".git", ".venv", "venv", "node_modules"}

# Directories removed wholesale wherever they are found.
CACHE_DIRS = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
}

# File suffixes removed wherever they are found.
JUNK_SUFFIXES = (".pyc", ".pyo")


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _walk(directory: Path, removed: list[Path]) -> None:
    """Depth-first walk, deleting caches and junk files as they appear."""
    try:
        entries = sorted(directory.iterdir())
    except (PermissionError, FileNotFoundError):
        return

    for entry in entries:
        if entry.is_dir():
            if entry.name in PRUNED:
                continue
            if entry.name in CACHE_DIRS:
                shutil.rmtree(entry, ignore_errors=True)
                removed.append(entry)
                continue
            _walk(entry, removed)
        elif entry.suffix in JUNK_SUFFIXES or entry.name.endswith("~"):
            entry.unlink(missing_ok=True)
            removed.append(entry)


def main() -> int:
    removed: list[Path] = []
    _walk(ROOT, removed)

    for path in removed:
        print(f"removed {_relative(path)}")
    print(f"clean: {len(removed)} item(s) removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
