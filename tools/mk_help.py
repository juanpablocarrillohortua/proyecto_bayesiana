"""Print the self-documenting target list for a Makefile.

Called by `make help`. Replaces the usual awk one-liner, which is not
available under cmd.exe on Windows. Every target whose rule line carries a
`## description` comment is listed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TARGET_RE = re.compile(r"^([A-Za-z][\w.\-]*)\s*:[^=]*?##\s*(.+)$")

CYAN = "\033[36m"
RESET = "\033[0m"


def collect(paths: list[str]) -> dict[str, str]:
    targets: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            match = TARGET_RE.match(line)
            if match:
                targets[match.group(1)] = match.group(2).strip()
    return targets


def main(argv: list[str]) -> int:
    paths = argv[1:] or ["Makefile"]
    targets = collect(paths)
    if not targets:
        print("No documented targets found.")
        return 0

    color = sys.stdout.isatty()
    width = max(len(name) for name in targets)

    print("Please use `make <target>' where <target> is one of")
    for name in sorted(targets):
        label = f"{CYAN}{name:<{width}}{RESET}" if color else f"{name:<{width}}"
        print(f"  {label}  {targets[name]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
