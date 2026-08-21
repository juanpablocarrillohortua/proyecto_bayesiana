"""Loader for ``scraper/config.yaml``.

This module deliberately holds no environment logic. Deployment tunables
live in the project root ``config.py`` so there is exactly one
``Settings`` object in the project; this file only exposes the static,
site-describing configuration.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read and cache ``config.yaml``."""
    target = path or CONFIG_PATH
    with target.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{target} did not parse to a mapping")
    return data


def config_section(name: str) -> Any:
    """Return one top-level section, failing loudly if it is absent."""
    cfg = load_config()
    if name not in cfg:
        raise KeyError(
            f"section {name!r} missing from {CONFIG_PATH.name}; "
            "the config file is out of date with the code"
        )
    return cfg[name]
