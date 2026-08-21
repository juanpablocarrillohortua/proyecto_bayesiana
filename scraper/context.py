"""The object threaded through every pipeline stage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from scraper.checkpoint import utc_now


@dataclass
class PipelineContext:
    """Shared run parameters and paths.

    Stages receive this and return a ``StageResult``; they never talk to
    each other directly.
    """

    operation: str
    city: str
    out_dir: Path
    force: bool = False
    #: Inclusive 1-based page window, or None for "whatever the shard has".
    page_from: int | None = None
    page_to: int | None = None
    started_at: datetime = field(default_factory=utc_now)
    extra: dict[str, Any] = field(default_factory=dict)

    def path(self, name: str) -> Path:
        """Resolve an artifact name inside the output directory."""
        return self.out_dir / name

    @property
    def raw_dir(self) -> Path:
        """Directory holding cached raw responses."""
        return self.out_dir / "raw"

    def query(self) -> dict[str, Any]:
        """The query block recorded in the final artifact metadata."""
        filters: dict[str, Any] = {}
        if self.page_from is not None or self.page_to is not None:
            filters["pages"] = f"{self.page_from}-{self.page_to}"
        return {
            "operation": self.operation,
            "city": self.city,
            "filters": filters,
        }

    def params_hash(self) -> str:
        """Stable digest of the parameters that shape the artifacts.

        The pipeline refuses to resume across a change in this value so
        two different queries can never blend into one output file.
        """
        payload = json.dumps(
            {
                "operation": self.operation,
                "city": self.city,
                "page_from": self.page_from,
                "page_to": self.page_to,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class StageResult:
    """What a stage reports back to the pipeline runner."""

    name: str
    records: int = 0
    skipped: bool = False
    notes: dict[str, Any] = field(default_factory=dict)
