"""The pipeline consolidator.

This is the single place that knows the stage order. Running the whole
thing is one command::

    make scrape
    python -m scraper run --stage all --operation arriendo --city bogota

and an individual stage is the same code path with a filtered list::

    python -m scraper run --stage 3

On top of the record-level checkpoints each stage keeps, the pipeline
keeps its own: a stage whose output artifact already exists is skipped,
and ``out/.checkpoint/pipeline.json`` records how far the run got.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from scraper.checkpoint import atomic_write_json, utc_now
from scraper.context import PipelineContext, StageResult
from scraper.logging_conf import get_logger
from scraper.stages import (
    s0_discovery,
    s1_enumerate,
    s2_fetch,
    s3_parse,
    s4_normalize,
    s5_consolidate,
    s6_qa,
)

log = get_logger(__name__)

CHECKPOINT = "pipeline.json"


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline."""

    number: int
    name: str
    run: Callable[[PipelineContext], StageResult]
    #: Artifact this stage produces; its presence means "already done".
    produces: str | None = None
    #: Artifact the previous stage must have left behind.
    requires: str | None = None
    #: Override for stages whose filename is not fixed.
    exists: Callable[[PipelineContext], bool] | None = None

    def is_done(self, ctx: PipelineContext) -> bool:
        """Whether this stage's output is already on disk."""
        if self.exists is not None:
            return self.exists(ctx)
        if self.produces is None:
            return False
        path = ctx.path(self.produces)
        return path.exists() and path.stat().st_size > 0


STAGES: tuple[Stage, ...] = (
    Stage(0, "discovery", s0_discovery.run, produces="00_discovery.json"),
    Stage(
        1,
        "enumerate",
        s1_enumerate.run,
        produces="01_requests.jsonl",
        requires="00_discovery.json",
    ),
    Stage(
        2,
        "fetch",
        s2_fetch.run,
        produces="02_fetch_log.jsonl",
        requires="01_requests.jsonl",
    ),
    Stage(
        3,
        "parse",
        s3_parse.run,
        produces="03_records_raw.jsonl",
        requires="02_fetch_log.jsonl",
    ),
    Stage(
        4,
        "normalize",
        s4_normalize.run,
        produces="04_records_clean.jsonl",
        requires="03_records_raw.jsonl",
    ),
    Stage(
        5,
        "consolidate",
        s5_consolidate.run,
        requires="04_records_clean.jsonl",
        exists=s5_consolidate.artifact_exists,
    ),
    Stage(6, "qa", s6_qa.run, produces="06_qa_report.json"),
)


class PipelineError(RuntimeError):
    """A stage could not run because its inputs are missing."""


def select_stages(spec: str) -> list[Stage]:
    """Turn ``"all"``, ``"3"`` or ``"1-4"`` into a stage list."""
    text = (spec or "all").strip().lower()
    if text == "all":
        return list(STAGES)
    if "-" in text:
        lo_text, _, hi_text = text.partition("-")
        lo, hi = int(lo_text), int(hi_text)
        return [s for s in STAGES if lo <= s.number <= hi]
    wanted = int(text)
    chosen = [s for s in STAGES if s.number == wanted]
    if not chosen:
        raise ValueError(f"no stage numbered {wanted}")
    return chosen


def _guard_inputs(stage: Stage, ctx: PipelineContext) -> None:
    """Fail with an actionable message rather than a deep traceback."""
    if stage.requires is None:
        return
    path = ctx.path(stage.requires)
    if not path.exists() or path.stat().st_size == 0:
        previous = stage.number - 1
        raise PipelineError(
            f"stage {stage.number} ({stage.name}) needs {path} -- "
            f"run --stage {previous} first"
        )


def _load_checkpoint(ctx: PipelineContext) -> dict:
    path = ctx.out_dir / ".checkpoint" / CHECKPOINT
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_checkpoint(ctx: PipelineContext, stage: Stage) -> None:
    atomic_write_json(
        ctx.out_dir / ".checkpoint" / CHECKPOINT,
        {
            "last_completed_stage": stage.number,
            "stage_name": stage.name,
            "params_hash": ctx.params_hash(),
            "operation": ctx.operation,
            "city": ctx.city,
            "updated_at": utc_now().isoformat(),
        },
    )


def _guard_params(ctx: PipelineContext, force: bool) -> None:
    """Refuse to blend artifacts from two different queries.

    Without this, switching ``--operation`` mid-directory would append
    venta listings onto an arriendo run and silently corrupt the output.
    """
    previous = _load_checkpoint(ctx)
    old_hash = previous.get("params_hash")
    if not old_hash or old_hash == ctx.params_hash() or force:
        return
    raise PipelineError(
        f"{ctx.out_dir} holds artifacts for "
        f"{previous.get('operation')}/{previous.get('city')} but this run "
        f"is {ctx.operation}/{ctx.city}. Use a different --out directory, "
        "or pass --force to overwrite."
    )


def _format_row(
    stage: Stage, status: str, records: int | str, elapsed: float
) -> str:
    return (
        f"{stage.number:<6} {stage.name:<13} {status:<9} "
        f"{records:>9} {elapsed:>10.1f}s"
    )


def run_pipeline(
    ctx: PipelineContext,
    stages: Sequence[Stage] | None = None,
    force: bool = False,
) -> list[tuple[Stage, StageResult, float]]:
    """Run the selected stages in order and print a summary table."""
    chosen = list(stages if stages is not None else STAGES)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    _guard_params(ctx, force)

    results: list[tuple[Stage, StageResult, float]] = []
    rows: list[str] = []
    total_elapsed = 0.0

    for stage in chosen:
        if stage.is_done(ctx) and not force:
            log.info("pipeline.skip", stage=stage.number, name=stage.name)
            rows.append(_format_row(stage, "skip", "-", 0.0))
            results.append((stage, StageResult(stage.name, skipped=True), 0.0))
            continue

        _guard_inputs(stage, ctx)
        log.info("pipeline.stage.start", stage=stage.number, name=stage.name)
        started = time.perf_counter()
        try:
            result = stage.run(ctx)
        except Exception:
            elapsed = time.perf_counter() - started
            rows.append(_format_row(stage, "FAILED", "-", elapsed))
            _print_summary(rows, total_elapsed + elapsed)
            log.error(
                "pipeline.stage.failed", stage=stage.number, name=stage.name
            )
            raise
        elapsed = time.perf_counter() - started
        total_elapsed += elapsed

        _save_checkpoint(ctx, stage)
        rows.append(_format_row(stage, "ok", result.records, elapsed))
        results.append((stage, result, elapsed))
        log.info(
            "pipeline.stage.done",
            stage=stage.number,
            name=stage.name,
            records=result.records,
            elapsed_s=round(elapsed, 2),
        )

    _print_summary(rows, total_elapsed)
    return results


def _print_summary(rows: list[str], total: float) -> None:
    """Render the end-of-run table."""
    print("=" * 56)
    print(
        f"{'stage':<6} {'name':<13} {'status':<9} "
        f"{'records':>9} {'elapsed':>11}"
    )
    print("-" * 56)
    for row in rows:
        print(row)
    print("-" * 56)
    minutes, seconds = divmod(total, 60)
    stamp = (
        f"{int(minutes)}m {seconds:04.1f}s" if minutes else f"{seconds:.1f}s"
    )
    print(f"{'':<30} {'total':>9} {stamp:>15}")
    print("=" * 56)
