"""Command line interface.

Must be invoked from the repository root so ``from config import
settings`` resolves against the project's single settings object.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from config import settings
from scraper import __version__
from scraper.context import PipelineContext
from scraper.logging_conf import configure_logging, get_logger
from scraper.pipeline import PipelineError, run_pipeline, select_stages

app = typer.Typer(
    add_completion=False,
    help="FincaRaiz pipeline scraper.",
)
log = get_logger(__name__)


def _parse_pages(spec: str | None) -> tuple[int | None, int | None]:
    """Parse ``--pages 1-20`` or ``--pages 5``."""
    if not spec:
        return None, None
    text = spec.strip()
    if "-" in text:
        lo, _, hi = text.partition("-")
        return int(lo), int(hi)
    return 1, int(text)


@app.command()
def run(
    stage: str = typer.Option(
        "all", help="Stage to run: all, a number 0-6, or a range like 1-4."
    ),
    operation: str = typer.Option("arriendo", help="arriendo or venta."),
    city: str = typer.Option("bogota", help="City key from config.yaml."),
    pages: str = typer.Option(
        None, help="Page window per shard, e.g. 1-20. Bounds a dev run."
    ),
    out: Path = typer.Option(
        None, help="Output directory (default: FR_OUT_DIR or ./out)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Ignore caches and recompute everything."
    ),
    log_level: str = typer.Option("INFO", help="DEBUG, INFO, WARNING."),
    json_logs: bool = typer.Option(
        False, "--json-logs", help="Force JSON logs even on a terminal."
    ),
) -> None:
    """Run the pipeline, or one stage of it."""
    configure_logging(log_level, force_json=json_logs)

    if operation not in ("arriendo", "venta"):
        raise typer.BadParameter("operation must be arriendo or venta")

    page_from, page_to = _parse_pages(pages)
    ctx = PipelineContext(
        operation=operation,
        city=city,
        out_dir=out or settings.out_dir,
        force=force,
        page_from=page_from,
        page_to=page_to,
    )

    try:
        stages = select_stages(stage)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    log.info(
        "run.start",
        version=__version__,
        operation=operation,
        city=city,
        stages=[s.number for s in stages],
        out=str(ctx.out_dir),
        concurrency=settings.concurrency,
    )

    try:
        run_pipeline(ctx, stages, force=force)
    except PipelineError as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    except KeyboardInterrupt:
        typer.secho(
            "\nInterrupted. Progress is checkpointed -- rerun the same "
            "command to resume.\n",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=130)


@app.command()
def version() -> None:
    """Print the scraper version."""
    typer.echo(__version__)


def main() -> None:
    """Entry point for ``python -m scraper``."""
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    app()
