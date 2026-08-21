"""Stage 2 - raw acquisition.

Idempotent by construction: the presence of ``out/raw/{request_id}.html``
is the unit of work, so an interrupted run resumes with no checkpoint
file at all. The checkpoint adds counters and periodic durability on top.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime
from typing import Any

from config import settings
from scraper.checkpoint import Checkpoint, CheckpointedWriter, utc_now
from scraper.context import PipelineContext, StageResult
from scraper.http import BlockedError, RateLimiter, build_client, fetch
from scraper.logging_conf import get_logger
from scraper.stages.s1_enumerate import load_requests

log = get_logger(__name__)

ARTIFACT = "02_fetch_log.jsonl"


class AbortRunError(RuntimeError):
    """Too many consecutive failures; stop the stage."""


class FetchState:
    """Shared mutable state across the worker coroutines."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.consecutive = 0
        self.fetched = 0
        self.cached = 0
        self.failed = 0
        self.stop = asyncio.Event()
        self.lock = asyncio.Lock()

    async def record_ok(self) -> None:
        """Reset the failure streak."""
        async with self.lock:
            self.consecutive = 0
            self.fetched += 1

    async def record_fail(self) -> None:
        """Count a failure and trip the breaker if the streak is long."""
        async with self.lock:
            self.consecutive += 1
            self.failed += 1
            if self.consecutive >= self.limit:
                self.stop.set()


async def _worker(
    descriptor: dict[str, Any],
    ctx: PipelineContext,
    client,
    limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    state: FetchState,
    writer: CheckpointedWriter,
    write_lock: asyncio.Lock,
) -> None:
    """Fetch one page unless it is already cached."""
    if state.stop.is_set():
        return

    request_id = descriptor["request_id"]
    raw_path = ctx.raw_dir / f"{request_id}.html"

    if raw_path.exists() and not ctx.force:
        async with write_lock:
            state.cached += 1
            if not writer.is_done(request_id):
                writer.write(
                    {
                        "request_id": request_id,
                        "url": descriptor["url"],
                        "effective_url": None,
                        "status": 200,
                        "elapsed_ms": 0,
                        "bytes": raw_path.stat().st_size,
                        "sha256": None,
                        "cached": True,
                        "error": None,
                        "fetched_at": utc_now().isoformat(),
                    }
                )
        return

    async with semaphore:
        if state.stop.is_set():
            return
        try:
            result = await fetch(client, descriptor["url"], limiter)
        except BlockedError:
            state.stop.set()
            raise
        except Exception as exc:  # noqa: BLE001 - logged, never fatal
            await state.record_fail()
            log.warning(
                "s2.fetch_failed",
                request_id=request_id,
                url=descriptor["url"],
                error=f"{type(exc).__name__}: {exc}",
            )
            async with write_lock:
                writer.write(
                    {
                        "request_id": request_id,
                        "url": descriptor["url"],
                        "effective_url": None,
                        "status": None,
                        "elapsed_ms": None,
                        "bytes": None,
                        "sha256": None,
                        "cached": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "fetched_at": utc_now().isoformat(),
                    }
                )
            return

    body = result.text.encode("utf-8", errors="replace")
    ctx.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(body)
    await state.record_ok()

    async with write_lock:
        writer.write(
            {
                "request_id": request_id,
                "url": descriptor["url"],
                "effective_url": result.effective_url,
                "status": result.status,
                "elapsed_ms": result.elapsed_ms,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "cached": False,
                "error": None,
                "fetched_at": utc_now().isoformat(),
            }
        )


async def _fetch_all(
    ctx: PipelineContext,
    descriptors: list[dict[str, Any]],
    writer: CheckpointedWriter,
) -> FetchState:
    """Drive the worker pool."""
    state = FetchState(settings.max_consecutive_failures)
    limiter = RateLimiter(settings.delay_min_s, settings.delay_max_s)
    semaphore = asyncio.Semaphore(settings.concurrency)
    write_lock = asyncio.Lock()

    async with build_client(rotate_ua=True) as client:
        tasks = [
            _worker(
                d, ctx, client, limiter, semaphore, state, writer, write_lock
            )
            for d in descriptors
        ]
        for coro in asyncio.as_completed(tasks):
            await coro

    if state.stop.is_set() and state.consecutive >= state.limit:
        raise AbortRunError(
            f"stopped after {state.consecutive} consecutive failures. "
            "Check connectivity, then rerun -- cached pages are kept so "
            "the stage resumes where it left off."
        )
    return state


def run(ctx: PipelineContext) -> StageResult:
    """Execute stage 2."""
    requests_path = ctx.path("01_requests.jsonl")
    descriptors = load_requests(requests_path)
    if not descriptors:
        raise RuntimeError(f"{requests_path} is empty -- run stage 1 first")

    checkpoint = Checkpoint(ctx.out_dir, "s2_fetch")
    out_path = ctx.path(ARTIFACT)
    if ctx.force:
        out_path.unlink(missing_ok=True)
        checkpoint.clear()

    started = datetime.now()
    with CheckpointedWriter(
        out_path, checkpoint, "request_id", settings.checkpoint_every
    ) as writer:
        pending = [
            d
            for d in descriptors
            if ctx.force or not writer.is_done(d["request_id"])
        ]
        log.info(
            "s2.start",
            total=len(descriptors),
            pending=len(pending),
            concurrency=settings.concurrency,
        )
        state = (
            asyncio.run(_fetch_all(ctx, pending, writer))
            if pending
            else FetchState(settings.max_consecutive_failures)
        )

    elapsed = (datetime.now() - started).total_seconds()
    print("\n--- stage 2: fetch -----------------------------------")
    print(f"  requests        : {len(descriptors)}")
    print(f"  newly fetched   : {state.fetched}")
    print(f"  already cached  : {state.cached}")
    print(f"  failed          : {state.failed}")
    print(f"  elapsed         : {elapsed:.1f}s")
    print("------------------------------------------------------\n")

    return StageResult(
        name="fetch",
        records=state.fetched + state.cached,
        notes={"failed": state.failed, "cached": state.cached},
    )
