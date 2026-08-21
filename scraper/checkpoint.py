"""Resume and checkpoint helpers shared by every stage.

Design note: the ``.jsonl`` output file -- not the checkpoint file -- is
the source of truth for what is already done. A process can die between
flushing records and updating its checkpoint, so on resume we rebuild the
done-set by streaming the output and discarding a torn final line. The
checkpoint file then only carries counters and timing, which are cheap to
lose.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> datetime:
    """Timezone-aware current time."""
    return datetime.now(timezone.utc)


def _replace_with_retry(tmp: str, path: Path, attempts: int = 6) -> None:
    """``os.replace`` with backoff on Windows PermissionError.

    On Windows a just-created file can be held briefly by antivirus or
    the search indexer, and the rename then fails with WinError 5 even
    though nothing is wrong. Observed intermittently during checkpoint
    writes, so the rename is retried rather than aborting a long run.
    """
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (2**attempt))


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via a temp file + rename so readers never see a
    half-written document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, ensure_ascii=False, indent=2, default=str
            )
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a ``.jsonl`` file, tolerating a torn last line.

    A truncated final line is the normal signature of a killed process,
    so it is skipped rather than raising.
    """
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Torn trailing write from an interrupted run.
                continue


def read_done_ids(path: Path, id_field: str) -> set[str]:
    """Rebuild the set of already-processed ids from a stage output."""
    return {str(rec[id_field]) for rec in iter_jsonl(path) if id_field in rec}


class Checkpoint:
    """Per-stage checkpoint file holding counters and progress."""

    def __init__(self, out_dir: Path, stage: str) -> None:
        self.path = out_dir / ".checkpoint" / f"{stage}.json"
        self.stage = stage
        self.data: dict[str, Any] = {
            "stage": stage,
            "counters": {},
            "processed": 0,
            "updated_at": None,
        }
        if self.path.exists():
            try:
                self.data.update(
                    json.loads(self.path.read_text(encoding="utf-8"))
                )
            except json.JSONDecodeError:
                pass

    def bump(self, key: str, amount: int = 1) -> None:
        """Increment a named counter."""
        counters = self.data.setdefault("counters", {})
        counters[key] = counters.get(key, 0) + amount

    def save(self, **extra: Any) -> None:
        """Persist the checkpoint atomically."""
        self.data.update(extra)
        self.data["updated_at"] = utc_now().isoformat()
        atomic_write_json(self.path, self.data)

    def clear(self) -> None:
        """Drop the checkpoint (used by ``--force``)."""
        self.path.unlink(missing_ok=True)


class CheckpointedWriter:
    """Append-only ``.jsonl`` writer that flushes and checkpoints
    periodically, so an interrupted stage loses at most
    ``every`` records.

    Usage::

        with CheckpointedWriter(path, ckpt, "listing_id", 25) as writer:
            for rec in records:
                if writer.is_done(rec["listing_id"]):
                    continue
                writer.write(rec)
    """

    def __init__(
        self,
        path: Path,
        checkpoint: Checkpoint,
        id_field: str,
        every: int = 25,
        resume: bool = True,
    ) -> None:
        self.path = path
        self.checkpoint = checkpoint
        self.id_field = id_field
        self.every = max(1, every)
        self.written = 0
        self._since_flush = 0
        self.done_ids: set[str] = (
            read_done_ids(path, id_field) if resume else set()
        )
        self._handle = None

    def __enter__(self) -> CheckpointedWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.done_ids else "w"
        self._handle = self.path.open(mode, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()
        if self._handle is not None:
            self._handle.close()
        self.checkpoint.save(processed=len(self.done_ids))

    def is_done(self, record_id: Any) -> bool:
        """True when this id already exists in the output."""
        return str(record_id) in self.done_ids

    def write(self, record: dict[str, Any]) -> None:
        """Append one record, flushing on the checkpoint interval."""
        if self._handle is None:
            raise RuntimeError("writer used outside its context manager")
        self._handle.write(
            json.dumps(record, ensure_ascii=False, default=str) + "\n"
        )
        self.done_ids.add(str(record.get(self.id_field)))
        self.written += 1
        self._since_flush += 1
        if self._since_flush >= self.every:
            self.flush()

    def flush(self) -> None:
        """Force data to disk and update the checkpoint."""
        if self._handle is None or self._since_flush == 0:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._since_flush = 0
        self.checkpoint.save(processed=len(self.done_ids))
