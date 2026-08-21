"""Checkpoint, resume and pipeline-wiring tests. No network."""

from __future__ import annotations

import json

import pytest

from scraper.checkpoint import (
    Checkpoint,
    CheckpointedWriter,
    atomic_write_json,
    iter_jsonl,
    read_done_ids,
)
from scraper.context import PipelineContext
from scraper.pipeline import PipelineError, select_stages
from scraper.stages.s1_enumerate import (
    Shard,
    is_permitted,
    page_url,
    split_is_valid,
)
from scraper.stages.s5_consolidate import deduplicate


def make_ctx(tmp_path, **kwargs):
    defaults = dict(operation="arriendo", city="bogota", out_dir=tmp_path)
    defaults.update(kwargs)
    return PipelineContext(**defaults)


class TestAtomicWrite:
    def test_roundtrip(self, tmp_path):
        target = tmp_path / "a" / "b.json"
        atomic_write_json(target, {"x": 1})
        assert json.loads(target.read_text(encoding="utf-8")) == {"x": 1}

    def test_leaves_no_temp_files(self, tmp_path):
        atomic_write_json(tmp_path / "b.json", {"x": 1})
        assert not list(tmp_path.glob("*.tmp"))

    def test_overwrite_is_clean(self, tmp_path):
        target = tmp_path / "b.json"
        atomic_write_json(target, {"v": 1})
        atomic_write_json(target, {"v": 2})
        assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}


class TestIterJsonl:
    def test_skips_torn_final_line(self, tmp_path):
        # The signature of a process killed mid-write.
        path = tmp_path / "x.jsonl"
        path.write_text(
            '{"id": "1"}\n{"id": "2"}\n{"id": "3", "part',
            encoding="utf-8",
        )
        assert [r["id"] for r in iter_jsonl(path)] == ["1", "2"]

    def test_missing_file_is_empty(self, tmp_path):
        assert list(iter_jsonl(tmp_path / "nope.jsonl")) == []

    def test_read_done_ids(self, tmp_path):
        path = tmp_path / "x.jsonl"
        path.write_text('{"id": "a"}\n{"id": "b"}\n', encoding="utf-8")
        assert read_done_ids(path, "id") == {"a", "b"}


class TestCheckpointedWriter:
    def test_writes_and_counts(self, tmp_path):
        ckpt = Checkpoint(tmp_path, "s")
        path = tmp_path / "out.jsonl"
        with CheckpointedWriter(path, ckpt, "id", every=2) as writer:
            for i in range(5):
                writer.write({"id": str(i)})
        assert writer.written == 5
        assert len(list(iter_jsonl(path))) == 5

    def test_resume_skips_existing(self, tmp_path):
        ckpt = Checkpoint(tmp_path, "s")
        path = tmp_path / "out.jsonl"
        with CheckpointedWriter(path, ckpt, "id", every=2) as writer:
            for i in range(3):
                writer.write({"id": str(i)})

        # A second run must see the first run's ids as already done and
        # append only what is new.
        with CheckpointedWriter(path, ckpt, "id", every=2) as writer2:
            assert writer2.is_done("0")
            assert not writer2.is_done("9")
            for i in range(5):
                if not writer2.is_done(str(i)):
                    writer2.write({"id": str(i)})
        ids = [r["id"] for r in iter_jsonl(path)]
        assert sorted(ids) == ["0", "1", "2", "3", "4"]

    def test_resume_after_torn_write(self, tmp_path):
        path = tmp_path / "out.jsonl"
        path.write_text('{"id": "1"}\n{"id": "2"}\n{"id"', encoding="utf-8")
        ckpt = Checkpoint(tmp_path, "s")
        with CheckpointedWriter(path, ckpt, "id", every=2) as writer:
            assert writer.done_ids == {"1", "2"}

    def test_checkpoint_file_written(self, tmp_path):
        ckpt = Checkpoint(tmp_path, "stage_x")
        with CheckpointedWriter(
            tmp_path / "o.jsonl", ckpt, "id", every=1
        ) as writer:
            writer.write({"id": "1"})
        assert ckpt.path.exists()
        assert json.loads(ckpt.path.read_text(encoding="utf-8"))["stage"] \
            == "stage_x"


class TestStageSelection:
    def test_all(self):
        assert [s.number for s in select_stages("all")] == list(range(7))

    def test_single(self):
        assert [s.number for s in select_stages("3")] == [3]

    def test_range(self):
        assert [s.number for s in select_stages("1-4")] == [1, 2, 3, 4]

    def test_unknown_stage(self):
        with pytest.raises(ValueError):
            select_stages("9")


class TestRobotsCompliance:
    """robots.txt disallows the multi-value '-y-' filter URLs."""

    def test_rejects_forbidden_substring(self):
        assert not is_permitted(
            "https://x.co/arriendo/apartamentos-y-casas/bogota"
        )

    def test_allows_single_value_path(self):
        assert is_permitted(
            "https://x.co/arriendo/apartamentos/bogota/bogota-dc"
        )


class TestPageUrl:
    def test_page_one_has_no_suffix(self):
        assert page_url("https://x.co/arriendo/bogota", 1) == \
            "https://x.co/arriendo/bogota"

    def test_later_pages_get_suffix(self):
        assert page_url("https://x.co/arriendo/bogota", 7).endswith(
            "/pagina7"
        )


class TestSplitValidation:
    """A split that silently did not happen must be rejected.

    Some URL shapes answer 200 but redirect back to the unfiltered city
    page, returning the parent's whole result set.
    """

    def _check(self, child_total, effective, token="suba"):
        """Build a parent/child pair and run the real validator."""
        parent = Shard(kind="type", url="p", operation="arriendo",
                       city="bogota", total=7050, last_page=336)
        child = Shard(kind="type_neighbourhood", url="c",
                      operation="arriendo", city="bogota",
                      type_slug="apartamentos", neighbourhood_slug=token,
                      total=child_total, last_page=10)
        child.notes["effective_url"] = effective
        return split_is_valid(parent, child)

    def test_accepts_a_real_split(self):
        ok, why = self._check(
            879, "https://x.co/arriendo/apartamentos/suba/bogota"
        )
        assert ok, why

    def test_rejects_redirect_away_from_token(self):
        ok, why = self._check(
            500, "https://x.co/arriendo/apartamentos/bogota/bogota-dc"
        )
        assert not ok and "redirected" in why

    def test_rejects_when_count_did_not_shrink(self):
        ok, why = self._check(
            7050, "https://x.co/arriendo/apartamentos/suba/bogota"
        )
        assert not ok and "shrink" in why

    def test_rejects_empty_shard(self):
        ok, why = self._check(
            0, "https://x.co/arriendo/apartamentos/suba/bogota"
        )
        assert not ok and "empty" in why

    def test_rejects_missing_paginator(self):
        ok, why = self._check(
            None, "https://x.co/arriendo/apartamentos/suba/bogota"
        )
        assert not ok and "paginator" in why


class TestDeduplicate:
    def test_keeps_freshest(self):
        rows = [
            {"listing_id": "1", "updated_at": "2026-07-01", "v": "old"},
            {"listing_id": "1", "updated_at": "2026-08-01", "v": "new"},
            {"listing_id": "2", "updated_at": "2026-08-01", "v": "x"},
        ]
        out = {r["listing_id"]: r for r in deduplicate(rows)}
        assert len(out) == 2
        assert out["1"]["v"] == "new"

    def test_drops_rows_without_id(self):
        assert deduplicate([{"listing_id": ""}]) == []


class TestParamsGuard:
    def test_hash_changes_with_operation(self, tmp_path):
        a = make_ctx(tmp_path, operation="arriendo")
        b = make_ctx(tmp_path, operation="venta")
        assert a.params_hash() != b.params_hash()

    def test_hash_is_stable(self, tmp_path):
        assert make_ctx(tmp_path).params_hash() == \
            make_ctx(tmp_path).params_hash()

    def test_pipeline_refuses_mismatched_resume(self, tmp_path):
        from scraper.pipeline import _guard_params, _save_checkpoint, STAGES

        arriendo = make_ctx(tmp_path, operation="arriendo")
        _save_checkpoint(arriendo, STAGES[0])
        venta = make_ctx(tmp_path, operation="venta")
        with pytest.raises(PipelineError, match="arriendo"):
            _guard_params(venta, force=False)

    def test_force_overrides_the_guard(self, tmp_path):
        from scraper.pipeline import _guard_params, _save_checkpoint, STAGES

        _save_checkpoint(make_ctx(tmp_path, operation="arriendo"),
                         STAGES[0])
        _guard_params(make_ctx(tmp_path, operation="venta"), force=True)


class TestStageInputGuard:
    def test_missing_input_is_actionable(self, tmp_path):
        from scraper.pipeline import STAGES, _guard_inputs

        ctx = make_ctx(tmp_path)
        with pytest.raises(PipelineError, match="run --stage"):
            _guard_inputs(STAGES[3], ctx)
