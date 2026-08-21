"""Stage 1 - search-space enumeration with adaptive sharding.

Why this stage is not a simple ``range(1, last_page + 1)``:

The site sorts by ``Popularidad`` (relevance) and that ordering is not
stable, so deep pages return overlapping windows. Measured on
arriendo/bogota: pages 476 and 800 shared 12 of 21 listings, and 105 rows
pulled from five deep pages contained only 52 distinct listings -- while
every response was a healthy HTTP 200. Sorting cannot be forced (the
``orden`` query parameter and the base64 ``hashed`` filter are both
ignored server-side).

Shallow pages, however, are exact: pages 1-3 of a shard shared nothing at
all, and refetching a page returned byte-identical ids. So the fix is to
split the search until every shard is shallow enough to page through
reliably. Axes, in order: property type, then neighbourhood.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import settings
from scraper.checkpoint import (
    Checkpoint,
    CheckpointedWriter,
    atomic_write_json,
)
from scraper.context import PipelineContext, StageResult
from scraper.extract import dig, extract_next_data
from scraper.http import RateLimiter, build_client, fetch
from scraper.logging_conf import get_logger
from scraper.settings import config_section

log = get_logger(__name__)

ARTIFACT = "01_requests.jsonl"
PROBE_CACHE = "s1_probes.json"


@dataclass
class Shard:
    """One slice of the search space."""

    kind: str
    url: str
    operation: str
    city: str
    type_slug: str | None = None
    neighbourhood_slug: str | None = None
    total: int | None = None
    last_page: int | None = None
    coverage_at_risk: bool = False
    #: Emit at most this many pages (used by backstop shards).
    page_cap: int | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def shard_id(self) -> str:
        """Stable identifier derived from the URL."""
        return hashlib.sha1(self.url.encode()).hexdigest()[:12]

    @property
    def token(self) -> str:
        """The path fragment a valid split must preserve."""
        return self.neighbourhood_slug or self.type_slug or ""


def _base() -> str:
    return config_section("site")["base_url"]


def page_url(base_url: str, page: int) -> str:
    """Page 1 is the bare URL; later pages get the /paginaN suffix."""
    if page <= 1:
        return base_url
    suffix = config_section("urls")["page_suffix"].format(page=page)
    return base_url + suffix


def is_permitted(url: str) -> bool:
    """Reject URLs matching a robots.txt Disallow pattern.

    The site disallows ``/arriendo/*-y-*`` and ``/venta/*-y-*``, the
    multi-value filter form, so sharding must never combine two values
    into one path segment.
    """
    forbidden = config_section("urls").get("forbidden_substrings", [])
    return not any(bad in url for bad in forbidden)


def build_root(operation: str, city: str) -> Shard:
    """The unsharded starting point."""
    urls = config_section("urls")
    city_cfg = config_section("cities")[city]
    url = _base() + urls["root"].format(
        operation=config_section("operations")[operation]["slug"],
        city=city_cfg["slug"],
        state=city_cfg["state"],
    )
    return Shard(kind="root", url=url, operation=operation, city=city)


def split_by_type(parent: Shard) -> list[Shard]:
    """Axis 1: one shard per property type."""
    urls = config_section("urls")
    city_cfg = config_section("cities")[parent.city]
    op_slug = config_section("operations")[parent.operation]["slug"]
    out = []
    for meta in config_section("property_types").values():
        url = _base() + urls["by_type"].format(
            operation=op_slug,
            type_slug=meta["slug"],
            city=city_cfg["slug"],
            state=city_cfg["state"],
        )
        if not is_permitted(url):
            continue
        out.append(
            Shard(
                kind="type",
                url=url,
                operation=parent.operation,
                city=parent.city,
                type_slug=meta["slug"],
            )
        )
    return out


def split_by_neighbourhood(parent: Shard) -> list[Shard]:
    """Axis 2: one shard per neighbourhood, within a type."""
    urls = config_section("urls")
    city_cfg = config_section("cities")[parent.city]
    op_slug = config_section("operations")[parent.operation]["slug"]
    hoods = city_cfg.get("neighbourhoods") or []
    if not parent.type_slug or not hoods:
        return []
    out = []
    for hood in hoods:
        url = _base() + urls["by_type_neighbourhood"].format(
            operation=op_slug,
            type_slug=parent.type_slug,
            neighbourhood=hood,
            city=city_cfg["slug"],
        )
        if not is_permitted(url):
            continue
        out.append(
            Shard(
                kind="type_neighbourhood",
                url=url,
                operation=parent.operation,
                city=parent.city,
                type_slug=parent.type_slug,
                neighbourhood_slug=hood,
            )
        )
    return out


class Prober:
    """Fetches page 1 of a shard to learn its size, with a disk cache."""

    def __init__(self, ctx: PipelineContext, discovery: dict[str, Any]):
        self.ctx = ctx
        self.pointers = discovery["pointers"]
        self.keys = discovery["paginator_keys"]
        self.selector = discovery["script_selector"]
        self.cache_path = ctx.out_dir / ".checkpoint" / PROBE_CACHE
        self.cache: dict[str, Any] = {}
        if self.cache_path.exists() and not ctx.force:
            try:
                self.cache = json.loads(
                    self.cache_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                self.cache = {}

    def save(self) -> None:
        """Persist probe results so a resumed run re-fetches nothing."""
        atomic_write_json(self.cache_path, self.cache)

    async def probe(self, client, limiter, shard: Shard) -> None:
        """Populate ``total``/``last_page`` on the shard."""
        hit = self.cache.get(shard.url)
        if hit is not None:
            shard.total = hit.get("total")
            shard.last_page = hit.get("last_page")
            shard.notes["effective_url"] = hit.get("effective_url")
            return

        result = await fetch(client, shard.url, limiter)
        data = extract_next_data(result.text, self.selector)
        paginator = dig(data, self.pointers["paginator"], {}) or {}
        shard.total = paginator.get(self.keys["total"])
        shard.last_page = paginator.get(self.keys["last_page"])
        shard.notes["effective_url"] = result.effective_url

        self.cache[shard.url] = {
            "total": shard.total,
            "last_page": shard.last_page,
            "effective_url": result.effective_url,
        }
        self.save()


def split_is_valid(parent: Shard, child: Shard) -> tuple[bool, str]:
    """Guard against a split that silently did not happen.

    ``/arriendo/apartamentos/bogota/bogota-dc/chapinero`` answers HTTP 200
    but redirects back to the unfiltered city page and returns the parent's
    full result set. Without this check the pipeline would believe it had
    sharded and would quietly collect duplicates instead.
    """
    effective = (child.notes.get("effective_url") or "").lower()
    if child.total is None or child.last_page is None:
        return False, "no paginator"
    if child.token and child.token not in effective:
        return False, f"redirected away from {child.token!r}"
    if parent.total is not None and child.total >= parent.total:
        return False, "result count did not shrink"
    if child.total == 0:
        return False, "empty shard"
    return True, ""


async def _enumerate(ctx: PipelineContext) -> list[Shard]:
    """Breadth-first split until every shard is shallow enough."""
    discovery = json.loads(
        ctx.path("00_discovery.json").read_text(encoding="utf-8")
    )
    limiter = RateLimiter(settings.delay_min_s, settings.delay_max_s)
    prober = Prober(ctx, discovery)
    limit = settings.max_safe_page
    final: list[Shard] = []

    async with build_client() as client:
        root = build_root(ctx.operation, ctx.city)
        await prober.probe(client, limiter, root)
        log.info(
            "s1.root",
            total=root.total,
            last_page=root.last_page,
            max_safe_page=limit,
        )

        queue: list[Shard] = [root]
        while queue:
            parent = queue.pop(0)
            if (parent.last_page or 0) <= limit:
                final.append(parent)
                continue

            if parent.kind == "root":
                children = split_by_type(parent)
            elif parent.kind == "type":
                children = split_by_neighbourhood(parent)
            else:
                children = []

            if not children:
                parent.coverage_at_risk = True
                parent.notes["reason"] = "no further split axis available"
                log.warning(
                    "s1.unsplittable",
                    url=parent.url,
                    last_page=parent.last_page,
                )
                final.append(parent)
                continue

            kept: list[Shard] = []
            for child in children:
                await prober.probe(client, limiter, child)
                ok, why = split_is_valid(parent, child)
                if ok:
                    kept.append(child)
                else:
                    log.info("s1.drop_shard", url=child.url, reason=why)

            covered = sum(c.total or 0 for c in kept)
            if not kept:
                parent.coverage_at_risk = True
                parent.notes["reason"] = "every child split was rejected"
                final.append(parent)
                continue

            # A split can lose listings: on arriendo/bogota roughly 30%
            # of inventory sits in barrios that are not one of the
            # seeded localidades, so no child shard reaches them. The
            # children are still used, and a "backstop" over the parent's
            # reliable shallow pages is added alongside them to recover
            # what they miss. Stage 5 deduplicates the overlap.
            if parent.total and covered < parent.total * 0.9:
                backstop = Shard(
                    kind=parent.kind,
                    url=parent.url,
                    operation=parent.operation,
                    city=parent.city,
                    type_slug=parent.type_slug,
                    neighbourhood_slug=parent.neighbourhood_slug,
                    total=parent.total,
                    last_page=parent.last_page,
                    coverage_at_risk=True,
                    page_cap=limit,
                    notes={
                        "role": "backstop",
                        "reason": f"children covered {covered}/{parent.total}",
                        "effective_url": parent.notes.get("effective_url"),
                    },
                )
                final.append(backstop)
                log.warning(
                    "s1.lossy_split",
                    url=parent.url,
                    parent_total=parent.total,
                    covered=covered,
                    backstop_pages=limit,
                )
            queue.extend(kept)

    prober.save()
    return final


def run(ctx: PipelineContext) -> StageResult:
    """Execute stage 1."""
    out_path = ctx.path(ARTIFACT)
    checkpoint = Checkpoint(ctx.out_dir, "s1_enumerate")
    if ctx.force:
        out_path.unlink(missing_ok=True)
        checkpoint.clear()
    elif out_path.exists():
        count = sum(1 for _ in out_path.open(encoding="utf-8"))
        log.info("stage1.skip", reason="artifact exists", requests=count)
        return StageResult(name="enumerate", records=count, skipped=True)

    shards = asyncio.run(_enumerate(ctx))

    lo = ctx.page_from or 1
    cap = settings.max_pages
    at_risk = 0
    with CheckpointedWriter(
        out_path,
        checkpoint,
        "request_id",
        settings.checkpoint_every,
        resume=False,
    ) as writer:
        for shard in shards:
            last = shard.last_page or 1
            if shard.page_cap:
                last = min(last, shard.page_cap)
            hi = min(last, ctx.page_to or last)
            if cap:
                hi = min(hi, lo + cap - 1)
            if shard.coverage_at_risk:
                at_risk += 1
            role = shard.notes.get("role", "primary")
            for page in range(lo, hi + 1):
                url = page_url(shard.url, page)
                if not is_permitted(url):
                    continue
                rid = hashlib.sha1(url.encode()).hexdigest()[:16]
                writer.write(
                    {
                        "request_id": rid,
                        "url": url,
                        "page": page,
                        "shard_id": shard.shard_id,
                        "shard_kind": shard.kind,
                        "shard_role": role,
                        "operation": shard.operation,
                        "city": shard.city,
                        "property_type_slug": shard.type_slug,
                        "neighbourhood_slug": shard.neighbourhood_slug,
                        "shard_total": shard.total,
                        "shard_last_page": shard.last_page,
                        "coverage_at_risk": shard.coverage_at_risk,
                    }
                )
        total_requests = writer.written

    atomic_write_json(
        ctx.path("01_shards.json"),
        [asdict(s) | {"shard_id": s.shard_id} for s in shards],
    )

    # Backstops re-cover their siblings' ground, so they are excluded
    # from the expected-row estimate to avoid double counting.
    primary = [s for s in shards if s.notes.get("role") != "backstop"]
    backstops = len(shards) - len(primary)
    expected = sum(s.total or 0 for s in primary)

    def emitted_pages(shard: Shard) -> int:
        pages = shard.last_page or 1
        return min(pages, shard.page_cap) if shard.page_cap else pages

    deepest = max((emitted_pages(s) for s in shards), default=0)
    print("\n--- stage 1: enumeration -----------------------------")
    print(f"  shards          : {len(primary)} (+{backstops} backstop)")
    print(f"  page requests   : {total_requests}")
    print(f"  expected rows   : ~{expected}")
    print(
        f"  deepest shard   : {deepest} pages emitted "
        f"(safe limit {settings.max_safe_page})"
    )
    print(f"  coverage at risk: {at_risk} shard(s)")
    print("------------------------------------------------------\n")

    return StageResult(
        name="enumerate",
        records=total_requests,
        notes={"shards": len(shards), "expected_rows": expected},
    )


def load_requests(path: Path) -> list[dict[str, Any]]:
    """Read ``01_requests.jsonl`` (used by stage 2)."""
    from scraper.checkpoint import iter_jsonl

    return list(iter_jsonl(path))
