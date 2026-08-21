"""Stage 5 - deduplicate, validate and emit the final artifact."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from typing import Any

from pydantic import ValidationError

from scraper import __version__
from scraper.checkpoint import atomic_write_json, iter_jsonl, utc_now
from scraper.context import PipelineContext, StageResult
from scraper.logging_conf import get_logger
from scraper.models import FinalArtifact, Listing, RunMetadata, RunStats

log = get_logger(__name__)

REJECTS = "05_rejected.jsonl"

CSV_COLUMNS = [
    "listing_id",
    "operation",
    "property_type",
    "price_amount",
    "price_currency",
    "admin_fee",
    "total_monthly_cost",
    "price_per_m2",
    "area_built_m2",
    "area_private_m2",
    "bedrooms",
    "bathrooms",
    "parking_spaces",
    "stratum",
    "floor",
    "floors_total",
    "age_bracket_code",
    "age_bracket_label",
    "construction_state",
    "city",
    "locality",
    "neighborhood",
    "address_text",
    "latitude",
    "longitude",
    "amenities",
    "agency_name",
    "publisher_type",
    "published_at",
    "updated_at",
    "source_url",
]


def artifact_name(ctx: PipelineContext, suffix: str) -> str:
    """``fincaraiz_{operation}_{city}_{YYYYMMDD}.{suffix}``."""
    stamp = utc_now().strftime("%Y%m%d")
    return f"fincaraiz_{ctx.operation}_{ctx.city}_{stamp}.{suffix}"


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one record per listing id, preferring the freshest.

    Duplicates are expected: overlapping shards and the site's unstable
    relevance ordering both produce them.
    """
    best: dict[str, dict[str, Any]] = {}
    for rec in records:
        key = str(rec.get("listing_id") or "")
        if not key:
            continue
        current = best.get(key)
        if current is None:
            best[key] = rec
            continue
        if str(rec.get("updated_at") or "") > str(
            current.get("updated_at") or ""
        ):
            best[key] = rec
    return list(best.values())


def compute_stats(listings: list[Listing]) -> RunStats:
    """Price summary plus counts by type and neighbourhood."""
    prices = [
        listing.price_amount
        for listing in listings
        if listing.price_amount and listing.price_amount > 0
    ]
    return RunStats(
        price_min=min(prices) if prices else 0,
        price_max=max(prices) if prices else 0,
        price_median=statistics.median(prices) if prices else 0,
        by_property_type=dict(
            Counter(
                listing.property_type or "desconocido" for listing in listings
            ).most_common()
        ),
        by_neighborhood=dict(
            Counter(
                listing.neighborhood or "desconocido" for listing in listings
            ).most_common()
        ),
    )


def write_csv(path, listings: list[Listing]) -> None:
    """Flattened convenience export."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for listing in listings:
            row = listing.model_dump(mode="json")
            row["amenities"] = "|".join(
                a["name"] for a in row.get("amenities") or []
            )
            writer.writerow({k: row.get(k) for k in CSV_COLUMNS})


def run(ctx: PipelineContext) -> StageResult:
    """Execute stage 5."""
    src = ctx.path("04_records_clean.jsonl")
    if not src.exists():
        raise RuntimeError(f"{src} missing -- run stage 4 first")

    out_json = ctx.path(artifact_name(ctx, "json"))
    out_csv = ctx.path(artifact_name(ctx, "csv"))
    reject_path = ctx.path(REJECTS)
    reject_path.unlink(missing_ok=True)

    raw_records = list(iter_jsonl(src))
    deduped = deduplicate(raw_records)

    # Count the true parse output, not stage 4's. Stage 4 keys its
    # resume set on listing_id, so it already drops repeats as a side
    # effect -- reading its count here would report "0 duplicates" on
    # every run and hide how much the shards actually overlapped.
    parsed_path = ctx.path("03_records_raw.jsonl")
    parsed_count = (
        sum(1 for _ in iter_jsonl(parsed_path))
        if parsed_path.exists()
        else len(raw_records)
    )
    duplicates = max(0, parsed_count - len(deduped))

    listings: list[Listing] = []
    rejected = 0
    with reject_path.open("w", encoding="utf-8") as rejects:
        for rec in deduped:
            try:
                listings.append(Listing.model_validate(rec))
            except ValidationError as exc:
                rejected += 1
                rejects.write(
                    json.dumps(
                        {
                            "listing_id": rec.get("listing_id"),
                            "errors": exc.errors(include_url=False),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

    listings.sort(key=lambda item: item.listing_id)

    fetch_log = list(iter_jsonl(ctx.path("02_fetch_log.jsonl")))
    requests = list(iter_jsonl(ctx.path("01_requests.jsonl")))
    discovery_path = ctx.path("00_discovery.json")
    strategy = "unknown"
    if discovery_path.exists():
        strategy = json.loads(discovery_path.read_text(encoding="utf-8")).get(
            "extraction_strategy", "unknown"
        )

    artifact = FinalArtifact(
        metadata=RunMetadata(
            extraction_strategy=strategy,
            query=ctx.query(),
            started_at=ctx.started_at,
            finished_at=utc_now(),
            pages_requested=len(requests),
            pages_fetched=sum(
                1 for entry in fetch_log if not entry.get("error")
            ),
            records_parsed=parsed_count,
            records_deduplicated=duplicates,
            records_rejected=rejected,
            scraper_version=__version__,
        ),
        stats=compute_stats(listings),
        listings=listings,
    )

    atomic_write_json(out_json, artifact.model_dump(mode="json"))
    write_csv(out_csv, listings)

    meta = artifact.metadata
    print("\n--- stage 5: consolidate -----------------------------")
    print(f"  parsed          : {meta.records_parsed}")
    print(f"  duplicates      : {meta.records_deduplicated}")
    print(f"  rejected        : {meta.records_rejected}")
    print(f"  final listings  : {len(listings)}")
    print(f"  json            : {out_json.name}")
    print(f"  csv             : {out_csv.name}")
    print("------------------------------------------------------\n")

    return StageResult(
        name="consolidate",
        records=len(listings),
        notes={
            "artifact": str(out_json),
            "duplicates": meta.records_deduplicated,
            "rejected": rejected,
        },
    )


def find_artifact(ctx: PipelineContext) -> Any:
    """Locate this run's final JSON, whatever day it was written."""
    pattern = f"fincaraiz_{ctx.operation}_{ctx.city}_*.json"
    matches = sorted(ctx.out_dir.glob(pattern))
    return matches[-1] if matches else None


def artifact_exists(ctx: PipelineContext) -> bool:
    """Whether stage 5 has a *current* output.

    The filename is date-stamped, so a second run on the same day would
    otherwise find yesterday's-shape artifact, skip, and hand back stale
    listings. If stage 4's output is newer than the artifact, the
    artifact is stale and stage 5 must run again.
    """
    found = find_artifact(ctx)
    if found is None:
        return False
    source = ctx.path("04_records_clean.jsonl")
    if source.exists() and source.stat().st_mtime > found.stat().st_mtime:
        return False
    return True
