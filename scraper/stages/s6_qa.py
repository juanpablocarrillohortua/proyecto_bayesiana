"""Stage 6 - quality assurance and coverage report.

This stage exists because of a specific measured risk: the site's
relevance ordering is unstable at depth, so a run can quietly collect far
fewer distinct listings than the result counter promised. Stage 1 shards
to avoid that, but the only honest way to know it worked is to compare
what we actually collected against what each shard claimed to hold.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from scraper.checkpoint import atomic_write_json, iter_jsonl
from scraper.context import PipelineContext, StageResult
from scraper.logging_conf import get_logger
from scraper.settings import config_section
from scraper.stages.s5_consolidate import find_artifact

log = get_logger(__name__)

ARTIFACT = "06_qa_report.json"

#: Fields whose null rate is worth reporting to the operator.
WATCHED_FIELDS = [
    "price_amount",
    "area_built_m2",
    "area_private_m2",
    "bedrooms",
    "bathrooms",
    "parking_spaces",
    "stratum",
    "floor",
    "admin_fee",
    "age_bracket_label",
    "construction_state",
    "latitude",
    "longitude",
    "locality",
    "neighborhood",
    "address_text",
]


def null_rates(listings: list[dict[str, Any]]) -> dict[str, float]:
    """Fraction of records where each watched field is null."""
    total = len(listings) or 1
    return {
        field: round(
            sum(1 for item in listings if item.get(field) in (None, ""))
            / total,
            4,
        )
        for field in WATCHED_FIELDS
    }


def coverage(
    listings: list[dict[str, Any]],
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    """Distinct listings collected versus what the shards advertised.

    Shard totals overlap between axes, so the expected figure is taken
    from the *leaf* shards actually enumerated.
    """
    per_shard: dict[str, int] = {}
    at_risk: list[str] = []
    backstops = 0
    for req in requests:
        shard = req.get("shard_id")
        if not shard or shard in per_shard:
            continue
        # Backstop shards re-cover ground their siblings already claim,
        # so counting their totals would inflate the denominator.
        if req.get("shard_role") == "backstop":
            per_shard[shard] = 0
            backstops += 1
            continue
        per_shard[shard] = req.get("shard_total") or 0
        if req.get("coverage_at_risk"):
            at_risk.append(shard)

    expected = sum(per_shard.values())
    distinct = len({item.get("listing_id") for item in listings})
    ratio = round(distinct / expected, 4) if expected else None
    return {
        "distinct_collected": distinct,
        "expected_from_shards": expected,
        "coverage_ratio": ratio,
        "shards": len(per_shard),
        "backstop_shards": backstops,
        "shards_at_risk": at_risk,
    }


def geo_sanity(listings: list[dict[str, Any]], city: str) -> dict[str, Any]:
    """Count coordinates outside the city bounding box."""
    boxes = config_section("qa")["geo_bbox"]
    box = boxes.get(city)
    if not box:
        return {"checked": False, "reason": f"no bbox for {city!r}"}
    outside = 0
    missing = 0
    for item in listings:
        lat, lon = item.get("latitude"), item.get("longitude")
        if lat is None or lon is None:
            missing += 1
            continue
        if not (box["lat_min"] <= lat <= box["lat_max"]):
            outside += 1
        elif not (box["lon_min"] <= lon <= box["lon_max"]):
            outside += 1
    return {
        "checked": True,
        "bbox": box,
        "missing_coords": missing,
        "outside_bbox": outside,
    }


def price_outliers(
    listings: list[dict[str, Any]], operation: str
) -> dict[str, Any]:
    """Flag prices outside a plausible band for the operation."""
    bands = config_section("qa")["price_sanity"]
    band = bands.get(operation)
    if not band:
        return {"checked": False}
    low, high, ids = 0, 0, []
    for item in listings:
        price = item.get("price_amount")
        if price is None:
            continue
        if price < band["min"]:
            low += 1
            ids.append(item.get("listing_id"))
        elif price > band["max"]:
            high += 1
            ids.append(item.get("listing_id"))
    return {
        "checked": True,
        "band": band,
        "below_min": low,
        "above_max": high,
        "sample_ids": ids[:10],
    }


def run(ctx: PipelineContext) -> StageResult:
    """Execute stage 6."""
    final_path = find_artifact(ctx)
    if final_path is None:
        raise RuntimeError(
            "no consolidated artifact found -- run stage 5 first"
        )

    payload = json.loads(final_path.read_text(encoding="utf-8"))
    listings = payload.get("listings", [])
    requests = list(iter_jsonl(ctx.path("01_requests.jsonl")))
    fetch_log = list(iter_jsonl(ctx.path("02_fetch_log.jsonl")))
    parse_errors = list(iter_jsonl(ctx.path("03_parse_errors.jsonl")))

    cov = coverage(listings, requests)
    warn_below = config_section("qa")["coverage_warn_below"]

    by_shard: dict[str, int] = defaultdict(int)
    for req in requests:
        by_shard[req.get("shard_id") or "?"] += 1

    report = {
        "artifact": final_path.name,
        "operation": ctx.operation,
        "city": ctx.city,
        "coverage": cov,
        "coverage_ok": (
            cov["coverage_ratio"] is None
            or cov["coverage_ratio"] >= warn_below
        ),
        "records": {
            "final": len(listings),
            "parsed": payload["metadata"]["records_parsed"],
            "duplicates": payload["metadata"]["records_deduplicated"],
            "rejected": payload["metadata"]["records_rejected"],
        },
        "fetch": {
            "requests": len(requests),
            "logged": len(fetch_log),
            "errors": sum(1 for e in fetch_log if e.get("error")),
            "cached": sum(1 for e in fetch_log if e.get("cached")),
        },
        "parse_errors": len(parse_errors),
        "null_rates": null_rates(listings),
        "geo": geo_sanity(listings, ctx.city),
        "price_outliers": price_outliers(listings, ctx.operation),
        "by_property_type": dict(
            Counter(
                item.get("property_type") or "desconocido" for item in listings
            ).most_common(15)
        ),
        "pages_per_shard_max": max(by_shard.values()) if by_shard else 0,
    }
    atomic_write_json(ctx.path(ARTIFACT), report)

    ratio = cov["coverage_ratio"]
    ratio_text = f"{ratio:.1%}" if ratio is not None else "n/a"
    flag = "" if report["coverage_ok"] else "   <-- BELOW THRESHOLD"
    print("\n--- stage 6: QA report -------------------------------")
    print(
        f"  coverage        : {cov['distinct_collected']} / "
        f"{cov['expected_from_shards']} = {ratio_text}{flag}"
    )
    print(
        f"  shards          : {cov['shards']} "
        f"({len(cov['shards_at_risk'])} at risk)"
    )
    print(f"  duplicates      : {report['records']['duplicates']}")
    print(f"  rejected        : {report['records']['rejected']}")
    print(f"  parse errors    : {report['parse_errors']}")
    print(f"  fetch errors    : {report['fetch']['errors']}")
    geo = report["geo"]
    if geo.get("checked"):
        print(
            f"  geo outside bbox: {geo['outside_bbox']} "
            f"(missing {geo['missing_coords']})"
        )
    print("  null rates      :")
    for field, rate in sorted(
        report["null_rates"].items(), key=lambda kv: -kv[1]
    )[:8]:
        print(f"      {field:<20} {rate:.1%}")
    print("------------------------------------------------------\n")

    if not report["coverage_ok"]:
        log.warning(
            "s6.low_coverage",
            ratio=ratio,
            threshold=warn_below,
            hint="lower FR_MAX_SAFE_PAGE to shard more aggressively",
        )

    return StageResult(name="qa", records=1, notes={"coverage_ratio": ratio})
