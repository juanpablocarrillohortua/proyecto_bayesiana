"""Stage 3 - raw HTML to one record per listing.

Pure: this module never touches the network. Every field path comes from
``config.yaml``, so adapting to a site change means editing YAML, not
Python.
"""

from __future__ import annotations

from typing import Any

from config import settings
from scraper.checkpoint import Checkpoint, CheckpointedWriter, utc_now
from scraper.context import PipelineContext, StageResult
from scraper.extract import (
    NextDataMissingError,
    dig,
    extract_next_data,
    parse_wkt_point,
    technical_sheet_index,
)
from scraper.logging_conf import get_logger
from scraper.settings import config_section

log = get_logger(__name__)

ARTIFACT = "03_records_raw.jsonl"
ERRORS = "03_parse_errors.jsonl"


def _sheet_lookup() -> dict[str, str]:
    """Output field -> technicalSheet field name."""
    return {
        out: sheet
        for sheet, out in config_section("technical_sheet_map").items()
    }


def parse_listing(
    listing: dict[str, Any],
    request_id: str,
    descriptor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map one raw listing object to a flat record.

    Two site quirks are handled here:

    * ``0`` normally means "not stated", not zero. The rendered
      ``technicalSheet`` omits unknown values, so it is used as the
      authority: a 0 with no sheet row becomes ``None``.
    * ``rooms`` is not the bedroom count -- it was 0 on most sampled
      listings while ``bedrooms`` held the real value. The mapping in
      ``config.yaml`` points at ``bedrooms`` deliberately.
    """
    field_map = config_section("field_map")
    zero_null = set(config_section("zero_means_null"))
    sheet_for = _sheet_lookup()
    sheet = technical_sheet_index(listing)

    record: dict[str, Any] = {}
    for out_field, path in field_map.items():
        record[out_field] = dig(listing, path)

    for out_field in zero_null:
        if record.get(out_field) not in (0, "0"):
            continue
        sheet_field = sheet_for.get(out_field)
        if sheet_field is None or sheet_field not in sheet:
            record[out_field] = None

    # Privacy: never persist contact details. Agency identity is kept
    # only for professional publishers.
    if listing.get("owner", {}).get("particular"):
        record["agency_name"] = None
        record["agency_id"] = None

    # The site hides the street address on some listings; honour that.
    if not record.get("show_address"):
        record["address_text"] = None

    if record.get("latitude") is None:
        point = parse_wkt_point(record.get("location_point_wkt"))
        if point:
            record["latitude"], record["longitude"] = point

    record["amenities"] = [
        {
            "id": f.get("id"),
            "name": f.get("name"),
            "group": f.get("group"),
        }
        for f in (listing.get("facilities") or [])
        if isinstance(f, dict) and f.get("name")
    ]
    record["image_urls"] = [
        img.get("image")
        for img in (listing.get("images") or [])
        if isinstance(img, dict) and img.get("image")
    ]
    record["age_bracket_label_raw"] = sheet.get("constructionYear")
    record["construction_state_raw"] = sheet.get("construction_state_name")

    record["listing_id"] = str(record.get("listing_id") or "")
    record["request_id"] = request_id
    record["scraped_at"] = utc_now().isoformat()
    if descriptor:
        record["shard_id"] = descriptor.get("shard_id")
        record["operation_slug"] = descriptor.get("operation")
    record.pop("location_point_wkt", None)
    record.pop("show_address", None)
    return record


def parse_page(
    html: str,
    request_id: str,
    selector: str,
    listings_pointer: str,
    descriptor: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract every listing on one cached page."""
    data = extract_next_data(html, selector)
    listings = dig(data, listings_pointer)
    if not isinstance(listings, list):
        raise NextDataMissingError(
            f"pointer {listings_pointer!r} did not resolve to a list"
        )
    return [
        parse_listing(item, request_id, descriptor)
        for item in listings
        if isinstance(item, dict)
    ]


def run(ctx: PipelineContext) -> StageResult:
    """Execute stage 3."""
    import json

    from scraper.checkpoint import iter_jsonl

    discovery = json.loads(
        ctx.path("00_discovery.json").read_text(encoding="utf-8")
    )
    selector = discovery["script_selector"]
    pointer = discovery["pointers"]["listings"]

    descriptors = {
        d["request_id"]: d for d in iter_jsonl(ctx.path("01_requests.jsonl"))
    }

    checkpoint = Checkpoint(ctx.out_dir, "s3_parse")
    out_path = ctx.path(ARTIFACT)
    err_path = ctx.path(ERRORS)
    if ctx.force:
        out_path.unlink(missing_ok=True)
        err_path.unlink(missing_ok=True)
        checkpoint.clear()

    raw_files = sorted(ctx.raw_dir.glob("*.html"))
    if not raw_files:
        raise RuntimeError(
            f"no cached pages in {ctx.raw_dir} -- run stage 2 first"
        )

    errors = 0
    pages_done: set[str] = set()
    if out_path.exists():
        pages_done = {
            r["request_id"] for r in iter_jsonl(out_path) if "request_id" in r
        }

    with (
        CheckpointedWriter(
            out_path, checkpoint, "listing_id", settings.checkpoint_every
        ) as writer,
        err_path.open("a", encoding="utf-8") as err_handle,
    ):
        for raw_file in raw_files:
            request_id = raw_file.stem
            if request_id in pages_done and not ctx.force:
                continue
            try:
                html = raw_file.read_text(encoding="utf-8", errors="replace")
                records = parse_page(
                    html,
                    request_id,
                    selector,
                    pointer,
                    descriptors.get(request_id),
                )
            except Exception as exc:  # noqa: BLE001 - never kill the batch
                errors += 1
                err_handle.write(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "file": str(raw_file),
                            "error": f"{type(exc).__name__}: {exc}",
                            "at": utc_now().isoformat(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log.warning(
                    "s3.parse_error",
                    request_id=request_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            for record in records:
                writer.write(record)
        parsed = writer.written

    print("\n--- stage 3: parse -----------------------------------")
    print(f"  pages read      : {len(raw_files)}")
    print(f"  records parsed  : {parsed}")
    print(f"  parse errors    : {errors}")
    print("------------------------------------------------------\n")

    return StageResult(name="parse", records=parsed, notes={"errors": errors})
