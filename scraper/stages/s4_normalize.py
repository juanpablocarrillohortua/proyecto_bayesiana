"""Stage 4 - normalise and clean.

Pure transforms over stage 3 output. No network.
"""

from __future__ import annotations

import html as html_mod
import re
import unicodedata
from datetime import date
from typing import Any

from config import settings
from scraper.checkpoint import Checkpoint, CheckpointedWriter, iter_jsonl
from scraper.context import PipelineContext, StageResult
from scraper.logging_conf import get_logger
from scraper.settings import config_section

log = get_logger(__name__)

ARTIFACT = "04_records_clean.jsonl"

# Match the FIRST number-like token rather than stripping non-digits.
# Stripping would turn "220 m2" into "2202" by absorbing the digit in
# the unit suffix.
_NUMBER = re.compile(r"-?\d[\d.,]*")
_WS = re.compile(r"\s+")
_SMMLV = re.compile(r"\bsmm?lv\b|salarios?\s+m[ií]nimos?", re.I)


def parse_cop(value: Any) -> float | None:
    """Parse a Colombian-formatted amount.

    The period is the thousands separator and the comma is the decimal
    separator, so ``"$ 2.500.000"`` is 2500000 and ``"1.234,56"`` is
    1234.56. Numbers arrive as real ints from the JSON API most of the
    time; this exists for the ``technicalSheet`` string fallbacks.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    match = _NUMBER.search(str(value))
    if not match:
        return None
    text = match.group(0).rstrip(".,")
    if not text or text in {"-", ".", ","}:
        return None

    has_dot, has_comma = "." in text, "," in text
    if has_dot and has_comma:
        # 2.500.000,50 -> dots are grouping, comma is the decimal point.
        text = text.replace(".", "").replace(",", ".")
    elif has_comma:
        text = text.replace(",", ".")
    elif has_dot:
        # A trailing 3-digit group means grouping (2.500.000), anything
        # else is a genuine decimal (220.5).
        tail = text.rsplit(".", 1)[-1]
        if len(tail) == 3 and text.count(".") >= 1:
            text = text.replace(".", "")
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: Any) -> int | None:
    """Coerce to int, preserving None."""
    number = parse_cop(value)
    if number is None:
        return None
    try:
        return int(round(number))
    except (ValueError, OverflowError):
        return None


def clean_text(value: Any) -> str | None:
    """Unescape HTML entities and collapse whitespace."""
    if value is None:
        return None
    text = html_mod.unescape(str(value))
    text = text.replace("\xa0", " ")
    text = _WS.sub(" ", text).strip()
    return text or None


def fold_accents(value: str | None) -> str | None:
    """Accent-free lowercase key, for grouping and joins."""
    if not value:
        return None
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    )
    return _WS.sub(" ", stripped).strip().lower() or None


def title_case(value: str | None) -> str | None:
    """Display casing for place names ('Bogota, d.c.' -> 'Bogota, D.C.')."""
    cleaned = clean_text(value)
    if not cleaned:
        return None
    return " ".join(
        part.upper() if len(part.replace(".", "")) <= 2 else part.capitalize()
        for part in cleaned.split(" ")
    )


def parse_date(value: Any) -> date | None:
    """Parse an ISO ``YYYY-MM-DD`` date, tolerating junk."""
    if not value:
        return None
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn one stage 3 record into a validated-shape listing dict."""
    types = config_section("property_types")
    states = config_section("construction_states")
    brackets = config_section("antiquity_brackets")
    currencies = config_section("currencies")
    base_url = config_section("site")["base_url"]

    type_id = to_int(raw.get("property_type_id"))
    type_meta = types.get(type_id) if type_id is not None else None

    currency_id = to_int(raw.get("price_currency_id"))
    currency = currencies.get(currency_id)
    description = clean_text(raw.get("description"))
    title = clean_text(raw.get("title"))
    foreign = currency != "COP"
    if not foreign:
        blob = f"{title or ''} {description or ''}"
        foreign = bool(_SMMLV.search(blob))

    price = parse_cop(raw.get("price_amount"))
    admin = parse_cop(raw.get("admin_fee"))
    area_built = parse_cop(raw.get("area_built_m2"))

    source_path = raw.get("source_path")
    source_url = (
        base_url + source_path
        if isinstance(source_path, str) and source_path.startswith("/")
        else source_path
    )

    operation = (raw.get("operation_slug") or "").lower() or None
    if operation is None:
        name = fold_accents(raw.get("operation_name")) or ""
        operation = "venta" if name.startswith("venta") else "arriendo"

    state_id = to_int(raw.get("construction_state_id"))
    bracket_code = to_int(raw.get("antiquity_code"))

    price_per_m2 = None
    if price is not None and area_built:
        price_per_m2 = round(price / area_built, 2)

    total_monthly = None
    if operation == "arriendo" and price is not None:
        total_monthly = price + (admin or 0.0)

    return {
        "listing_id": str(raw.get("listing_id") or ""),
        "source_url": source_url,
        "operation": operation,
        "scraped_at": raw.get("scraped_at"),
        "title": title,
        "description": description,
        "property_type": (
            type_meta["canonical"]
            if type_meta
            else fold_accents(raw.get("property_type_name"))
        ),
        "property_type_raw": clean_text(raw.get("property_type_name")),
        "price_amount": price,
        "price_currency": currency
        or clean_text(raw.get("price_currency_symbol")),
        "price_currency_is_foreign": foreign,
        "admin_fee": admin,
        "admin_included_in_price": raw.get("admin_included"),
        "area_built_m2": area_built,
        "area_private_m2": parse_cop(raw.get("area_private_m2")),
        "area_terrain_m2": parse_cop(raw.get("area_terrain_m2")),
        "bedrooms": to_int(raw.get("bedrooms")),
        "bathrooms": to_int(raw.get("bathrooms")),
        "parking_spaces": to_int(raw.get("parking_spaces")),
        "stratum": to_int(raw.get("stratum")),
        "floor": to_int(raw.get("floor")),
        "floors_total": to_int(raw.get("floors_total")),
        "age_bracket_code": bracket_code,
        "age_bracket_label": (
            clean_text(raw.get("age_bracket_label_raw"))
            or brackets.get(bracket_code)
        ),
        # Never derived: the site publishes a bracket, not a year.
        "age_years": None,
        "construction_year": to_int(raw.get("construction_year")),
        "construction_state": (
            clean_text(raw.get("construction_state_raw"))
            or states.get(state_id)
        ),
        "city": title_case(raw.get("city")),
        "locality": title_case(raw.get("locality")),
        "neighborhood": title_case(raw.get("neighborhood")),
        "zone": title_case(raw.get("zone")),
        "address_text": clean_text(raw.get("address_text")),
        "latitude": parse_cop(raw.get("latitude")),
        "longitude": parse_cop(raw.get("longitude")),
        "amenities": [
            {
                "id": a.get("id"),
                "name": clean_text(a.get("name")),
                "group": clean_text(a.get("group")),
            }
            for a in (raw.get("amenities") or [])
            if clean_text(a.get("name"))
        ],
        "image_urls": list(raw.get("image_urls") or []),
        "agency_name": clean_text(raw.get("agency_name")),
        "agency_id": (str(raw["agency_id"]) if raw.get("agency_id") else None),
        "publisher_type": clean_text(raw.get("publisher_type")),
        "published_at": parse_date(raw.get("published_at")),
        "updated_at": parse_date(raw.get("updated_at")),
        "price_per_m2": price_per_m2,
        "total_monthly_cost": total_monthly,
    }


def run(ctx: PipelineContext) -> StageResult:
    """Execute stage 4."""
    src = ctx.path("03_records_raw.jsonl")
    if not src.exists():
        raise RuntimeError(f"{src} missing -- run stage 3 first")

    checkpoint = Checkpoint(ctx.out_dir, "s4_normalize")
    out_path = ctx.path(ARTIFACT)
    if ctx.force:
        out_path.unlink(missing_ok=True)
        checkpoint.clear()

    foreign = 0
    with CheckpointedWriter(
        out_path, checkpoint, "listing_id", settings.checkpoint_every
    ) as writer:
        for raw in iter_jsonl(src):
            listing_id = str(raw.get("listing_id") or "")
            if not listing_id or writer.is_done(listing_id):
                continue
            record = normalize_record(raw)
            if record["price_currency_is_foreign"]:
                foreign += 1
            writer.write(record)
        written = writer.written

    print("\n--- stage 4: normalize -------------------------------")
    print(f"  records cleaned : {written}")
    print(f"  non-COP flagged : {foreign}")
    print("------------------------------------------------------\n")

    return StageResult(
        name="normalize", records=written, notes={"foreign": foreign}
    )
