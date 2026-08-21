"""Pydantic v2 schemas for pipeline artifacts.

Field naming note: the brief asked for "rooms", "currencie", "# parkings"
and "amennities". Those map to ``bedrooms``, ``price_currency``,
``parking_spaces`` and ``amenities`` here -- see the mapping table in
scraper/README.md.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Operation = Literal["arriendo", "venta"]


class Amenity(BaseModel):
    """One facility offered by a property."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    name: str
    group: str | None = None


class RequestDescriptor(BaseModel):
    """A single page to fetch. One line of ``01_requests.jsonl``."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    url: str
    page: int
    shard_id: str
    shard_kind: Literal["root", "type", "type_neighbourhood"]
    #: "backstop" shards re-cover a lossy split; excluded from coverage
    #: denominators so they cannot inflate the ratio.
    shard_role: Literal["primary", "backstop"] = "primary"
    operation: Operation
    city: str
    property_type_slug: str | None = None
    neighbourhood_slug: str | None = None
    #: Result count the shard reported, for stage 6 coverage maths.
    shard_total: int | None = None
    shard_last_page: int | None = None
    #: True when the shard could not be split below max_safe_page.
    coverage_at_risk: bool = False


class FetchLogEntry(BaseModel):
    """One line of ``02_fetch_log.jsonl``."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    url: str
    effective_url: str | None = None
    status: int | None = None
    elapsed_ms: int | None = None
    bytes: int | None = None
    sha256: str | None = None
    cached: bool = False
    error: str | None = None
    fetched_at: datetime


class RawRecord(BaseModel):
    """A listing as extracted, before normalisation. Stage 3 output."""

    model_config = ConfigDict(extra="allow")

    listing_id: str
    request_id: str
    scraped_at: datetime


class Listing(BaseModel):
    """A validated, normalised listing. Stage 4/5 output."""

    model_config = ConfigDict(extra="forbid")

    # --- identity -----------------------------------------------------
    listing_id: str
    source_url: str | None = None
    operation: Operation
    scraped_at: datetime

    # --- descriptive --------------------------------------------------
    title: str | None = None
    description: str | None = None
    property_type: str | None = None
    property_type_raw: str | None = None

    # --- money --------------------------------------------------------
    price_amount: float | None = None
    price_currency: str | None = None
    #: True when the listing is not quoted in COP (e.g. USD, SMMLV).
    price_currency_is_foreign: bool = False
    admin_fee: float | None = None
    admin_included_in_price: bool | None = None

    # --- geometry -----------------------------------------------------
    area_built_m2: float | None = None
    area_private_m2: float | None = None
    area_terrain_m2: float | None = None

    # --- counts -------------------------------------------------------
    bedrooms: int | None = None
    bathrooms: int | None = None
    parking_spaces: int | None = None
    stratum: int | None = None
    floor: int | None = None
    floors_total: int | None = None

    # --- age ----------------------------------------------------------
    # The site publishes a bracket, never an exact year, so age_years
    # stays null rather than inventing precision a model would trust.
    age_bracket_code: int | None = None
    age_bracket_label: str | None = None
    age_years: float | None = None
    construction_year: int | None = None
    construction_state: str | None = None

    # --- location -----------------------------------------------------
    city: str | None = None
    locality: str | None = None
    neighborhood: str | None = None
    zone: str | None = None
    address_text: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    # --- collections --------------------------------------------------
    amenities: list[Amenity] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)

    # --- provenance ---------------------------------------------------
    # Agency identity only, and only for professional publishers.
    # Contact details are never collected.
    agency_name: str | None = None
    agency_id: str | None = None
    publisher_type: str | None = None
    published_at: date | None = None
    updated_at: date | None = None

    # --- derived ------------------------------------------------------
    price_per_m2: float | None = None
    total_monthly_cost: float | None = None


class RunMetadata(BaseModel):
    """The ``metadata`` block of the final artifact."""

    model_config = ConfigDict(extra="forbid")

    source: str = "fincaraiz.com.co"
    extraction_strategy: str
    query: dict[str, Any]
    started_at: datetime
    finished_at: datetime
    pages_requested: int = 0
    pages_fetched: int = 0
    records_parsed: int = 0
    records_deduplicated: int = 0
    records_rejected: int = 0
    scraper_version: str = "1.0.0"


class RunStats(BaseModel):
    """The ``stats`` block of the final artifact."""

    model_config = ConfigDict(extra="forbid")

    price_min: float = 0
    price_max: float = 0
    price_median: float = 0
    by_property_type: dict[str, int] = Field(default_factory=dict)
    by_neighborhood: dict[str, int] = Field(default_factory=dict)


class FinalArtifact(BaseModel):
    """The single consolidated ``.json`` deliverable."""

    model_config = ConfigDict(extra="forbid")

    metadata: RunMetadata
    stats: RunStats
    listings: list[Listing]
