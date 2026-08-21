"""Stage 3 parsing tests. No network."""

from __future__ import annotations

import pytest

from scraper.extract import (
    NextDataMissingError,
    dig,
    extract_next_data,
    parse_wkt_point,
    technical_sheet_index,
)
from scraper.stages.s3_parse import parse_listing, parse_page

SELECTOR = "script#__NEXT_DATA__"
POINTER = "props.pageProps.fetchResult.searchFast.data"


def test_extract_next_data_finds_the_blob(listing_html):
    data = extract_next_data(listing_html, SELECTOR)
    assert data["buildId"] == "test-build-id"


def test_extract_next_data_raises_on_block_page(blocked_html):
    # A block page must fail loudly, never parse as "zero listings".
    with pytest.raises(NextDataMissingError):
        extract_next_data(blocked_html, SELECTOR)


def test_parse_page_yields_one_record_per_listing(listing_html):
    records = parse_page(listing_html, "req1", SELECTOR, POINTER)
    assert len(records) == 3
    assert len({r["listing_id"] for r in records}) == 3
    assert all(r["request_id"] == "req1" for r in records)


class TestDig:
    """Dotted-path lookups, including list indices."""

    def test_nested_dict(self):
        assert dig({"a": {"b": {"c": 7}}}, "a.b.c") == 7

    def test_list_index(self):
        obj = {"locations": {"city": [{"name": "Bogota"}]}}
        assert dig(obj, "locations.city.0.name") == "Bogota"

    def test_missing_returns_default(self):
        assert dig({"a": 1}, "a.b.c", "fallback") == "fallback"

    def test_index_out_of_range(self):
        assert dig({"xs": []}, "xs.3.name") is None

    def test_null_value_yields_default(self):
        assert dig({"a": None}, "a", "dflt") == "dflt"


class TestZeroMeansNull:
    """The site writes 0 for "not stated".

    ``technicalSheet`` is the disambiguator: it omits unknown values, so
    a 0 with no sheet row must become None while a genuine 0 backed by a
    sheet row is preserved.
    """

    def test_zero_with_blank_sheet_becomes_null(self, raw_listings):
        # Fixture listing 2 has garage=0 and an empty sheet row.
        record = parse_listing(raw_listings[1], "req")
        assert raw_listings[1]["garage"] == 0
        assert record["parking_spaces"] is None

    def test_real_value_survives(self, raw_listings):
        # Fixture listing 1 has garage=1 and a populated sheet row.
        record = parse_listing(raw_listings[0], "req")
        assert record["parking_spaces"] == 1

    def test_admin_fee_zero_is_undisclosed(self, raw_listings):
        # commonExpenses.amount == 0 means "not published", not free.
        for raw, rec in zip(
            raw_listings,
            [parse_listing(r, "req") for r in raw_listings],
        ):
            if raw["commonExpenses"]["amount"] == 0:
                assert rec["admin_fee"] is None


class TestBedroomsNotRooms:
    """``rooms`` is not the bedroom count on this site."""

    def test_maps_to_bedrooms_field(self, raw_listings):
        raw = dict(raw_listings[0])
        raw["rooms"] = 0
        raw["bedrooms"] = 4
        assert parse_listing(raw, "req")["bedrooms"] == 4


class TestPrivacy:
    """Personal contact data must never reach disk."""

    def test_private_seller_loses_agency_identity(self, raw_listings):
        record = parse_listing(raw_listings[1], "req")
        assert raw_listings[1]["owner"]["particular"] is True
        assert record["agency_name"] is None
        assert record["agency_id"] is None

    def test_agency_kept_for_professional_publisher(self, raw_listings):
        record = parse_listing(raw_listings[0], "req")
        assert record["agency_name"]

    def test_no_contact_fields_anywhere(self, raw_listings):
        blob = str([parse_listing(r, "req") for r in raw_listings])
        for banned in ("masked_phone", "whatsapp", "subsidiaries"):
            assert banned not in blob

    def test_hidden_address_is_dropped(self, raw_listings):
        record = parse_listing(raw_listings[1], "req")
        assert raw_listings[1]["showAddress"] is False
        assert record["address_text"] is None

    def test_visible_address_is_kept(self, raw_listings):
        assert parse_listing(raw_listings[0], "req")["address_text"]


class TestCoordinates:
    """Latitude/longitude, including the WKT fallback."""

    def test_direct_fields_used_when_present(self, raw_listings):
        record = parse_listing(raw_listings[0], "req")
        assert record["latitude"] == raw_listings[0]["latitude"]

    def test_wkt_fallback_when_fields_are_null(self, raw_listings):
        # Fixture listing 3 has null lat/lon but a POINT string.
        record = parse_listing(raw_listings[2], "req")
        assert record["latitude"] == pytest.approx(4.65)
        assert record["longitude"] == pytest.approx(-74.05)

    def test_wkt_axis_order_is_lon_lat(self):
        # WKT is longitude-first; the parser must swap.
        assert parse_wkt_point("POINT (-74.05 4.65)") == (4.65, -74.05)

    def test_wkt_garbage_returns_none(self):
        assert parse_wkt_point("not a point") is None
        assert parse_wkt_point(None) is None


class TestAmenities:
    """Amenities come from the per-listing facilities array."""

    def test_amenities_extracted(self, raw_listings):
        record = parse_listing(raw_listings[0], "req")
        assert isinstance(record["amenities"], list)
        if raw_listings[0].get("facilities"):
            assert record["amenities"][0]["name"]
            assert "group" in record["amenities"][0]


def test_technical_sheet_index_skips_blank_rows():
    listing = {"technicalSheet": [
        {"field": "garage", "value": "", "text": "Parqueaderos"},
        {"field": "stratum", "value": "3", "text": "Estrato"},
    ]}
    sheet = technical_sheet_index(listing)
    assert "garage" not in sheet
    assert sheet["stratum"] == "3"


def test_age_bracket_label_captured(raw_listings):
    # antiquity is a bracket code; the sheet carries the human label.
    record = parse_listing(raw_listings[0], "req")
    assert record["antiquity_code"] is not None
    assert record["age_bracket_label_raw"]
