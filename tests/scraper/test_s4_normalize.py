"""Stage 4 normalisation tests. No network."""

from __future__ import annotations

import pytest

from scraper.models import Listing
from scraper.stages.s3_parse import parse_listing
from scraper.stages.s4_normalize import (
    clean_text,
    fold_accents,
    normalize_record,
    parse_cop,
    parse_date,
    to_int,
)


class TestParseCop:
    """Colombian numerals: '.' groups thousands, ',' is the decimal."""

    @pytest.mark.parametrize("raw,expected", [
        ("$ 2.500.000", 2500000.0),
        ("2.500.000", 2500000.0),
        ("$1.200.000", 1200000.0),
        ("850.000", 850000.0),
        ("1.234,56", 1234.56),
        ("$ 3.600.000,00", 3600000.0),
        ("220 m2", 220.0),
        ("1,5", 1.5),
        ("220.5", 220.5),
        (2500000, 2500000.0),
        (1234.5, 1234.5),
    ])
    def test_parses(self, raw, expected):
        assert parse_cop(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [None, "", "   ", "-", "sin dato", "$"])
    def test_unparseable_is_none(self, raw):
        assert parse_cop(raw) is None

    def test_booleans_are_not_numbers(self):
        # bool is an int subclass; True must not become 1.0.
        assert parse_cop(True) is None

    def test_thousands_vs_decimal_disambiguation(self):
        # A trailing 3-digit group is grouping...
        assert parse_cop("2.500") == 2500.0
        # ...anything else is a real decimal.
        assert parse_cop("2.5") == 2.5


class TestToInt:
    def test_rounds(self):
        assert to_int("3,7") == 4

    def test_none_passthrough(self):
        assert to_int(None) is None


class TestText:
    def test_unescapes_entities(self):
        assert clean_text("Caf&eacute;  &amp;  m&aacute;s") == "Café & más"

    def test_collapses_whitespace(self):
        assert clean_text("  a\n\n b\t c ") == "a b c"

    def test_empty_becomes_none(self):
        assert clean_text("   ") is None

    def test_fold_accents(self):
        assert fold_accents("Bogotá, D.C.") == "bogota, d.c."
        assert fold_accents("Engativá") == "engativa"

    def test_fold_accents_none(self):
        assert fold_accents(None) is None


class TestDates:
    def test_iso(self):
        assert str(parse_date("2026-08-02")) == "2026-08-02"

    def test_datetime_string_truncated(self):
        assert str(parse_date("2026-08-02T14:36:53+00:00")) == "2026-08-02"

    def test_junk(self):
        assert parse_date("no") is None
        assert parse_date(None) is None


class TestNormalizeRecord:
    """End-to-end stage 3 -> stage 4 over the real fixture."""

    @pytest.fixture
    def records(self, raw_listings):
        return [
            normalize_record(parse_listing(r, "req"))
            for r in raw_listings
        ]

    def test_all_validate_against_the_schema(self, records):
        for record in records:
            Listing.model_validate(record)

    def test_currency_normalised_to_cop(self, records):
        assert records[0]["price_currency"] == "COP"
        assert records[0]["price_currency_is_foreign"] is False

    def test_property_type_controlled_vocabulary(self, records):
        allowed = {
            "casa", "apartamento", "apartaestudio", "oficina", "local",
            "bodega", "lote", "finca", "parqueadero", "consultorio",
            "edificio", "cabana", "casa_campestre", "casa_lote",
            "habitacion",
        }
        for record in records:
            assert record["property_type"] in allowed

    def test_price_per_m2_derived(self, records):
        record = records[0]
        if record["price_amount"] and record["area_built_m2"]:
            expected = record["price_amount"] / record["area_built_m2"]
            assert record["price_per_m2"] == pytest.approx(expected, rel=1e-6)

    def test_price_per_m2_null_when_area_missing(self, raw_listings):
        raw = parse_listing(raw_listings[0], "req")
        raw["area_built_m2"] = None
        assert normalize_record(raw)["price_per_m2"] is None

    def test_total_monthly_cost_for_arriendo(self, records):
        record = records[0]
        assert record["operation"] == "arriendo"
        expected = record["price_amount"] + (record["admin_fee"] or 0)
        assert record["total_monthly_cost"] == pytest.approx(expected)

    def test_total_monthly_cost_absent_for_venta(self, raw_listings):
        raw = parse_listing(raw_listings[0], "req")
        raw["operation_slug"] = "venta"
        raw["operation_name"] = "Venta"
        assert normalize_record(raw)["total_monthly_cost"] is None

    def test_age_years_is_never_invented(self, records):
        # The site publishes a bracket, never an exact year. Emitting a
        # midpoint here would feed fake precision into the model.
        for record in records:
            assert record["age_years"] is None
            assert record["age_bracket_code"] is not None

    def test_age_bracket_label_present(self, records):
        assert records[0]["age_bracket_label"]

    def test_source_url_absolute(self, records):
        for record in records:
            if record["source_url"]:
                assert record["source_url"].startswith("https://")

    def test_place_names_title_cased(self, records):
        for record in records:
            if record["city"]:
                assert not record["city"].islower()

    def test_amenities_shape(self, records):
        for record in records:
            for amenity in record["amenities"]:
                assert set(amenity) == {"id", "name", "group"}
                assert amenity["name"]


class TestForeignCurrencyFlag:
    def test_smmlv_is_flagged(self, raw_listings):
        raw = parse_listing(raw_listings[0], "req")
        raw["description"] = "Canon equivalente a 2 SMMLV mensuales"
        assert normalize_record(raw)["price_currency_is_foreign"] is True

    def test_unknown_currency_id_is_flagged(self, raw_listings):
        raw = parse_listing(raw_listings[0], "req")
        raw["price_currency_id"] = 1
        assert normalize_record(raw)["price_currency_is_foreign"] is True
