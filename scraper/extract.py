"""Pure extraction helpers. No network, no side effects.

The site is a Next.js application: every search page carries the complete
result set in a single ``<script id="__NEXT_DATA__">`` JSON blob, so no
browser automation is needed.
"""

from __future__ import annotations

import json
from typing import Any

from selectolax.parser import HTMLParser


class NextDataMissingError(RuntimeError):
    """Raised when the embedded JSON blob is absent or unparseable.

    This means the site changed shape. Callers should fail loudly rather
    than fall back to guessing, so the operator finds out immediately.
    """


def dig(obj: Any, path: str, default: Any = None) -> Any:
    """Follow a dotted path through nested dicts and lists.

    Numeric segments index into lists: ``"locations.city.0.name"``.
    Returns ``default`` if any segment is missing.
    """
    current = obj
    for segment in path.split("."):
        if current is None:
            return default
        if segment.isdigit():
            if not isinstance(current, (list, tuple)):
                return default
            index = int(segment)
            if index >= len(current):
                return default
            current = current[index]
        else:
            if not isinstance(current, dict) or segment not in current:
                return default
            current = current[segment]
    return default if current is None else current


def extract_next_data(html: str, selector: str) -> dict[str, Any]:
    """Parse ``__NEXT_DATA__`` out of a page.

    Raises:
        NextDataMissingError: if the script tag is absent or not valid JSON.
    """
    node = HTMLParser(html).css_first(selector)
    if node is None:
        raise NextDataMissingError(
            f"no node matched {selector!r}; the page may be a captcha "
            "or block page, or the site was redesigned"
        )
    text = node.text(strip=False)
    if not text:
        raise NextDataMissingError(f"{selector!r} matched but was empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NextDataMissingError(f"{selector!r} was not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise NextDataMissingError("__NEXT_DATA__ did not parse to an object")
    return data


def parse_wkt_point(wkt: str | None) -> tuple[float, float] | None:
    """Parse ``POINT (lon lat)`` into ``(lat, lon)``.

    Note the axis order swap: WKT is longitude-first.
    """
    if not wkt or not isinstance(wkt, str):
        return None
    inner = wkt.strip().removeprefix("POINT").strip()
    inner = inner.strip("()").strip()
    parts = inner.split()
    if len(parts) != 2:
        return None
    try:
        lon, lat = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return lat, lon


def technical_sheet_index(listing: dict[str, Any]) -> dict[str, str]:
    """Map ``technicalSheet`` rows to ``{field: value}``.

    Only rows with a non-empty value are included. This is what lets
    stage 3 tell "zero" apart from "not stated": the site renders a blank
    sheet row for an unknown value while the raw field still reads 0.
    """
    out: dict[str, str] = {}
    for row in listing.get("technicalSheet") or []:
        if not isinstance(row, dict):
            continue
        field = row.get("field")
        value = row.get("value")
        if field and value not in (None, "", " "):
            out[str(field)] = str(value).strip()
    return out
