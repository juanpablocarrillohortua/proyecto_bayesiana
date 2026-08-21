"""Shared test fixtures.

Stages 3 and 4 are specified as pure, so the whole test package is
blocked from making network calls. Any accidental request raises rather
than silently reaching the live site.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class NetworkUsedInTests(RuntimeError):
    """Raised if a test tries to open a socket."""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly on any HTTP attempt from inside a test."""
    import httpx

    def blocked(*args, **kwargs):
        raise NetworkUsedInTests(
            "tests must not touch the network; use a fixture instead"
        )

    monkeypatch.setattr(httpx.Client, "request", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "request", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)


@pytest.fixture(scope="session")
def listing_html() -> str:
    """A trimmed real search page with three curated listings."""
    return (FIXTURES / "listing_page.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def blocked_html() -> str:
    """A page with no embedded JSON, as an anti-bot interstitial."""
    return (FIXTURES / "blocked_page.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def raw_listings(listing_html: str) -> list[dict]:
    """The three listing objects straight out of the fixture."""
    blob = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        listing_html, re.S,
    ).group(1)
    data = json.loads(blob)
    return data["props"]["pageProps"]["fetchResult"]["searchFast"]["data"]
