"""Stage 0 - discovery and reconnaissance.

Establishes *how* the data is delivered and writes the strategy to
``out/00_discovery.json``. Every later stage reads its pointers from that
file rather than hardcoding them, so a site change is a one-file fix.
"""

from __future__ import annotations

import asyncio
from typing import Any

from scraper.checkpoint import atomic_write_json, utc_now
from scraper.context import PipelineContext, StageResult
from scraper.extract import dig, extract_next_data
from scraper.http import RateLimiter, build_client, fetch
from scraper.logging_conf import get_logger
from scraper.settings import config_section

log = get_logger(__name__)

ARTIFACT = "00_discovery.json"


def parse_robots(text: str) -> dict[str, Any]:
    """Extract the ``User-agent: *`` group and the sitemap list.

    Written by hand rather than with ``urllib.robotparser`` because we
    want to *report* the rules, not just test one URL against them.
    """
    disallow: list[str] = []
    allow: list[str] = []
    sitemaps: list[str] = []
    crawl_delay: float | None = None
    in_star = False

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()

        if key == "sitemap":
            sitemaps.append(value)
            continue
        if key == "user-agent":
            in_star = value == "*"
            continue
        if not in_star:
            continue
        if key == "disallow" and value:
            disallow.append(value)
        elif key == "allow" and value:
            allow.append(value)
        elif key == "crawl-delay":
            try:
                crawl_delay = float(value)
            except ValueError:
                pass

    return {
        "wildcard_disallow": disallow,
        "wildcard_allow": allow,
        "crawl_delay": crawl_delay,
        "sitemaps": sitemaps,
    }


def path_is_allowed(path: str, disallow: list[str]) -> bool:
    """Check a path against the wildcard Disallow patterns.

    Supports the ``*`` wildcard, which the site uses for its
    ``/arriendo/*-y-*`` multi-filter rules.
    """
    for pattern in disallow:
        if "*" in pattern:
            head, _, tail = pattern.partition("*")
            if path.startswith(head) and tail and tail in path:
                return False
            if path.startswith(head) and not tail:
                return False
        elif path.startswith(pattern):
            return False
    return True


def root_url(operation: str, city: str) -> str:
    """Build the unsharded listing URL for this run."""
    cfg = config_section("site")
    urls = config_section("urls")
    city_cfg = config_section("cities")[city]
    return cfg["base_url"] + urls["root"].format(
        operation=config_section("operations")[operation]["slug"],
        city=city_cfg["slug"],
        state=city_cfg["state"],
    )


def _vocabularies(next_data: dict[str, Any]) -> dict[str, Any]:
    """Harvest the site's own filter tables from apolloState."""
    pointers = config_section("extraction")["pointers"]
    apollo = dig(next_data, pointers["apollo_state"], {}) or {}
    out: dict[str, Any] = {}
    for key, node in apollo.items():
        if not key.startswith("Filter:") or not isinstance(node, dict):
            continue
        options = node.get("options")
        if isinstance(options, list) and options:
            out[key.split(":", 1)[1]] = options
    return out


async def _discover(ctx: PipelineContext) -> dict[str, Any]:
    """Fetch robots.txt and one listing page, then describe the site."""
    site = config_section("site")
    extraction = config_section("extraction")
    limiter = RateLimiter(0.5, 1.0)

    async with build_client() as client:
        robots_res = await fetch(client, site["robots_url"], limiter)
        robots = parse_robots(robots_res.text)

        target = root_url(ctx.operation, ctx.city)
        page_res = await fetch(client, target, limiter)

    # Fails loudly on a site redesign instead of degrading silently.
    next_data = extract_next_data(page_res.text, extraction["script_selector"])

    pointers = extraction["pointers"]
    listings = dig(next_data, pointers["listings"])
    paginator = dig(next_data, pointers["paginator"])
    if not isinstance(listings, list) or not isinstance(paginator, dict):
        raise RuntimeError(
            "__NEXT_DATA__ was found but the listing pointers do not "
            f"resolve ({pointers['listings']!r} / "
            f"{pointers['paginator']!r}). Update scraper/config.yaml."
        )

    vocab = _vocabularies(next_data)
    from urllib.parse import urlparse

    return {
        "discovered_at": utc_now().isoformat(),
        "base_url": site["base_url"],
        "extraction_strategy": extraction["strategy"],
        "requires_browser": False,
        "browser_rationale": (
            "All listing fields are present in the embedded "
            "__NEXT_DATA__ JSON, so Selenium/Playwright is not needed."
        ),
        "script_selector": extraction["script_selector"],
        "pointers": pointers,
        "paginator_keys": extraction["paginator_keys"],
        "build_id": dig(next_data, pointers["build_id"]),
        "robots": robots,
        "robots_allows_target": path_is_allowed(
            urlparse(target).path, robots["wildcard_disallow"]
        ),
        "probe": {
            "url": target,
            "effective_url": page_res.effective_url,
            "status": page_res.status,
            "listings_on_page": len(listings),
            "paginator": paginator,
        },
        "vocabularies": vocab,
        "vocabulary_counts": {k: len(v) for k, v in vocab.items()},
    }


def run(ctx: PipelineContext) -> StageResult:
    """Execute stage 0."""
    out_path = ctx.path(ARTIFACT)
    if out_path.exists() and not ctx.force:
        log.info("stage0.skip", reason="artifact exists", path=str(out_path))
        return StageResult(name="discovery", records=1, skipped=True)

    report = asyncio.run(_discover(ctx))
    atomic_write_json(out_path, report)

    paginator = report["probe"]["paginator"]
    delay = report["robots"]["crawl_delay"]
    print("\n--- stage 0: discovery -------------------------------")
    print(f"  strategy        : {report['extraction_strategy']}")
    print(f"  browser needed  : {report['requires_browser']}")
    print(f"  build id        : {report['build_id']}")
    print(f"  robots allows   : {report['robots_allows_target']}")
    print(
        "  crawl-delay     : "
        + (f"{delay}s" if delay else "not specified (using our own)")
    )
    print(f"  disallow rules  : {len(report['robots']['wildcard_disallow'])}")
    print(f"  results total   : {paginator.get('total')}")
    print(
        f"  pages (perPage) : {paginator.get('lastPage')} "
        f"({paginator.get('perPage')}/page)"
    )
    for name, count in sorted(report["vocabulary_counts"].items()):
        print(f"  vocab {name:<16}: {count}")
    print("------------------------------------------------------\n")

    return StageResult(
        name="discovery",
        records=1,
        notes={"total": paginator.get("total")},
    )
