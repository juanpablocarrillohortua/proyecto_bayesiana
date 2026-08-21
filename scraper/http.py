"""Async HTTP with polite rate limiting, retries and block detection."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
)

from config import settings
from scraper.logging_conf import get_logger

log = get_logger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

UA_POOL = [
    DEFAULT_UA,
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like "
    "Gecko) Chrome/123.0.0.0 Safari/537.36",
]

#: Markers that indicate an anti-bot interstitial rather than content.
BLOCK_MARKERS = (
    "cf-browser-verification",
    "captcha-delivery",
    "g-recaptcha",
    "Attention Required! | Cloudflare",
    "Just a moment...",
)


class BlockedError(RuntimeError):
    """A hard anti-bot block was detected.

    We stop the run rather than attempt to evade it.
    """


class RetryableStatusError(Exception):
    """A 429 or 5xx worth retrying."""

    def __init__(self, status: int, retry_after: float | None = None):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after


@dataclass
class FetchResult:
    """Outcome of one HTTP GET."""

    url: str
    effective_url: str
    status: int
    text: str
    elapsed_ms: int


class RateLimiter:
    """Serialises a jittered delay between request starts."""

    def __init__(self, min_s: float, max_s: float) -> None:
        self.min_s = min_s
        self.max_s = max(min_s, max_s)
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        """Sleep for a random delay inside the configured window."""
        async with self._lock:
            await asyncio.sleep(random.uniform(self.min_s, self.max_s))


def build_headers(rotate_ua: bool = False) -> dict[str, str]:
    """Realistic browser headers, Colombian Spanish locale."""
    agent = settings.user_agent or (
        random.choice(UA_POOL) if rotate_ua else DEFAULT_UA
    )
    return {
        "User-Agent": agent,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
        # Accept-Encoding is deliberately NOT set here. httpx advertises
        # exactly the codecs it can decode; hardcoding "br" without the
        # brotli package installed makes the server send brotli that
        # httpx hands back as undecoded bytes, and every selector then
        # silently misses. Let httpx negotiate it.
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Connection": "keep-alive",
    }


def build_client(rotate_ua: bool = False) -> httpx.AsyncClient:
    """Create the shared client. Cookies persist for the whole run."""
    kwargs: dict[str, Any] = {
        "headers": build_headers(rotate_ua),
        "timeout": httpx.Timeout(settings.timeout_s),
        "follow_redirects": True,
        "cookies": httpx.Cookies(),
    }
    if settings.proxy_url:
        kwargs["proxy"] = settings.proxy_url
    try:
        return httpx.AsyncClient(http2=True, **kwargs)
    except ImportError:
        # h2 not installed; HTTP/1.1 works fine for this site.
        return httpx.AsyncClient(**kwargs)


def _detect_block(status: int, text: str) -> str | None:
    """Return a reason string when the response looks like a block."""
    if status in (401, 403):
        return f"HTTP {status}"
    head = text[:4000]
    for marker in BLOCK_MARKERS:
        if marker in head:
            return f"anti-bot marker {marker!r}"
    return None


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse the ``Retry-After`` header when present."""
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    limiter: RateLimiter,
) -> FetchResult:
    """GET one URL with backoff on 429/5xx and hard-block detection.

    Raises:
        BlockedError: on an anti-bot response; the caller must stop.
        RetryableStatusError: when retries are exhausted.
    """
    attempts = max(1, settings.max_retries)
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(attempts),
        retry=retry_if_exception_type(
            (RetryableStatusError, httpx.TransportError)
        ),
        reraise=True,
    ):
        with attempt:
            await limiter.wait()
            response = await client.get(url)
            body = response.text

            reason = _detect_block(response.status_code, body)
            if reason:
                raise BlockedError(
                    f"blocked while fetching {url} ({reason}). Stopping "
                    "instead of attempting to evade. Slow the crawl "
                    "(FR_DELAY_MIN_S / FR_CONCURRENCY) and retry later."
                )

            if response.status_code == 429 or response.status_code >= 500:
                delay = _retry_after_seconds(response)
                number = attempt.retry_state.attempt_number
                wait_s = (
                    delay
                    if delay is not None
                    else min(60.0, (2**number) + random.uniform(0, 1.5))
                )
                log.warning(
                    "http.retry",
                    url=url,
                    status=response.status_code,
                    sleep_s=round(wait_s, 2),
                    attempt=number,
                )
                await asyncio.sleep(wait_s)
                raise RetryableStatusError(response.status_code, delay)

            elapsed = int(response.elapsed.total_seconds() * 1000)
            return FetchResult(
                url=url,
                effective_url=str(response.url),
                status=response.status_code,
                text=body,
                elapsed_ms=elapsed,
            )
    raise RetryableStatusError(0)  # pragma: no cover - reraise fires first
