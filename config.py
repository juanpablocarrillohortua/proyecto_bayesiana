"""Configuration Module."""
from pathlib import Path

from pydantic import PostgresDsn, computed_field
from pydantic_core import MultiHostUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):  # pylint: disable=too-few-public-methods
    """App configuration class.

    Scraper tunables are read from the environment or ``.env`` using the
    ``FR_`` prefix, e.g. ``FR_CONCURRENCY=2``. Structural configuration
    (JSON pointers, URL templates, controlled vocabularies) does not live
    here -- see ``scraper/config.yaml``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FR_",
        extra="ignore",
    )

    # --- politeness / throughput -------------------------------------
    concurrency: int = 4
    delay_min_s: float = 1.5
    delay_max_s: float = 3.0
    timeout_s: float = 30.0
    max_retries: int = 4
    max_consecutive_failures: int = 8

    # --- pipeline behaviour ------------------------------------------
    #: Shards deeper than this are split again (see scraper/README.md).
    max_safe_page: int = 40
    #: Cap on pages emitted per shard; None means "no cap".
    max_pages: int | None = None
    checkpoint_every: int = 25
    out_dir: Path = Path("./out")

    # --- network identity --------------------------------------------
    proxy_url: str | None = None
    user_agent: str | None = None


settings = Settings()