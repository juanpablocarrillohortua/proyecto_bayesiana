# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Coursework for Estadística Bayesiana (Universidad Externado): a Bayesian analysis
of the Bogotá residential property market, on listing data this repo scrapes
itself from fincaraiz.com.co.

Two halves, at very different maturity:

- **Data acquisition** — [scraper/](scraper/), finished and tested (97 offline
  tests). Cleaning and validation happen inside it, at stages 4–6.
- **Analysis** — [notebooks/](notebooks/) and [utils/](utils/) are in progress
  (univariate EDA); [src/](src/) is empty, no Bayesian model has been written
  yet, and there is no PyMC/Stan dependency installed.

## Commands

Run everything from the repository root. The Makefile picks `.venv/Scripts/python.exe`
when present, so `make` needs no activated venv; direct invocations do.

```bash
make help                       # every target, generated from the ## comments
make install                    # runtime deps;  make install-dev = ruff

make quality                    # clean + ruff check + ruff format --check  (read-only)
make format                     # auto-fix
make test                       # 97 offline tests
make validate                   # quality + test

make scrape                     # arriendo/bogota, stages 0-6, ~45 min
make scrape-venta               # venta/bogota,   ~70 min
make scrape PAGES=1-3           # bounded run for development
make scrape OPERATION=venta CITY=bogota STAGE=3-5
```

One test, one stage, one lint target:

```bash
.venv/Scripts/python.exe -m pytest tests/scraper/test_s4_normalize.py -q
.venv/Scripts/python.exe -m pytest tests/scraper/test_s3_parse.py::test_parse_page_yields_one_record_per_listing -q
.venv/Scripts/python.exe -m scraper run --stage 3 --operation venta --city bogota --out ./out-venta-bogota
.venv/Scripts/python.exe -m ruff check utils          # lint one directory
```

`make quality` runs `clean` first, which deletes `__pycache__`, `.ruff_cache`,
`.pytest_cache` and `.ipynb_checkpoints` — expect that, it is not a bug.

## Scraper architecture

`scraper/pipeline.py` is the only module that knows the stage order. Each stage
is `run(ctx: PipelineContext) -> StageResult`, reads its predecessor's artifact
from disk and writes its own; stages never call each other. That is what makes
any stage independently runnable and any run resumable.

```
0 discovery   robots + site vocabularies  -> 00_discovery.json
1 enumerate   adaptive sharding           -> 01_requests.jsonl, 01_shards.json
2 fetch       async, polite, cached       -> raw/{request_id}.html, 02_fetch_log.jsonl
3 parse       pure, no network            -> 03_records_raw.jsonl
4 normalize   COP numerals, vocabularies  -> 04_records_clean.jsonl
5 consolidate dedup + pydantic validate   -> fincaraiz_{op}_{city}_{date}.{json,csv}
6 qa          coverage, null rates        -> 06_qa_report.json
```

Three invariants worth not breaking:

- **Stages 3 and 4 are pure.** No network, no clock, no filesystem beyond their
  own artifact. `tests/scraper/conftest.py` monkeypatches `httpx` to raise, so a
  regression here fails the suite loudly.
- **One output directory per query.** `_guard_params` in `pipeline.py` hashes
  operation/city/page-window and refuses to resume a directory belonging to a
  different query, so an `arriendo` run can never absorb `venta` listings. Hence
  `out-<operation>-<city>/`, not a shared `out/`.
- **Resume is driven by artifacts, not just checkpoints.** The `.jsonl` is the
  source of truth (a torn last line from a killed process is discarded), and
  stage 2 treats the presence of `raw/{id}.html` as its unit of work. A stage
  whose output already exists is skipped unless `--force`.

### Why the sharding exists

The site is Next.js: the whole result set sits in one `<script id="__NEXT_DATA__">`
blob at `props.pageProps.fetchResult.searchFast.data[]`, so **no browser
automation and no detail-page fetch are needed**. The hard part is coverage —
the default `Popularidad` sort is unstable, so deep pages return overlapping
windows while answering HTTP 200 the whole way (five deep pages once gave 105
rows and 52 distinct listings). The sort cannot be forced; `?orden=` and the
base64 `?hashed=` payload are both ignored server-side.

So stage 1 splits the search — by property type, then neighbourhood — until
every shard is shallower than `max_safe_page` (40). It verifies each split
actually happened (a shard URL can 200 but redirect back to the unfiltered city
page), and emits `backstop` shards when a split covers under 90% of its parent,
because ~30% of Bogotá inventory sits in unseeded barrios. Stage 6 excludes
backstops from the coverage denominator. **Coverage is reported, never assumed**
— 91.3% on the completed venta run.

### Configuration is split on purpose

| Where | What | Changes when |
|---|---|---|
| [config.py](config.py) + `.env`, `FR_` prefix | concurrency, delays, timeouts, retries, `max_safe_page`, `out_dir`, proxy | you tune a deployment |
| [scraper/config.yaml](scraper/config.yaml) | JSON pointers, URL templates, field map, vocabularies, neighbourhood seeds | *the site* changes |

No CSS selector, JSON pointer or URL shape is hardcoded in Python — if scraped
fields go null or counts collapse, fix `config.yaml`, and run `--stage 0 --force`
first to re-harvest the site's own filter tables. `scraper/README.md` has the
symptom→config-key table.

There is exactly **one** `Settings` object in the project, at the repo root;
`scraper/settings.py` only loads the YAML. `scraper/cli.py` does
`from config import settings`, so the CLI must run from the repository root.

## Data semantics that constrain the modelling

These are decisions already made in the parser; do not "fix" them in the
analysis.

- **`age_years` is always null, deliberately.** The site publishes a bracket
  ("más de 30 años"), never a year. Use `age_bracket_code` as an ordinal factor —
  a bracket midpoint would be invented precision.
- **Nulls are informative, not missing at random.** The site writes `0` for
  values it does not publish, and stage 4 restores those to null using the
  rendered `technicalSheet` as the authority. `admin_fee` is null on 43.5% of
  venta rows and 75.6% of arriendo; `floor` ~50%; `parking_spaces` 34.5%/63.5%.
  Model the disclosure mechanism rather than zero-filling.
- **`bedrooms`, not `rooms`.** The site's `rooms` field was 0 on 12 of 21 sampled
  listings while `bedrooms` held the real value.
- **Not every listing is quoted in COP.** Check before pooling prices — via
  `price_currency_is_foreign` in the JSON artifact, or `price_currency` in the
  CSV, which does not carry that flag.
- **Contact details are never collected**, by design — see [REVIEW_TOS.md](REVIEW_TOS.md)
  before scaling up a run or changing city. The scraper halts on a block rather
  than evading it; do not add captcha solving, IP rotation or header spoofing.
- In the CSV, `amenities` is pipe-joined names (`"Ascensor|Gimnasio"`); the JSON
  artifact keeps them as `{id,name,group}` objects.

Datasets are gitignored (`*.csv`, `out-*/`), so a fresh clone has no data until
`make scrape` runs. The current working dataset is
`data/fincaraiz_venta_bogota_20260803.csv` (31,252 listings).

## Analysis code

[utils/plotting.py](utils/plotting.py) is a large single-class toolkit
(`EDAPlotter`, ~9.5k lines): every method resolves a column to
numeric/categorical/datetime through one `resolve_kind` and renders it through
one `StyleConfig` inside a scoped `rc_context`, so importing it never touches
global matplotlib state and callers pass no styling arguments. Public surface is
`histplot`, `boxplot`, `barplot`, `curveplot`, `scatterplot`, `qqplot`,
`corr_heatmap`, `triple_plot`, `summary_grid`, `report_numeric`,
`run_normality_test`. Add plots as methods there rather than starting a second
plotting module. [utils/triple_plot.py](utils/triple_plot.py) holds the
standalone `normality_report` panel; [utils/geo.py](utils/geo.py) holds the
Cundinamarca bounding-box coordinate sanity check; [utils/map_graph.py](utils/map_graph.py)
holds `create_map`, a Folium marker-cluster map (one marker per row, so filter
before calling it on all 31k listings). [utils/README.md](utils/README.md)
documents all four.

Notebooks run from `notebooks/`, so they prepend the parent directory to
`sys.path` before `from utils... import ...`. Figure text is Spanish (the
reader's language); code, docstrings and diagnostics are English.

## Conventions and known rough edges

- **`make` recipes execute under `cmd.exe`** on this machine — there is no `sh`
  on PATH. No `find`/`awk`/`rm`/`grep`, and a tab-indented `#` line is *not* a
  comment, it crashes make. Keep every recipe a Python invocation and put real
  logic in [tools/](tools/).
- **ruff is the only linter**, configured in [ruff.toml](ruff.toml) (standalone —
  this project is not a package). `ruff check` alone does **not** cover all of
  PEP 8: the E1xx/E2xx/E3xx layout rules are preview-gated, so `make quality`
  pairs it with `ruff format --check`. Run both.
- `unfixable = ["F401", "F811", "F841"]` is deliberate — ruff classifies those
  fixes as *safe*, and `--fix` once emptied a file of not-yet-used imports.
  They are still reported by `make lint`; only auto-deletion is off.
- `line-length = 79` is tight for pandas chains and is the knob expected to
  cause friction; raise it in `ruff.toml` if it fights the analysis code.
- Notebooks are linted cell by cell (`notebooks/x.ipynb:cell 2:1:1`, markdown
  cells counted); `E402` and `F401` are ignored there only.
- **`requirements.txt` must stay UTF-8.** It was UTF-16-LE with a BOM until
  2026-08-07 (a PowerShell `>` redirect wrote it that way) and was missing the
  analysis imports; it is now a plain `pip freeze` covering the scraper *and*
  `utils/`/notebooks. Regenerate it with
  `.venv/Scripts/python.exe -m pip freeze`, never with a shell redirect.
- Commit messages follow `type(scope): summary` (`feat(scraper):`, `chore(notebooks):`, `eda:`).
