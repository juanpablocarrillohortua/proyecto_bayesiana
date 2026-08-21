# FincaRaíz pipeline scraper

A sequential, resumable, checkpointed scraper for
[fincaraiz.com.co](https://www.fincaraiz.com.co). Each stage reads its
predecessor's artifact from disk and writes its own, so any stage runs in
isolation and an interrupted run resumes where it stopped.

> Read [../REVIEW_TOS.md](../REVIEW_TOS.md) before running this at scale.

## One command

```bash
make scrape                          # arriendo / bogota, all stages
make scrape-venta                    # for-sale listings
make scrape PAGES=1-3                # bounded run for development
```

Equivalently, from the repository root:

```bash
python -m scraper run --stage all --operation arriendo --city bogota
```

Each query gets **its own output directory**, `out-<operation>-<city>/`, because
the pipeline refuses to mix two queries' artifacts (see the params guard
below). Override with `make scrape OUT=./somewhere`.

Rough scale, measured live on 2026-08-02:

| Query | Advertised | Pages after sharding | Wall time |
|---|---|---|---|
| `arriendo` / bogota | 25,286 | ~1,143 | ~45 min |
| `venta` / bogota | 44,150 | ~2,000 | ~70 min |

Expect 18–22k distinct listings from a full arriendo run — see *Known
limitations* on why complete coverage is not provable.

## Pipeline

```
                 ┌──────────────────────────────────────────────┐
                 │ scraper/pipeline.py   (owns the stage order)  │
                 └──────────────────────────────────────────────┘
                                      │
  stage 0  discovery ──────────►  out/00_discovery.json
     robots.txt + strategy            (pointers every later stage reads)
                                      │
  stage 1  enumerate ──────────►  out/01_requests.jsonl
     adaptive sharding                out/01_shards.json
                                      │
  stage 2  fetch ──────────────►  out/raw/{request_id}.html
     async, polite, cached            out/02_fetch_log.jsonl
                                      │
  stage 3  parse ──────────────►  out/03_records_raw.jsonl
     pure, no network                 out/03_parse_errors.jsonl
                                      │
  stage 4  normalize ──────────►  out/04_records_clean.jsonl
     COP numerals, vocabularies
                                      │
  stage 5  consolidate ────────►  out/fincaraiz_{op}_{city}_{date}.json
     dedup + pydantic validate        out/fincaraiz_{op}_{city}_{date}.csv
                                      │                out/05_rejected.jsonl
  stage 6  qa ─────────────────►  out/06_qa_report.json
     coverage, null rates
```

Run one stage, a range, or everything:

```bash
python -m scraper run --stage 0        # just discovery
python -m scraper run --stage 2        # just fetch
python -m scraper run --stage 3-5      # parse through consolidate
python -m scraper run --stage all --force   # recompute from scratch
```

## The problem this design exists to solve

The site is a Next.js app and every search page embeds its full result set in
one `<script id="__NEXT_DATA__">` blob, so **no browser automation is needed**
and no per-listing detail fetch is needed either — all requested fields are
already in the search results.

The hard part is coverage. The default sort is `Popularidad` (relevance) and
**that ordering is not stable**, so deep pagination returns overlapping
windows. Measured on `arriendo/bogota`:

```
p1    vs p476 :  0/21 shared      p476  vs p800  : 12/21 shared
p800  vs p1204: 15/21 shared      p1204 vs p1205 : 21/21  (identical)
105 rows pulled from five deep pages  ->  only 52 distinct listings
```

Every one of those responses was a healthy HTTP 200. A naive
`for page in range(1, 1205)` crawl therefore collects far fewer than the
advertised 25,280 listings **and fails invisibly**. The sort cannot be forced:
the `orden` query parameter is ignored, and so is the base64 `hashed` filter
payload (the server re-derives filters from the URL path and echoes back
`Popularidad` regardless).

Shallow pages, on the other hand, are exact — pages 1-3 of a shard shared
nothing, and refetching a page returned identical ids. So stage 1 **splits the
search until every shard is shallow**, on two axes: property type, then
neighbourhood. `max_safe_page` (default 40) is the threshold.

Three consequences worth knowing:

1. **Split verification.** `/arriendo/apartamentos/bogota/bogota-dc/chapinero`
   answers HTTP 200 but redirects back to the unfiltered city page and returns
   the parent's whole result set. Stage 1 checks the post-redirect URL still
   contains the shard token *and* that the result count actually shrank, and
   discards the shard otherwise.
2. **Backstop shards.** About 30% of Bogotá inventory sits in barrios that are
   not one of the seeded localidades, so the neighbourhood split loses them.
   When a split covers under 90% of its parent, stage 1 also emits the
   parent's first `max_safe_page` pages as a `backstop` shard. Stage 5
   deduplicates the overlap; stage 6 excludes backstops from the coverage
   denominator so they cannot flatter the ratio.
3. **Coverage is measured, not assumed.** Stage 6 reports distinct listings
   against what the shards advertised. Treat a low ratio as a signal to lower
   `FR_MAX_SAFE_PAGE` or add neighbourhood seeds.

## Configuration

Two files, deliberately split:

| Where | What | Why |
|---|---|---|
| [`../config.py`](../config.py) + `.env` | rates, concurrency, timeouts, proxy, `max_safe_page`, output dir | deployment tuning, machine-specific, sometimes secret |
| [`config.yaml`](config.yaml) | JSON pointers, URL templates, vocabularies, field map, neighbourhood seeds | describes the *site*; changes when the site changes |

Tunables use the `FR_` prefix:

```bash
FR_CONCURRENCY=2
FR_DELAY_MIN_S=2.0
FR_DELAY_MAX_S=4.0
FR_MAX_SAFE_PAGE=25      # shard harder, better coverage, more requests
FR_MAX_PAGES=3           # cap pages per shard (development)
FR_OUT_DIR=./out
FR_PROXY_URL=http://user:pass@host:port
```

There is exactly one `Settings` object in the project — the scraper imports
the existing root one rather than defining its own. **Run the CLI from the
repository root** so `from config import settings` resolves.

## Adapting when the site changes

No CSS selector, JSON pointer or URL shape is hardcoded in Python. Symptoms
map to config keys:

| Symptom | Fix in `config.yaml` |
|---|---|
| `NextDataMissing: no node matched ...` | `extraction.script_selector` |
| Stage 0 says pointers do not resolve | `extraction.pointers.listings` / `.paginator` |
| Every field is null but pages parse | `field_map` |
| Pagination stops at page 1 | `extraction.paginator_keys`, `urls.page_suffix` |
| A shard 404s or redirects | `urls.*`, `cities.<city>.neighbourhoods` |
| New property type appears | `property_types` |
| A `0` should not have become null | `zero_means_null`, `technical_sheet_map` |

Re-run `--stage 0 --force` first: it re-harvests the site's own filter tables
and prints their sizes, which is the quickest way to see what moved.

## Field mapping

The brief's names on the left, the emitted names on the right.

| Requested | Emitted | Source path |
|---|---|---|
| price | `price_amount` | `price.amount` |
| currencie | `price_currency` | `currency_id` → `COP` |
| built_area | `area_built_m2` | `m2Built` |
| private_area | `area_private_m2` | `m2apto` |
| stratum | `stratum` | `stratum` |
| rooms | `bedrooms` | `bedrooms` — **not** `rooms` |
| bathrooms | `bathrooms` | `bathrooms` |
| # parkings | `parking_spaces` | `garage` |
| administration | `admin_fee` | `commonExpenses.amount` |
| years since built | `age_bracket_code` / `age_bracket_label` | `antiquity` + sheet |
| # floor | `floor` | `floor` |
| state | `construction_state` | `constStatesID` → Usado/Nuevo |
| property type | `property_type` | `property_type.id` → vocabulary |
| latitude | `latitude` | `latitude`, WKT fallback |
| longitude | `longitude` | `longitude` |
| locality | `locality` | `locations.locality[0].name` |
| direction | `address_text` | `address`, gated on `showAddress` |
| amennities | `amenities[]` | `facilities[]` → `{id,name,group}` |

### Three traps encoded in the parser

- **`rooms` is not the bedroom count.** It was `0` on 12 of 21 sampled
  listings while `bedrooms` held the real value. The site's own rendered sheet
  labels `bedrooms` as *Habitaciones*.
- **`0` usually means "not stated".** `garage=0` and `floor=0` appear
  alongside *blank* rows in the rendered `technicalSheet`. That sheet is used
  as the authority: a `0` with no sheet row becomes `null`. Same for
  `commonExpenses.amount = 0`, which means undisclosed, not free.
- **`antiquity` is a bracket code, not a year count.** Raw value `5` renders
  as *"más de 30 años"*, and `construction_year` was null on every sample. So
  `age_years` is deliberately always `null` — the site never publishes an
  exact year, and emitting a bracket midpoint would feed fake precision into a
  Bayesian model. Use `age_bracket_code` as an ordinal factor instead.

## Checkpointing and resume

Two independent layers:

- **Record level.** Every stage appends to its `.jsonl` and `flush()` +
  `fsync()` every `FR_CHECKPOINT_EVERY` records (default 25), updating
  `out/.checkpoint/{stage}.json` atomically. The `.jsonl` itself is the source
  of truth on resume: it is re-read to rebuild the done-set, and a torn final
  line from a killed process is discarded. Stage 1 additionally caches shard
  probes, so a re-run costs no network. Stage 2 treats the presence of
  `out/raw/{id}.html` as its unit of work and so resumes even with no
  checkpoint at all.
- **Stage level.** `pipeline.py` skips any stage whose output already exists
  and records progress in `out/.checkpoint/pipeline.json`. It also refuses to
  resume when the run parameters changed, so an `arriendo` directory can never
  silently absorb `venta` listings — pass `--force` if that is what you want.

Ctrl-C at any point, rerun the same command, and it continues.

## Testing

```bash
make test          # or: python -m pytest tests -q
```

Stages 3 and 4 are covered against trimmed real fixtures captured from the
live site, including the awkward cases: a `0` with a blank sheet row, a
private seller, a hidden address, and a listing needing the WKT coordinate
fallback. A `conftest.py` autouse fixture monkeypatches `httpx` to raise, so
the suite cannot reach the network even by accident.

## Known limitations

- **Coverage is not guaranteed to be 100%.** The site's unstable ordering
  makes complete enumeration impossible to prove; stage 6 reports what was
  actually achieved rather than claiming completeness.
- Only Bogotá ships neighbourhood seeds. Other cities fall back to type-only
  sharding, which is shallower — add seeds to `cities.<city>.neighbourhoods`.
- `perPage` is fixed at 21 server-side; no page-size override was found.
- Price-band sharding is not implemented: no URL form for it was found, and
  the `hashed` filter payload is ignored by the server.
- A full arriendo/Bogotá run is roughly 1,100–1,200 requests and about an hour
  at the default polite rate.
