<div style="width:100%; background-color:#00522c; border-bottom:3px solid #34a853; padding:20px 0; text-align:center;">
  <img src="https://pattern-lab-externado-prod.web.app/images/logo-uec.svg" alt="Banner del proyecto" width="40%">
</div>

# proyecto_finca_raiz_colombia

Analysis of the Bogotá residential property market, using listing data
collected from [FincaRaíz](https://www.fincaraiz.com.co).

Universidad Externado de Colombia — cuarto semestre.

## Status

| Part | State |
|---|---|
| Data acquisition ([scraper/](scraper/)) | **Working.** Full venta run: 31,252 listings at 91.3% coverage |
| Cleaning / validation | **Working.** Runs inside the scraper pipeline (stages 4–6) |
| Shared helpers ([utils/](utils/)) | **In progress.** Plotting toolkit, normality panel, geo checks, map |
| Exploratory analysis ([notebooks/](notebooks/)) | **In progress.** Univariate pass and data-quality audit |
| Bayesian models ([src/](src/)) | Not started |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere

make install                    # runtime dependencies
make install-dev                # ruff, for the PEP 8 gate

make scrape                     # arriendo listings  (~45 min)
make scrape-venta               # for-sale listings  (~70 min)
```

`make help` lists every target. Run commands from the repository root — the
scraper imports the project-wide settings object from [config.py](config.py).

Notebooks are the exception: launch them from `notebooks/`. They put the
repository root on `sys.path` themselves so that `from utils... import ...`
resolves.

### Installing make

Python ships with the virtualenv; `make` does not, and Windows has none by
default. Check with `make --version` before installing anything.

**Windows** — either package manager works, in an *administrator* terminal:

```powershell
winget install ezwinports.make    # GNU Make 4.4.1
choco install make                # if you already use Chocolatey
```

Open a new terminal afterwards so the updated `PATH` takes effect. There is no
POSIX shell involved: the recipes run under `cmd.exe`, which is why every one of
them is a Python invocation.

**macOS** — `make` comes with the Xcode command line tools:

```bash
xcode-select --install            # GNU Make 3.81, enough for this project
brew install make                 # optional: GNU Make 4.x, installed as gmake
```

If Homebrew's version is the one you want, either call `gmake` instead of
`make`, or put its `gnubin` directory ahead of `/usr/bin` on your `PATH`.

**Skipping make entirely.** Nothing here depends on it. Every target is a thin
wrapper around one command, so you can run that command yourself:

```bash
.venv/Scripts/python.exe -m pytest tests -q                  # = make test
.venv/Scripts/python.exe -m ruff check scraper src utils     # = make lint-py
.venv/Scripts/python.exe -m scraper run --stage all \
    --operation arriendo --city bogota --out ./out-arriendo-bogota   # = make scrape
```

Pass `--out` yourself when you do: the Makefile supplies
`./out-<operation>-<city>`, and the pipeline refuses to mix two queries in one
directory. Read the [Makefile](Makefile) to see what any other target expands to.

## Layout

```
proyecto_finca_raiz_colombia/
├── README.md                   # This file — what the project is and how to run it
├── CLAUDE.md                   # Orientation for Claude Code
├── REVIEW_TOS.md               # Read this before scraping at scale
├── LICENSE                     # MIT
├── .gitignore                  # Datasets and pipeline output stay out of git
├── .env                        # FR_* scraper tunables (gitignored, optional)
├── Makefile                    # Quality gate + scraper entry points
├── config.py                   # The single Settings object, reads .env
├── ruff.toml                   # PEP 8 rules (line length 79)
├── requirements.txt            # Runtime dependencies
├── requirements/
│   └── dev.txt                 # Lint tooling (ruff)
├── scraper/                    # The data pipeline — see scraper/README.md
│   ├── README.md               # Field map, sharding, symptom→config table
│   ├── pipeline.py             # Stage orchestration; one command runs 0–6
│   ├── stages/                 # s0_discovery … s6_qa, one module per stage
│   ├── config.yaml             # Everything site-specific lives here
│   ├── models.py               # Pydantic schema of a listing and the artifact
│   └── http.py, extract.py, …  # Fetching, parsing, checkpointing
├── tests/
│   └── scraper/                # 97 offline tests over saved fixtures
├── tools/                      # clean.py, mk_help.py — called by the Makefile
├── data/                       # Scraped datasets (gitignored, run make scrape)
├── notebooks/                  # Exploratory analysis
│   └── Basic_EDA.ipynb         # Univariate pass over the venta dataset
├── utils/                      # Shared helpers — see utils/README.md
│   ├── README.md               # What every helper does
│   ├── plotting.py             # EDAPlotter, the chart toolkit
│   ├── triple_plot.py          # Normality panel
│   ├── geo.py                  # Coordinate sanity checks
│   └── map_graph.py            # Listings map
├── src/                        # Model code (empty for now)
├── outputs/
│   ├── tables/                 # CSV files holding results and summaries
│   └── plots/                  # Figures and visualisations
└── out-<operation>-<city>/     # Pipeline working directories (gitignored)
```

## The dataset

The scraper emits a validated JSON artifact plus a flat CSV per query. Measured
on the completed `venta` / `bogota` run:

```
pages fetched   1,935          fetch errors        4
parsed          39,206         parse errors        0
duplicates      7,954          schema rejections   0
final           31,252         coverage           91.3%
```

Composition of that run: apartamento 12,337 · casa 9,135 · oficina 2,488 ·
bodega 2,185 · local 1,693 · apartaestudio 1,313.

Datasets are gitignored (`*.csv`, `out-*/`), so a clone has no data until you
run `make scrape`. The current working file is
`data/fincaraiz_venta_bogota_20260803.csv`.

### Variable dictionary

The 31 columns of the CSV, in order.

| Variable | Unit | Description |
|---|---|---|
| `listing_id` | id | Unique identifier of the listing on FincaRaíz. |
| `operation` | category | Transaction advertised: `venta` (sale) or `arriendo` (rent). |
| `property_type` | category | Class of property: `apartamento`, `casa`, `oficina`, `bodega`, `local`, `lote`, and similar. |
| `price_amount` | COP | Asking price published in the listing. |
| `price_currency` | category | Currency the asking price is quoted in. |
| `admin_fee` | COP / month | Monthly building administration fee (*administración*). |
| `total_monthly_cost` | COP / month | Rent plus administration fee; defined for rental listings. |
| `price_per_m2` | COP / m² | Asking price divided by built area. |
| `area_built_m2` | m² | Constructed area of the property. |
| `area_private_m2` | m² | Private interior area of the unit. |
| `bedrooms` | count | Number of bedrooms. |
| `bathrooms` | count | Number of bathrooms. |
| `parking_spaces` | count | Number of parking spaces included with the property. |
| `stratum` | ordinal 1–6 | Colombian socio-economic stratum assigned to the address. |
| `floor` | floor number | Floor of the building on which the unit sits. |
| `floors_total` | count | Number of floors in the building. |
| `age_bracket_code` | ordinal 0–5 | Age of the property as a bracket: 0 brand new, 1 under 1 year, 2 = 1–8, 3 = 9–15, 4 = 16–30, 5 over 30 years. |
| `age_bracket_label` | category | Spanish text label of the age bracket. |
| `construction_state` | category | Condition declared in the listing: `Usado`, `Excelente estado`, `En construcción`, and similar. |
| `city` | name | Municipality where the property is located. |
| `locality` | category | Bogotá *localidad* containing the property. |
| `neighborhood` | name | Neighbourhood (*barrio*) of the property. |
| `address_text` | text | Street address as published in the listing. |
| `latitude` | degrees (WGS84) | Latitude of the property's map location. |
| `longitude` | degrees (WGS84) | Longitude of the property's map location. |
| `amenities` | list (`\|`-separated) | Amenities and features attributed to the property or its building. |
| `agency_name` | name | Real-estate agency publishing the listing. |
| `publisher_type` | category | Kind of publisher: `inmobiliaria` (agency) or `desarrollador` (developer). |
| `published_at` | date (YYYY-MM-DD) | Date the listing was first published. |
| `updated_at` | date (YYYY-MM-DD) | Date the listing was last updated. |
| `source_url` | URL | Web address of the listing. |

The JSON artifact carries a few fields the CSV flattens away — notably the
`price_currency_is_foreign` flag, so from the CSV, judge currency on
`price_currency` before pooling prices. It also keeps `amenities` as
`{id, name, group}` objects instead of pipe-joined names.

### Three things to know before modelling

- **`age_years` is always null, deliberately.** The site publishes an age
  *bracket* ("más de 30 años"), never an exact year. Use `age_bracket_code` as
  an ordinal factor; a midpoint would be invented precision.
- **Nulls are informative, not missing at random.** The site writes `0` for
  values it does not publish, and the scraper restores those to null rather
  than treating them as real zeros. The rate depends on the operation, which is
  itself a signal:

  | Field | null in `arriendo` | null in `venta` |
  |---|---|---|
  | `admin_fee` | 75.6% | 43.5% |
  | `floor` | 55.9% | 50.5% |
  | `parking_spaces` | 63.5% | 34.5% |

  Treat these as *not disclosed* and consider modelling the disclosure
  mechanism rather than dropping or zero-filling the rows.
- **Coverage is measured, not assumed.** 91.3% on the venta run. The remainder
  is unreachable because the site's result ordering is unstable at depth — see
  the explanation in [scraper/README.md](scraper/README.md). Any market-level
  total from this data is a lower bound.

## Analysis

### Helpers at a glance

Four modules in [utils/](utils/), documented in [utils/README.md](utils/README.md).

| Module | What it does |
|---|---|
| [plotting.py](utils/plotting.py) | `EDAPlotter`, the whole chart set behind one class: hand it the DataFrame, ask for a column by name. |
| [triple_plot.py](utils/triple_plot.py) | Histogram, boxplot and Q-Q in one panel with a normality test, on raw, log-scaled or log-transformed data. |
| [geo.py](utils/geo.py) | Checks latitude and longitude against a Cundinamarca bounding box and counts what falls outside it. |
| [map_graph.py](utils/map_graph.py) | Draws the listings on an interactive Folium map with clustered markers. |

### Where the notebook is

[notebooks/Basic_EDA.ipynb](notebooks/Basic_EDA.ipynb) is a univariate pass over
the venta dataset. It currently produces:

- bar charts for the categorical and discrete columns
- normality panels for the continuous columns, run three ways — untransformed,
  on a log axis, and on log-transformed data
- a coordinate check against the Cundinamarca box
- a missing-value matrix
- Pearson and Kendall correlation heatmaps
- a map of the listing locations

No conclusions have been written up yet, and no model has been fitted.

## Code quality

```bash
make quality     # ruff check + format check across notebooks, scraper, src, utils
make test        # 97 offline tests
make validate    # both
make format      # auto-fix
```

`make quality` checks notebooks cell by cell. Line length is PEP 8's 79 —
raise it in [ruff.toml](ruff.toml) if it fights the analysis code.

## Before scraping at scale

Read [REVIEW_TOS.md](REVIEW_TOS.md). It covers the site's terms of use and
Ley 1581 de 2012 / Decreto 1377 de 2013. In short: the scraper never records
contact details, defaults to a conservative rate, and stops rather than
attempting to evade a block — but holding and publishing the data is still your
call to make deliberately.

## License

[MIT](LICENSE) © 2026 Juan Pablo Carrillo Hortua
