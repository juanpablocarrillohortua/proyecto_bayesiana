# utils/

Shared helpers for the exploratory analysis. Four modules, each doing one job:

| Module | What it gives you |
|---|---|
| [plotting.py](plotting.py) | `EDAPlotter` — the whole chart set behind one class |
| [triple_plot.py](triple_plot.py) | `normality_report` — a three-panel distribution diagnostic |
| [geo.py](geo.py) | coordinate sanity checks against a Cundinamarca bounding box |
| [map_graph.py](map_graph.py) | the listings drawn on an interactive map |

Notebooks run from `notebooks/`, so they put the repository root on `sys.path`
before importing:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

from utils.plotting import EDAPlotter
from utils.triple_plot import normality_report
from utils.geo import count_outside_bogota, outside_bbox_mask
from utils.map_graph import create_map
```

Code and docstrings are in English; the text drawn on the figures is Spanish,
because that is the language of whoever reads the chart.

---

## plotting.py

One class, `EDAPlotter`. You give it the DataFrame once, then ask for charts by
column name:

```python
plotter = EDAPlotter(df)
plotter.barplot(x="property_type")
plotter.corr_heatmap(annot=True)
```

It works out on its own whether a column is a number, a category or a date, and
takes care of the styling, so you never pass colours, sizes or fonts. Every
method also accepts `df=` if you want to chart something other than the frame
you constructed it with.

### Charts

| Method | What it draws |
|---|---|
| `histplot` | The distribution of one variable. |
| `boxplot` | A number compared across the levels of a category. |
| `barplot` | Counts per category, or a measure compared across categories. |
| `curveplot` | How a value evolves along an ordered axis. |
| `scatterplot` | The relationship between two numbers. |
| `qqplot` | Whether a variable follows the distribution you expect. |
| `corr_heatmap` | How every numeric column relates to every other one. |
| `triple_plot` | Histogram, boxplot and Q-Q for one number, in a single call. |
| `summary_grid` | One suitable panel per column, laid out in a grid. |
| `plot` | Calls any of the above by name, so a loop or a config can drive it. |

### Numbers and words

| Method | What it gives you |
|---|---|
| `run_normality_test` | Runs a normality test and says what the result means. |
| `describe_distribution` | Summary statistics for one variable, plus that test. |
| `verdict_text` | The normality verdict written out in plain language. |
| `report_numeric` | Runs `triple_plot` over every numeric column at once. |

### How a column is interpreted

| Method | What it does |
|---|---|
| `resolve_kind` | Decides whether a column is a number, a category or a date. |
| `is_numeric` | Quick yes/no version of that decision. |
| `is_categorical` | The same, for categories. |
| `ordered_levels` | Decides the order the levels of a category appear in. |

You can override any of it — pass `treat_as` to a method, or name the columns
when you build the plotter, when a code like `stratum` should be read as a
category rather than a number.

### Colours

| Method | What it does |
|---|---|
| `color_map` | The colour each level of a column gets, kept stable across charts. |
| `set_color_map` | Pins your own colours for one column. |
| `reset_color_cache` | Forgets every colour assigned so far. |

### Supporting pieces

| Object | What it is |
|---|---|
| `StyleConfig` | The single place the look is defined — palettes, fonts, sizes. |
| `TransformMeta` | Remembers which transformation was applied, so axis labels stay honest. |

---

## triple_plot.py

`normality_report(df, col, ...)` draws a 1×3 panel — histogram, boxplot, Q-Q
plot — and runs a normality test underneath it. It returns a dict of the
statistics plus the figure, so the numbers are usable and not only readable off
the image.

### `log_scale` and `log_transformation` are not the same thing

They are mutually exclusive, and passing both raises — enabling both would apply
the logarithm twice to the same view.

| | `log_scale=True` | `log_transformation=True` |
|---|---|---|
| The data | untouched | replaced by log10(x) |
| The axes | drawn on a log10 scale | linear |
| mean, median, skew, kurtosis | computed on x | computed on log10(x) |
| Normality tests | one, on x | **two** — one on log10(x), one on x |
| `neg_strategy` | only `drop` applies | all three apply |

The two-test behaviour under `log_transformation` is the point of it: you see
whether the log actually bought you normality, next to what you started with.

### What happens to values ≤ 0

A logarithm has nothing to say about them, so `neg_strategy` decides:

- `shift` — `log10(x - min + 1)`, keeps every row
- `signed` — `sign(x)·log10(1+|x|)`, symmetric around zero
- `drop` — discards them, zeros included

Under `log_scale` only `drop` is available, and it applies to the plot, the
statistics and the test alike, so all three describe exactly the same rows.

### Which test runs

`test="auto"` picks Shapiro-Wilk up to n = 5,000 and D'Agostino-Pearson K² above
it — the cap is about where Shapiro's p-value approximation stops being
trustworthy, not about speed. `test="ks"` gives Lilliefors instead.

Each sample resolves `auto` with **its own n**, so in a `log_transformation` run
the two tests may genuinely be different tests over different sample sizes. That
is deliberate rather than an oversight: the panel prints the name and n of each,
so nobody compares two p-values that were never comparable.

Two numbers on the panel are limits rather than measurements, and are printed as
such: a Lilliefors p-value at 0.001 is the floor of the statsmodels table, and an
exact p of 0 is underflow. When a test produces no p-value at all the verdict
reads *inconclusive* — it is not silently treated as a rejection.

### Built for large columns

The full dataset is 31k rows, so the panel subsamples what it *draws* while
testing everything: caps on Q-Q points, on the outliers drawn on the boxplot, and
on the rows the KDE is estimated from. It also warns when a column is mostly
ties, since the tests assume a continuous variable and a lattice of repeated
values breaks that.

### What comes back

`n` and how many rows were dropped (in total, and for being ≤ 0), the `mode` and
`transform` applied, `mean`, `median`, `std`, `skew`, `kurtosis`, the `test` name
with its `statistic`, `p_value`, `is_normal` and `n_test`, the same four for the
untransformed data when two tests ran, and `fig`.

---

## geo.py

The scraper takes coordinates from the site as published, and some of them are
wrong. These helpers catch the plainly broken ones.

`CUNDINAMARCA_BBOX` is a rectangle over the whole department, rounded outward.
It is deliberately generous: **it exists to catch impossible coordinates, not to
decide which municipality a listing belongs to.** Soacha, Chía and Zipaquirá all
count as inside, and so do slices of Boyacá, Tolima and Meta that the rectangle
happens to cover. A listing flagged by it is not in a neighbouring town — it is
somewhere the property cannot be.

| Name | What it does |
|---|---|
| `inside_bbox_mask` | Flags the points inside the box, edges included. |
| `outside_bbox_mask` | Flags the points outside it. |
| `count_outside_bogota` | Counts how many fall in, out, or are missing. |
| `BBox` | The box itself — pass your own to any of the three. |

**`outside` is not `~inside`.** `Series.between` returns `False` for a missing
value, so negating the inside mask would report every null coordinate as being
outside the box. A null is not evidence of anything, so it is excluded from both
masks and counted on its own.

`count_outside_bogota` returns `total`, `inside`, `outside`, `missing`,
`pct_outside` and the box used, with `inside + outside + missing == total`
always holding.

---

## map_graph.py

`create_map(df)` returns a [Folium](https://python-visualization.github.io/folium/)
map of the listings — OpenStreetMap tiles, centred on the mean coordinate, with
the points collected into a `MarkerCluster` so a dense city stays readable at low
zoom. Rows missing either coordinate are dropped first, and if nothing is left it
raises rather than handing back a blank map. The two column names are overridable
if your frame calls them something else.

Two things worth knowing before you call it on the full dataset:

- **It builds one marker per row.** On 31k listings that is slow to construct and
  heavy in the notebook. Filter or sample first — a locality, a price band, a
  property type — unless you really need every point.
- **Clean the coordinates first.** The map centres on the mean latitude and
  longitude, so a handful of broken points drag the whole view off Bogotá. Use
  `outside_bbox_mask` from [geo.py](geo.py) to drop them before mapping.
