# utils/

Shared helpers for the exploratory analysis. Five modules, each doing one job:

| Module | What it gives you |
|---|---|
| [plotting.py](plotting.py) | `EDAPlotter` — the whole chart set behind one class |
| [triple_plot.py](triple_plot.py) | `normality_report` — a three-panel distribution diagnostic |
| [cred_intervals.py](cred_intervals.py) | `credible_intervals` — a posterior's intervals, exact and simulated, in one table |
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
from utils.cred_intervals import credible_intervals
from utils.geo import count_outside_bogota, outside_bbox_mask
from utils.map_graph import create_map
```

Code and docstrings are in English; the text drawn on the figures is Spanish,
because that is the language of whoever reads the chart. The same rule covers
the one module that returns a table meant to be read rather than indexed —
[cred_intervals.py](cred_intervals.py) labels its columns in Spanish, unaccented
so they stay typeable, and puts the accents in the values.

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

## cred_intervals.py

`credible_intervals(dist, nivel=0.95, ...)` takes a posterior as a frozen
`scipy.stats` distribution and returns the 2×2 grid of intervals the course
script prints, as one Spanish table.

```python
from scipy.stats import gamma

posterior = gamma(alpha_post, scale=1 / beta_post)
credible_intervals(posterior, nivel=0.95, plot=True)
```

```
fuente          tipo  nivel  limite_inferior  limite_superior    ancho
Exacto Colas Iguales   0.95         0.242209         5.571643 5.329434
Exacto          HPDI   0.95         0.042363         4.765168 4.722805
  MCMC Colas Iguales   0.95         0.240902         5.602988 5.362086
  MCMC          HPDI   0.95         0.045158         4.772519 4.727361
```

Two axes, four rows. `fuente` is where the numbers come from and `tipo` is which
interval:

| `fuente` | Where it reads |
|---|---|
| Exacto | The distribution's own `ppf` and `cdf`. Deterministic. |
| MCMC | A sample of `n_sim` draws simulated from it. |

| `tipo` | Which interval |
|---|---|
| Colas Iguales | Equal mass left outside on each side: `ppf(α/2)`, `ppf(1−α/2)` exactly, percentiles on the draws. |
| HPDI | The narrowest interval of that credibility: a minimisation of the width exactly, a sorted-window scan on the draws. |

### Each method is its own function

The four cells are public and self-contained: give one the distribution and the
level and it computes that interval on its own, returning a plain
`(inferior, superior)` tuple. `credible_intervals` does nothing but call the
four and assemble the frame.

| Function | What it does |
|---|---|
| `colas_iguales_exacto(dist, nivel)` | `ppf` on each tail. |
| `hpdi_exacto(dist, nivel)` | Minimises the width with `scipy.optimize.minimize`. |
| `colas_iguales_mcmc(dist, nivel, n_sim, random_state)` | Percentiles of a simulated sample. |
| `hpdi_mcmc(dist, nivel, n_sim, random_state)` | Sliding window over the sorted sample. |

```python
inf, sup = hpdi_exacto(posterior, 0.95)
```

The two `mcmc` functions each draw their own sample. They share a default
`random_state`, so called with the defaults they draw the *same* sample and
their two rows stay mutually consistent — but neither depends on the other.

### It is the course script, generalised

Every piece corresponds to one statement of the reference implementation, with
Gamma swapped for whatever you pass:

| Script | Module |
|---|---|
| `stats.gamma.ppf(0.025, a=α, scale=1/β)` | `colas_iguales_exacto` |
| `interval_width` + `minimize(..., bounds=[(0, ppf(0.05))])` | `hpdi_exacto`, same call, `ancho` nested inside |
| `np.random.gamma(shape=α, scale=1/β, size=100000)` | `dist.rvs(n_sim, random_state=rng)` |
| `np.percentile(samples, 2.5)` | `colas_iguales_mcmc` |
| `calc_hpdi_mcmc(samples, 0.95)` | `hpdi_mcmc`, scan inlined |

Defaults are the script's own numbers: `n_sim=100_000`, `random_state=42`. On
the script's own Gamma the two `Exacto` rows come back **bit-identical** to it.

The optimiser's bounds are the script's, `(ppf(0), ppf(1 − nivel))`. `ppf(0)` is
`0.0` for Gamma and Beta — exactly the literal `0` the script writes — and
`-inf` for a distribution on the real line, which L-BFGS-B reads as "unbounded
below", so it generalises for free. The upper limit is still recomputed as
`ppf(cdf(lb) + nivel)` after `minimize` returns, rather than read off the
optimiser, exactly as the script does.

**There is no input validation and no self-test.** Passing something that is not
a frozen distribution raises whatever scipy raises, and nothing in the repo
checks the module against the script any more — that is the deliberate cost of
keeping the file to the basic code.

### Colas Iguales and HPDI are not the same interval

On a symmetric posterior they agree to the last decimal. On a skewed one they do
not, and `ancho` is where you see it: for `gamma(2)` the equal-tailed interval
spans 5.33 and the HPDI 4.72, because the HPDI is free to slide toward the mode
instead of leaving 2.5% in each tail by construction.

Which one you want is a modelling decision, not a default. Equal tails are
invariant to a monotone reparameterisation and the HPDI is not; the HPDI is the
shortest interval of that credibility and the equal-tailed one is not. The table
gives you both rather than choosing.

On a monotone density the HPDI comes back one-sided. That is the right answer
for such a density, not a failure of the search.

### Why simulate what `ppf` already knows

For a conjugate posterior both intervals are available in closed form, so the
`MCMC` rows are not doing inference — **they show that the simulation lands where
the closed form already is.** Read the table down each `tipo`: the two rows
should agree, and the size of the gap is how much the sample size is costing you.

Worth knowing when you read that gap: **the two estimators are not equally
noisy.** At `n_sim=100_000`, over 40 seeds, the simulated equal-tailed limit has
a standard deviation of about 0.7% of the posterior's own sd, while the HPDI
limit has about 2.0% — roughly threefold, because the sorted-window scan picks a
minimum over many near-equal widths and a different draw moves which window wins.
An `MCMC HPDI` row that sits further from its `Exacto` twin than the equal-tailed
row does is the expected behaviour, not a symptom.

### Several posteriors

One call takes one distribution. For the three Dirichlet marginals, call it
three times and concatenate:

```python
pd.concat(
    [
        credible_intervals(beta(a, a0_post - a)).assign(categoria=nombre)
        for nombre, a in zip(categorias, alpha_post)
    ],
    ignore_index=True,
)
```

A multivariate `dirichlet(alpha)` has no `ppf`, so it cannot be passed directly;
its Beta marginals, `Beta(αᵢ, α₀ − αᵢ)`, are what to build from it — which is
what the notebook already does to plot them.

### The figure

`plot=True` draws the posterior with the four intervals stacked underneath it,
one bar per row with its limits annotated. It comes back in
**`df.attrs["fig"]`**, not as a second return value, so the call still ends a
notebook cell with the table and `df.attrs["fig"].savefig(...)` still works.

### What comes back

A DataFrame with `fuente`, `tipo`, `nivel`, `limite_inferior`, `limite_superior`
and `ancho` — the script's own columns, nothing more. Four rows, always. The
four functions called on their own return a plain `(inferior, superior)` tuple
instead.

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
