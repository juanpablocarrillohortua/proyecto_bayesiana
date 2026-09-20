"""Credible intervals for a frozen ``scipy.stats`` distribution.

Given a posterior as a frozen distribution, :func:`credible_intervals`
reports its interval computed four ways -- equal tails, highest density,
Metropolis-Hastings chains and a percentile bootstrap -- and lays them
side by side in one table, so the sampling machinery can be checked
against the closed-form answer it is supposed to reproduce.

Code and docstrings are in English; the returned table and the optional
figure are in Spanish, because that is the language of whoever reads
them. Column labels are unaccented ``snake_case`` so they stay typeable;
the accents live in the values.

Importing this module has no side effects: no theme is applied, no
figure is created and the global matplotlib state is left untouched.
Styling is scoped to a ``plt.rc_context`` opened inside the plotting
helper. Nothing here needs PyMC, Stan or arviz -- the R-hat and ESS
diagnostics are computed from the chains directly.

Examples
--------
>>> from scipy.stats import beta
>>> from utils.cred_intervals import credible_intervals
>>> df = credible_intervals(beta(30, 20), methods=("eti", "hdi"))
>>> list(df["metodo"])
['Colas iguales', 'HDI (máxima densidad)']
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# --- STYLE PALETTE (same tokens as triple_plot.py) ---
C_DARK = "#2c3e50"
C_MID = "#34495e"
C_SOFT = "#7f8c8d"
C_GRID = "#eeeeee"
C_FILL = "#d6eaf8"

#: One colour per method, so a row in the table and its bar on the
#: figure are the same colour.
METHOD_COLORS = {
    "eti": "#2b5c8f",
    "hdi": "#2e8b57",
    "mcmc": "#e06d53",
    "bootstrap": "#b07aa1",
}

# --- STATISTICAL CONSTANTS ---
#: Methods computed by :func:`credible_intervals`, in table order.
METHODS = ("eti", "hdi", "mcmc", "bootstrap")
#: Spanish label drawn in the ``metodo`` column and on the figure.
METHOD_LABELS = {
    "eti": "Colas iguales",
    "hdi": "HDI (máxima densidad)",
    "mcmc": "Cadenas de Markov (MCMC)",
    "bootstrap": "Bootstrap percentil",
}
#: Methods that need a continuous density: ``hdi`` optimises over the
#: quantile function and ``mcmc`` proposes on the real line.
CONTINUOUS_ONLY = ("hdi", "mcmc")
#: Optimal scaling of a Gaussian random-walk proposal in one dimension
#: (Roberts, Gelman & Gilks 1997), applied to the target's own sd.
RW_OPTIMAL_SCALE = 2.38
#: Split-R-hat above this is the usual "the chains have not mixed"
#: threshold; below it the chains agree to within their own noise.
RHAT_MAX = 1.01
#: Effective sample size below this makes a 95% interval limit too noisy
#: to quote at three decimals.
ESS_MIN = 400.0
#: Lower tail mass kept away from exactly 0 while searching for the HDI:
#: ``ppf(0)`` is ``-inf`` on unbounded support and would make the width
#: infinite at the boundary of the search interval.
HDI_EPS = 1e-12
#: Bootstrap resampling is done in blocks of at most this many cells, so
#: peak memory stays bounded whatever ``n_boot * n_sample`` the caller
#: asks for.
BOOT_CHUNK_CELLS = 5_000_000
#: Tail mass left outside the density curve drawn by ``plot=True``.
PLOT_TAIL = 0.001

__all__ = [
    "ChainDiagnostics",
    "METHODS",
    "credible_intervals",
    "metropolis_hastings",
]


@dataclass
class ChainDiagnostics:
    """Convergence summary of a :func:`metropolis_hastings` run.

    Carried alongside the draws so a caller never has to recompute them
    to decide whether the chains can be believed.

    Attributes
    ----------
    n_chains : int
        Number of independent chains run.
    n_draws : int
        Draws kept per chain, after warmup.
    acceptance : float
        Share of proposals accepted, pooled over chains. A
        random-walk sampler in one dimension is tuned when this sits
        roughly between 0.2 and 0.5.
    rhat : float
        Split Gelman-Rubin statistic. Defined even for a single chain,
        which is split in half.
    ess : float
        Bulk effective sample size, pooled over chains.

    Examples
    --------
    >>> ChainDiagnostics(4, 100, 0.3, 1.0, 250.0).converged
    False
    """

    n_chains: int
    n_draws: int
    acceptance: float
    rhat: float
    ess: float

    @property
    def converged(self) -> bool:
        """Whether R-hat and ESS both clear their usual thresholds."""
        return bool(self.rhat <= RHAT_MAX and self.ess >= ESS_MIN)

    def label(self) -> str:
        """Render the diagnostics as one Spanish line for the table."""
        return (
            f"R̂ = {self.rhat:.4f} · ESS = {self.ess:,.0f} · "
            f"aceptación = {self.acceptance:.2f}"
        )


# ----------------------------------------------------------------------
# Input guards
# ----------------------------------------------------------------------
def _check_frozen(dist: Any, name: str) -> None:
    """Reject anything that is not a univariate frozen distribution."""
    missing = [
        attr for attr in ("ppf", "cdf", "rvs") if not hasattr(dist, attr)
    ]
    if missing:
        raise ValueError(
            f"{name} does not look like a frozen scipy.stats "
            f"distribution: it has no {', '.join(missing)}. Pass a "
            f"frozen univariate distribution such as beta(4.25, 2.6). A "
            f"multivariate one like dirichlet(alpha) has no ppf; use "
            f"its Beta marginals, Beta(alpha_i, alpha_0 - alpha_i), one "
            f"per category."
        )


def _is_discrete(dist: Any) -> bool:
    """Whether the frozen distribution has a pmf rather than a pdf."""
    from scipy import stats as sps

    return isinstance(getattr(dist, "dist", None), sps.rv_discrete)


def _check_methods(methods: Sequence[str]) -> tuple[str, ...]:
    """Validate the requested methods and put them in table order."""
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(
            f"methods={tuple(methods)!r} contains {unknown[0]!r}, which "
            f"is not recognised. Use any of {METHODS}."
        )
    if not methods:
        raise ValueError(f"methods is empty; pass at least one of {METHODS}.")
    return tuple(m for m in METHODS if m in set(methods))


def _check_support(dist: Any, name: str, methods: Sequence[str]) -> None:
    """Refuse the density-based methods on a discrete distribution."""
    blocked = [m for m in methods if m in CONTINUOUS_ONLY]
    if blocked and _is_discrete(dist):
        raise ValueError(
            f"{name} is a discrete distribution, so "
            f"{', '.join(blocked)} do not apply: the HDI search "
            f"optimises over a continuous quantile function and the "
            f"sampler proposes on the real line. Pass "
            f"methods=('eti', 'bootstrap'), both of which are valid "
            f"here."
        )


# ----------------------------------------------------------------------
# Analytic intervals
# ----------------------------------------------------------------------
def _eti(dist: Any, level: float) -> tuple[float, float]:
    """Equal-tailed interval: the quantile function, evaluated twice."""
    tail = (1.0 - level) / 2.0
    return float(dist.ppf(tail)), float(dist.ppf(1.0 - tail))


def _hdi(dist: Any, level: float) -> tuple[float, float]:
    """Narrowest interval holding ``level`` of the mass.

    Parameterised by the mass left in the lower tail: every candidate
    ``[ppf(p), ppf(p + level)]`` already has the right coverage by
    construction, so only its width has to be minimised, over the single
    bounded variable ``p``.

    On a monotone density (``expon``, ``Beta(1, 5)``) the optimum sits on
    a boundary and the interval comes back one-sided. That is the
    correct answer for such a density, not a failure of the search.
    """
    from scipy.optimize import minimize_scalar

    upper = 1.0 - level

    def width(p: float) -> float:
        return float(dist.ppf(p + level) - dist.ppf(p))

    if upper <= HDI_EPS:  # level so close to 1 there is nothing to search
        return _eti(dist, level)

    res = minimize_scalar(
        width,
        bounds=(HDI_EPS, upper - HDI_EPS),
        method="bounded",
        options={"xatol": 1e-10},
    )
    # The boundaries are not explored by `bounded`, and they are exactly
    # where a monotone density puts its optimum.
    candidates = [HDI_EPS, upper - HDI_EPS]
    if res.success:
        candidates.append(float(res.x))
    p_best = min(candidates, key=width)
    return float(dist.ppf(p_best)), float(dist.ppf(p_best + level))


# ----------------------------------------------------------------------
# MCMC: sampler and diagnostics
# ----------------------------------------------------------------------
def _split_rhat(chains: np.ndarray) -> float:
    """Split Gelman-Rubin statistic for a 2-D ``(chain, draw)`` array.

    Each chain is halved first, so a single chain that drifts is still
    caught: the two halves then play the role of two chains.
    """
    n_chains, n_draws = chains.shape
    half = n_draws // 2
    if half < 2:
        return float("nan")
    split = np.concatenate(
        [chains[:, :half], chains[:, n_draws - half :]], axis=0
    )
    m, n = split.shape
    means = split.mean(axis=1)
    variances = split.var(axis=1, ddof=1)
    within = float(variances.mean())
    between = float(n * means.var(ddof=1))
    if within <= 0:
        return float("nan")
    var_hat = ((n - 1) / n) * within + between / n
    return float(np.sqrt(var_hat / within)) if m > 1 else float("nan")


def _autocorr(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of one chain, via FFT."""
    x = x - x.mean()
    n = x.size
    size = int(2 ** np.ceil(np.log2(2 * n)))
    freq = np.fft.rfft(x, n=size)
    acov = np.fft.irfft(freq * np.conjugate(freq), n=size)[:n].real
    acov /= n
    if acov[0] <= 0:
        return np.zeros(n)
    return acov / acov[0]


def _ess(chains: np.ndarray) -> float:
    """Bulk effective sample size with Geyer's positive-sequence rule.

    Consecutive autocorrelation pairs are summed until a pair turns
    negative, which is where the estimate stops being trustworthy; the
    truncation is what keeps the sum from accumulating pure noise out in
    the tail of the correlogram.
    """
    n_chains, n_draws = chains.shape
    if n_draws < 4:
        return float("nan")
    rho = np.mean([_autocorr(c) for c in chains], axis=0)

    n_pairs = (n_draws - 1) // 2
    pairs = rho[1 : 2 * n_pairs + 1].reshape(n_pairs, 2).sum(axis=1)
    negative = np.flatnonzero(pairs <= 0)
    keep = int(negative[0]) if negative.size else n_pairs
    tau = 1.0 + 2.0 * float(pairs[:keep].sum())
    if tau <= 0:
        return float(n_chains * n_draws)
    return float(n_chains * n_draws / tau)


def _quantile_mcse(dist: Any, q: float, p: float, n_eff: float) -> float:
    """Monte-Carlo standard error of a sampled quantile.

    Not the standard error of the mean: a quantile estimated from
    ``n_eff`` draws has sd ``sqrt(p(1-p)/n_eff) / f(q)``, so the same
    number of draws pins down a limit in a dense part of the
    distribution far better than one out in a flat tail. ``f`` is taken
    from the target itself, which is known here in closed form.

    Returns ``nan`` where the density vanishes, since the quantile is
    then not locally identified and no finite error applies.
    """
    density = float(dist.pdf(q))
    if not np.isfinite(density) or density <= 0 or n_eff <= 0:
        return float("nan")
    return float(np.sqrt(p * (1.0 - p) / n_eff) / density)


def metropolis_hastings(
    dist: Any,
    *,
    n_chains: int = 4,
    n_draws: int = 10_000,
    n_warmup: int = 1_000,
    scale: float | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, ChainDiagnostics]:
    """Sample a frozen distribution with a random-walk Metropolis chain.

    The target already has a ``ppf``, so this is not how you would draw
    from it in anger -- it is here so the interval can be recomputed by
    a sampler and compared against the exact one.

    Proposals landing outside the support get ``logpdf = -inf`` and are
    rejected on the spot, which is what lets an unconstrained Gaussian
    proposal sample a Beta target correctly with no reflection or
    reparameterisation.

    Parameters
    ----------
    dist : frozen scipy.stats distribution
        The target. Must be continuous and univariate.
    n_chains : int, default 4
        Independent chains, started at dispersed quantiles of the
        target so R-hat has something to detect.
    n_draws : int, default 10000
        Draws kept per chain, after warmup.
    n_warmup : int, default 1000
        Draws discarded at the start of each chain.
    scale : float, optional
        Standard deviation of the Gaussian proposal. Defaults to
        ``2.38 * dist.std()``, the optimal one-dimensional scaling.
    random_state : int, default 42
        Seed. Every draw in this module comes from it.

    Returns
    -------
    tuple
        ``(chains, diagnostics)`` where ``chains`` is a
        ``(n_chains, n_draws)`` array of post-warmup draws and
        ``diagnostics`` is a :class:`ChainDiagnostics`.

    Raises
    ------
    ValueError
        If ``dist`` is not a frozen distribution, if it is discrete, or
        if ``n_chains``/``n_draws`` are below 1.

    Examples
    --------
    >>> from scipy.stats import norm
    >>> chains, diag = metropolis_hastings(
    ...     norm(), n_chains=2, n_draws=500, n_warmup=100
    ... )
    >>> chains.shape
    (2, 500)
    """
    _check_frozen(dist, "dist")
    _check_support(dist, "dist", ("mcmc",))
    if n_chains < 1 or n_draws < 1 or n_warmup < 0:
        raise ValueError(
            f"n_chains={n_chains!r}, n_draws={n_draws!r} and "
            f"n_warmup={n_warmup!r} must be positive (warmup may be 0)."
        )

    sd = float(dist.std())
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(
            "The target has no finite, positive standard deviation, so "
            "the proposal cannot be scaled from it. Pass scale= "
            "explicitly."
        )
    step = float(scale) if scale is not None else RW_OPTIMAL_SCALE * sd
    if step <= 0:
        raise ValueError(f"scale={scale!r} must be positive.")

    rng = np.random.default_rng(random_state)
    total = n_warmup + n_draws

    # Dispersed starts: chains that begin together cannot disagree, and
    # R-hat only means something when they had the chance to.
    quantiles = np.linspace(0.1, 0.9, n_chains)
    starts = np.atleast_1d(dist.ppf(quantiles)).astype(float)

    current = starts.copy()
    log_current = np.asarray(dist.logpdf(current), dtype=float)
    draws = np.empty((n_chains, total), dtype=float)
    n_accepted = 0

    for i in range(total):
        proposal = current + step * rng.standard_normal(n_chains)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_proposal = np.asarray(dist.logpdf(proposal), dtype=float)
        log_proposal = np.nan_to_num(log_proposal, nan=-np.inf, neginf=-np.inf)
        log_u = np.log(rng.random(n_chains))
        accept = log_u < (log_proposal - log_current)
        current = np.where(accept, proposal, current)
        log_current = np.where(accept, log_proposal, log_current)
        draws[:, i] = current
        n_accepted += int(accept.sum())

    kept = draws[:, n_warmup:]
    diagnostics = ChainDiagnostics(
        n_chains=int(n_chains),
        n_draws=int(n_draws),
        acceptance=float(n_accepted / (n_chains * total)),
        rhat=_split_rhat(kept),
        ess=_ess(kept),
    )
    return kept, diagnostics


# ----------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------
def _bootstrap_limits(
    dist: Any,
    level: float,
    n_sample: int,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Percentile bootstrap of the two interval limits.

    What is resampled is the pair of sample *quantiles*, not the mean:
    the quantiles are the quantity the other three rows report, so
    bootstrapping them is what makes this row comparable to them rather
    than an answer to a different question.

    Returns
    -------
    tuple
        ``(low, high, se_low, se_high)`` -- the limits averaged over the
        resamples, and their spread across resamples, which is the
        Monte-Carlo standard error of each.
    """
    tail = (1.0 - level) / 2.0
    sample = np.asarray(dist.rvs(n_sample, random_state=rng), dtype=float)

    # Chunked so peak memory does not follow n_boot * n_sample.
    per_chunk = max(1, BOOT_CHUNK_CELLS // max(n_sample, 1))
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    done = 0
    while done < n_boot:
        size = min(per_chunk, n_boot - done)
        idx = rng.integers(0, n_sample, size=(size, n_sample))
        limits = np.quantile(sample[idx], [tail, 1.0 - tail], axis=1)
        lows.append(limits[0])
        highs.append(limits[1])
        done += size

    low = np.concatenate(lows)
    high = np.concatenate(highs)
    return (
        float(low.mean()),
        float(high.mean()),
        float(low.std(ddof=1)) if low.size > 1 else 0.0,
        float(high.std(ddof=1)) if high.size > 1 else 0.0,
    )


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def _fmt_param(value: float) -> str:
    """Render a shape parameter readably.

    Plain ``.4g`` turns a conjugate posterior's shape into
    ``1.93e+04``, which hides exactly the count the reader came for.
    The thousands separator keeps it legible at any magnitude.
    """
    return f"{float(value):,.6g}"


def _dist_label(dist: Any) -> str:
    """Name a frozen distribution the way its constructor reads."""
    name = getattr(getattr(dist, "dist", None), "name", "distribución")
    args = [_fmt_param(a) for a in getattr(dist, "args", ())]
    args += [
        f"{k}={_fmt_param(v)}" for k, v in getattr(dist, "kwds", {}).items()
    ]
    return f"{name}({', '.join(args)})" if args else str(name)


def _panel(
    ax: Any,
    ax_bars: Any,
    dist: Any,
    rows: list[dict],
    name: str | None,
    first: bool = True,
) -> None:
    """Draw one density plus its stacked interval bars.

    ``first`` marks the leftmost column of a multi-panel figure, which
    is the only one to carry the method names and the y axis title:
    repeating them per panel crowds the neighbouring axis and says
    nothing new.
    """
    import seaborn as sns

    lo = float(dist.ppf(PLOT_TAIL))
    hi = float(dist.ppf(1.0 - PLOT_TAIL))
    grid = np.linspace(lo, hi, 600)
    density = np.asarray(dist.pdf(grid), dtype=float)

    ax.grid(axis="y", color=C_GRID, linestyle="-", linewidth=1, zorder=0)
    ax.fill_between(grid, density, color=C_FILL, zorder=2)
    ax.plot(grid, density, color=C_DARK, linewidth=1.6, zorder=3)

    mean = float(dist.mean())
    ax.axvline(mean, color=C_SOFT, linestyle=":", linewidth=1.4, zorder=4)
    # Only a mapping names its panels; a lone distribution is already
    # named by the subtitle, and a second copy would collide with it.
    if name is not None:
        ax.set_title(
            name,
            loc="left",
            fontsize=13,
            fontweight="bold",
            pad=10,
            color=C_MID,
        )
    if first:
        ax.set_ylabel("Densidad", fontsize=10.5)
    # labelbottom, not set_xticklabels: the axes are shared down the
    # column, so blanking the labels would blank them on the bars too.
    ax.tick_params(labelbottom=False)

    for offset, row in enumerate(rows):
        y = -offset
        color = METHOD_COLORS[row["key"]]
        ax_bars.plot(
            [row["low"], row["high"]],
            [y, y],
            color=color,
            linewidth=5.0,
            solid_capstyle="butt",
            zorder=3,
        )
        ax_bars.plot([mean], [y], marker="o", color=C_DARK, ms=4, zorder=4)
        ax_bars.text(
            row["low"],
            y + 0.28,
            f"{row['low']:,.4g}",
            fontsize=8.5,
            ha="right",
            va="bottom",
            color=C_SOFT,
        )
        ax_bars.text(
            row["high"],
            y + 0.28,
            f"{row['high']:,.4g}",
            fontsize=8.5,
            ha="left",
            va="bottom",
            color=C_SOFT,
        )

    ax_bars.set_yticks([-i for i in range(len(rows))])
    ax_bars.set_yticklabels(
        [METHOD_LABELS[r["key"]] for r in rows] if first else [],
        fontsize=9.5,
    )
    ax_bars.set_ylim(-len(rows) + 0.4, 0.8)
    ax_bars.set_xlim(lo, hi)
    ax.set_xlim(lo, hi)
    sns.despine(ax=ax, left=True, bottom=True)
    sns.despine(ax=ax_bars, left=True, bottom=True)
    ax.tick_params(colors=C_SOFT, labelsize=9.5)
    ax_bars.tick_params(colors=C_SOFT, labelsize=9.5)


def _build_figure(
    panels: list[tuple[str | None, Any, list[dict]]],
    level: float,
    figsize: tuple[float, float],
    show: bool,
) -> Any:
    """Assemble the density + interval figure and close it."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    with plt.rc_context():
        sns.set_theme(style="white", context="notebook")
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["text.color"] = "#333333"

        n_panels = len(panels)
        width = figsize[0] * (1 if n_panels == 1 else min(n_panels, 3))
        fig, axes = plt.subplots(
            2,
            n_panels,
            figsize=(width, figsize[1]),
            squeeze=False,
            gridspec_kw={"height_ratios": [2.0, 1.4]},
            sharex="col",
        )

        fig.text(
            0.02,
            0.965,
            f"Intervalos de credibilidad al {level:.0%}",
            fontsize=18,
            fontweight="bold",
            color=C_DARK,
        )
        fig.text(
            0.02,
            0.912,
            " · ".join(_dist_label(d) for _, d, _ in panels),
            fontsize=11.5,
            color=C_SOFT,
        )

        for col, (name, dist, rows) in enumerate(panels):
            _panel(
                axes[0][col], axes[1][col], dist, rows, name, first=col == 0
            )

        plt.subplots_adjust(
            top=0.84, left=0.10, right=0.97, bottom=0.10, hspace=0.12
        )
        if show:
            plt.show()

    # Always closed: a notebook loop over marginals would otherwise pile
    # figures up. `fig` still accepts savefig().
    plt.close(fig)
    return fig


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def credible_intervals(
    dist: Any | Mapping[str, Any],
    level: float = 0.95,
    *,
    methods: Sequence[str] = METHODS,
    n_chains: int = 4,
    n_draws: int = 10_000,
    n_warmup: int = 1_000,
    proposal_scale: float | None = None,
    n_sample: int = 2_000,
    n_boot: int = 1_000,
    random_state: int = 42,
    plot: bool = False,
    figsize: tuple[float, float] = (11.0, 6.0),
    show: bool = True,
) -> pd.DataFrame:
    """Compare the credible interval of a posterior, computed four ways.

    The equal-tailed row is exact and serves as the reference: the
    ``error_abs`` column measures every other row against it, so a
    sampler that has not converged shows up as a number rather than as a
    plot nobody checks.

    Parameters
    ----------
    dist : frozen scipy.stats distribution or mapping of them
        The posterior. Pass a mapping -- ``{"Prior": d1, "Posterior":
        d2}``, or one entry per Dirichlet marginal -- to get every
        distribution in one table, under a ``distribucion`` column.
    level : float, default 0.95
        Credible mass, as a fraction in ``(0, 1)``.
    methods : sequence of str, default :data:`METHODS`
        Any subset of ``('eti', 'hdi', 'mcmc', 'bootstrap')``. Order is
        ignored; rows always come out in :data:`METHODS` order.
    n_chains : int, default 4
        Chains for the ``mcmc`` row.
    n_draws : int, default 10000
        Post-warmup draws per chain.
    n_warmup : int, default 1000
        Draws discarded per chain.
    proposal_scale : float, optional
        Proposal sd for the sampler; defaults to ``2.38 * dist.std()``.
    n_sample : int, default 2000
        Size of the reference sample the ``bootstrap`` row resamples.
    n_boot : int, default 1000
        Number of bootstrap resamples.
    random_state : int, default 42
        Seed for the sampler and the bootstrap alike.
    plot : bool, default False
        Draw the density with the intervals stacked underneath. The
        figure is returned in ``df.attrs["fig"]``, not as a second
        return value, so the call still ends a notebook cell with a
        table.
    figsize : tuple of float, default (11.0, 6.0)
        Size of a single panel, in inches. Widened for a mapping.
    show : bool, default True
        Call ``plt.show()`` on the figure. Ignored when ``plot`` is
        ``False``.

    Returns
    -------
    pandas.DataFrame
        Spanish columns ``metodo``, ``nivel``, ``limite_inferior``,
        ``limite_superior``, ``amplitud``, ``ee_inferior``,
        ``ee_superior``, ``error_abs``, ``n_muestras`` and
        ``diagnostico``, preceded by ``distribucion`` when a mapping was
        passed. One row per method per distribution.

    Raises
    ------
    ValueError
        If ``level`` is outside ``(0, 1)``, if ``dist`` is not a frozen
        univariate distribution, if ``methods`` names something
        unrecognised, or if ``hdi``/``mcmc`` are asked of a discrete
        distribution.

    Warns
    -----
    UserWarning
        When the chains miss the R-hat or ESS thresholds, so the
        ``mcmc`` row is reported but not to be trusted at face value.

    Examples
    --------
    >>> from scipy.stats import beta
    >>> df = credible_intervals(beta(30, 20), methods=("eti",))
    >>> round(float(df.loc[0, "limite_inferior"]), 4)
    0.4624
    """
    if not 0.0 < level < 1.0:
        raise ValueError(
            f"level={level!r} must be a fraction strictly between 0 and "
            f"1, such as 0.95. It is not a percentage."
        )
    wanted = _check_methods(methods)

    if isinstance(dist, Mapping):
        if not dist:
            raise ValueError(
                "dist is an empty mapping; pass at least one frozen "
                "distribution."
            )
        targets = [(str(k), v) for k, v in dist.items()]
        named = True
    else:
        targets = [("distribución", dist)]
        named = False

    for name, frozen in targets:
        _check_frozen(frozen, f"dist[{name!r}]" if named else "dist")
        _check_support(frozen, f"dist[{name!r}]" if named else "dist", wanted)

    rng = np.random.default_rng(random_state)
    records: list[dict[str, Any]] = []
    panels: list[tuple[str | None, Any, list[dict]]] = []

    for name, frozen in targets:
        reference = _eti(frozen, level)
        rows: list[dict[str, Any]] = []

        for key in wanted:
            if key == "eti":
                low, high = reference
                se_low = se_high = 0.0
                n_used = 0
                note = "exacto (ppf)"
            elif key == "hdi":
                low, high = _hdi(frozen, level)
                se_low = se_high = 0.0
                n_used = 0
                note = "exacto (optimización numérica)"
            elif key == "mcmc":
                chains, diag = metropolis_hastings(
                    frozen,
                    n_chains=n_chains,
                    n_draws=n_draws,
                    n_warmup=n_warmup,
                    scale=proposal_scale,
                    random_state=int(rng.integers(0, 2**32 - 1)),
                )
                tail = (1.0 - level) / 2.0
                pooled = chains.ravel()
                low, high = (
                    float(np.quantile(pooled, tail)),
                    float(np.quantile(pooled, 1.0 - tail)),
                )
                # Against the effective sample size, not the nominal
                # one: consecutive draws of a random walk are
                # correlated, so 40,000 of them are not 40,000
                # independent draws and must not be quoted as such.
                ess = diag.ess if np.isfinite(diag.ess) else float(pooled.size)
                se_low = _quantile_mcse(frozen, low, tail, ess)
                se_high = _quantile_mcse(frozen, high, 1.0 - tail, ess)
                n_used = int(pooled.size)
                note = diag.label()
                if not diag.converged:
                    warnings.warn(
                        f"MCMC for {name!r} did not clear the usual "
                        f"thresholds (rhat={diag.rhat:.4f} > {RHAT_MAX}, "
                        f"or ess={diag.ess:,.0f} < {ESS_MIN:,.0f}). Raise "
                        f"n_draws, raise n_warmup, or set "
                        f"proposal_scale; acceptance was "
                        f"{diag.acceptance:.2f}.",
                        UserWarning,
                        stacklevel=2,
                    )
            else:  # bootstrap
                low, high, se_low, se_high = _bootstrap_limits(
                    frozen, level, n_sample, n_boot, rng
                )
                n_used = int(n_boot)
                note = f"B = {n_boot:,} · n = {n_sample:,}"

            rows.append(
                {
                    "key": key,
                    "low": low,
                    "high": high,
                    "se_low": se_low,
                    "se_high": se_high,
                    "n": n_used,
                    "note": note,
                }
            )

        for row in rows:
            record: dict[str, Any] = {}
            if named:
                record["distribucion"] = name
            record.update(
                {
                    "metodo": METHOD_LABELS[row["key"]],
                    "nivel": float(level),
                    "limite_inferior": float(row["low"]),
                    "limite_superior": float(row["high"]),
                    "amplitud": float(row["high"] - row["low"]),
                    "ee_inferior": float(row["se_low"]),
                    "ee_superior": float(row["se_high"]),
                    "error_abs": max(
                        abs(row["low"] - reference[0]),
                        abs(row["high"] - reference[1]),
                    ),
                    "n_muestras": int(row["n"]),
                    "diagnostico": row["note"],
                }
            )
            records.append(record)

        panels.append((name if named else None, frozen, rows))

    table = pd.DataFrame.from_records(records)

    if plot:
        table.attrs["fig"] = _build_figure(panels, level, figsize, show)

    return table


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _smoke() -> bool:
    """Exercise every entry point; return True when something failed."""
    from scipy.stats import beta, binom, gamma, norm

    failed = False

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal failed
        if condition:
            print(f"ok    {label}")
        else:
            failed = True
            print(f"FAIL  {label} {detail}")

    def raises(label: str, fn) -> None:
        try:
            fn()
        except ValueError:
            print(f"ok    {label}")
        else:
            nonlocal failed
            failed = True
            print(f"FAIL  {label} (no ValueError)")

    # The notebook's own posterior: Beta(4.25 + k, 2.6 + n - k).
    posterior = beta(4.25 + 19_300, 2.6 + 11_952)
    df = credible_intervals(posterior, n_draws=4_000, n_warmup=500, n_boot=400)
    check("four rows on the notebook posterior", len(df) == 4, str(len(df)))
    worst = float(df["error_abs"].max())
    check(
        "every row within 5e-3 of the exact ETI", worst < 5e-3, f"{worst:.2e}"
    )

    # A skewed target: the HDI must be strictly narrower than the ETI.
    skewed = credible_intervals(gamma(2.0), methods=("eti", "hdi"))
    w_eti = float(skewed.loc[0, "amplitud"])
    w_hdi = float(skewed.loc[1, "amplitud"])
    check(
        "HDI narrower than ETI on gamma(2)",
        w_hdi < w_eti,
        f"{w_hdi:.4f} vs {w_eti:.4f}",
    )

    # A symmetric target: the two must agree.
    sym = credible_intervals(norm(), methods=("eti", "hdi"))
    gap = abs(float(sym.loc[0, "amplitud"]) - float(sym.loc[1, "amplitud"]))
    check("HDI == ETI on norm()", gap < 1e-4, f"{gap:.2e}")

    # Chain diagnostics on a well-behaved target.
    _, diag = metropolis_hastings(
        beta(30, 20), n_chains=4, n_draws=6_000, n_warmup=1_000
    )
    check("rhat below threshold", diag.rhat <= RHAT_MAX, f"{diag.rhat:.4f}")
    check("ess above threshold", diag.ess >= ESS_MIN, f"{diag.ess:,.0f}")
    check(
        "acceptance in the 0.2-0.5 band",
        0.15 < diag.acceptance < 0.6,
        f"{diag.acceptance:.2f}",
    )

    # Mapping input, three Dirichlet-style marginals.
    marginals = credible_intervals(
        {
            "Economico": beta(860 + 4_100, 4_140 + 27_152),
            "Normal": beta(3_450 + 21_000, 1_550 + 10_252),
            "Lujo": beta(690 + 6_152, 4_310 + 25_100),
        },
        methods=("eti", "hdi", "bootstrap"),
        n_boot=200,
        n_sample=800,
    )
    check("3 x 3 rows for a mapping", len(marginals) == 9, str(len(marginals)))
    check(
        "distribucion column present",
        "distribucion" in marginals.columns,
        str(list(marginals.columns)),
    )

    # Figure.
    drawn = credible_intervals(
        beta(30, 20),
        methods=("eti", "hdi", "bootstrap"),
        n_boot=200,
        n_sample=800,
        plot=True,
        show=False,
    )
    check("figure stashed in attrs", drawn.attrs.get("fig") is not None)

    # Guards.
    raises("level=1.5 rejected", lambda: credible_intervals(beta(2, 2), 1.5))
    raises("bare float rejected", lambda: credible_intervals(0.5))
    raises(
        "unknown method rejected",
        lambda: credible_intervals(beta(2, 2), methods=("xyz",)),
    )
    raises(
        "hdi on a discrete target rejected",
        lambda: credible_intervals(binom(10, 0.5), methods=("hdi",)),
    )
    ok_discrete = credible_intervals(
        binom(10, 0.5), methods=("eti", "bootstrap"), n_boot=200, n_sample=500
    )
    check("discrete works with eti+bootstrap", len(ok_discrete) == 2)

    return failed


if __name__ == "__main__":
    import sys

    import matplotlib as mpl

    mpl.use("Agg")
    sys.exit(1 if _smoke() else 0)
