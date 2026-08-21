"""
Normality diagnostic panel (histogram + boxplot + Q-Q plot + test).

Flat editorial styling, tuned for large datasets. Code, docstrings and
diagnostics are in English; only the text drawn on the figure is in
Spanish, because that is what the reader of the panel sees.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors

# --- STYLE PALETTE ---
C_DARK = "#2c3e50"
C_MID = "#34495e"
C_SOFT = "#7f8c8d"
C_BLUE = "#5dade2"
C_GREY = "#95a5a6"
C_ORANGE = "#e67e22"
C_GRID = "#eeeeee"
C_BOX = "#f8f9f9"

# --- STATISTICAL CONSTANTS ---
#: Tests accepted by :func:`normality_report`.
TESTS = ("auto", "shapiro", "dagostino", "ks")
#: Strategies accepted for x <= 0 under a log transformation.
NEG_STRATEGIES = ("shift", "signed", "drop")
#: Analysis modes; see :func:`normality_report`.
MODES = ("identity", "log_scale", "log_transformation")
#: Above this n the Shapiro-Wilk p-value approximation stops being
#: reliable (scipy raises its own warning).
SHAPIRO_MAX_N = 5_000
#: statsmodels floors the Lilliefors p-value at this value: the table
#: does not go any lower, so it is not a measurement.
LILLIEFORS_P_FLOOR = 1e-3
#: Unique-value ratio below which the sample is treated as a lattice
#: (massive ties) for the tests that assume continuity.
TIES_UNIQUE_RATIO = 0.8
#: Minimum number of observations for the panel to mean anything.
MIN_N = 8


# ----------------------------------------------------------------------
# Internal helpers (plain numpy = fast)
# ----------------------------------------------------------------------
def _log_transform(
    x: np.ndarray, neg_strategy: str
) -> tuple[np.ndarray, str, int]:
    """Apply log10, handling zeros and negatives.

    The strategy is chosen by name, not by the sign of the minimum:
    ``'drop'`` discards every ``x <= 0``, whether there are negatives or
    only zeros. Special-casing zeros (sending them to ``log10(0+1) = 0``)
    turns them into artificial outliers that flip the skewness.

    Returns
    -------
    tuple
        ``(transformed_x, label, n_discarded)``.
    """
    if neg_strategy not in NEG_STRATEGIES:
        raise ValueError(
            f"neg_strategy must be one of {NEG_STRATEGIES}, "
            f"not {neg_strategy!r}."
        )

    mn = x.min()
    if mn > 0:
        return np.log10(x), "log(x)", 0

    if neg_strategy == "drop":
        pos = x[x > 0]
        n_out = int(x.size - pos.size)
        if pos.size < MIN_N:
            raise ValueError(
                f"Only {pos.size} positive values remain after 'drop' "
                f"(minimum {MIN_N})."
            )
        return np.log10(pos), "log(x), x>0", n_out

    if neg_strategy == "signed":
        # Symmetric log: keeps the sign and the ordering.
        return (
            np.sign(x) * np.log10(1.0 + np.abs(x)),
            "sign(x)·log(1+|x|)",
            0,
        )

    # 'shift': move the distribution into the positive domain.
    label = "log(x + 1)" if mn == 0 else f"log(x - min + 1)  [min={mn:,.4g}]"
    return np.log10(x - mn + 1.0), label, 0


def _fd_bins(x: np.ndarray, q1: float, q3: float, max_bins: int = 120) -> int:
    """Freedman-Diaconis rule, capped (avoids thousands of useless bins)."""
    iqr = q3 - q1
    rng = x[-1] - x[0]  # x arrives sorted
    if iqr <= 0 or rng <= 0:
        return int(np.clip(np.sqrt(x.size), 10, max_bins))
    h = 2.0 * iqr * x.size ** (-1 / 3)
    return int(np.clip(np.ceil(rng / h), 10, max_bins))


def _hist_density(
    xs: np.ndarray, log_bins: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Density histogram, binned in log space when ``log_bins``.

    Uniform bins under a log axis pile every bar against the left edge,
    so on a log scale the binning happens on log10(x) and the edges are
    mapped back with ``10**``. The density is then per log-unit, which is
    exactly what ``kdeplot(log_scale=True)`` estimates, so histogram and
    KDE share one vertical scale.

    Returns
    -------
    tuple
        ``(density, edges)`` with edges in original units.
    """
    base = np.log10(xs) if log_bins else xs
    b_q1, b_q3 = np.percentile(base, [25, 75])
    counts, edges = np.histogram(base, bins=_fd_bins(base, b_q1, b_q3))
    density = counts / (counts.sum() * np.diff(edges))
    return density, (10**edges if log_bins else edges)


def _warn_ties(x: np.ndarray, name: str) -> None:
    """Warn when ties are dense enough to void a continuous test."""
    n_unique = np.unique(x).size
    if n_unique < TIES_UNIQUE_RATIO * x.size:
        warnings.warn(
            f"{name} assumes a continuous distribution, but the sample "
            f"has {n_unique:,} distinct values across {x.size:,} "
            f"observations. With that many ties the statistic is biased "
            f"and the p-value is not interpretable.",
            stacklevel=3,
        )


def _normality_test(x: np.ndarray, test: str) -> tuple[str, float, float, int]:
    """Pick the test from the size of THIS sample.

    Every sample resolves ``'auto'`` with its own n, so the test on ``x``
    and the test on ``log(x)`` may end up being different tests over
    different sizes. That is deliberate: the ``n`` is returned as well
    and the panel prints it, so the difference is visible and no one
    compares p-values that are not comparable.

    Returns
    -------
    tuple
        ``(name, statistic, p_value, n)``.
    """
    n = int(x.size)
    if test == "auto":
        test = "shapiro" if n <= SHAPIRO_MAX_N else "dagostino"

    if test == "shapiro":
        # The 5000 cap is about the p-value approximation, not the cost:
        # the algorithm is O(n log n). scipy warns on its own.
        _warn_ties(x, "Shapiro-Wilk")
        s, p = stats.shapiro(x)
        return "Shapiro-Wilk", float(s), float(p), n

    if test == "dagostino":  # skewness + kurtosis, cheap for large n
        s, p = stats.normaltest(x)
        return "D'Agostino-Pearson K²", float(s), float(p), n

    if test == "ks":
        sd = x.std(ddof=1)
        if sd <= 0 or not np.isfinite(sd):
            return "Lilliefors (KS)", np.nan, np.nan, n
        _warn_ties(x, "Lilliefors (KS)")
        # lilliefors estimates mean and sd internally and applies the
        # correction for the estimated parameters.
        s, p = lilliefors(x, dist="norm")
        return "Lilliefors (KS)", float(s), float(p), n

    raise ValueError(f"test must be one of {TESTS}, not {test!r}.")


def _fmt_block(
    name: str,
    stat: float,
    p: float,
    n: int,
    label: str,
    alpha: float,
) -> tuple[bool | None, str]:
    """Text block for one test. Returns ``(verdict, text)``.

    ``verdict`` is ``None`` when there is no p-value: without one H0 can
    be neither rejected nor retained, and treating it as rejected (which
    is what ``nan > alpha`` does) would invent a conclusion.

    The returned text is in Spanish because it is drawn on the panel.
    """
    head = f"{label}\n{name} · n = {n:,}"
    stat_s = "n/d" if not np.isfinite(stat) else f"{stat:,.5g}"

    if p is None or not np.isfinite(p):
        return None, (
            f"{head}\nestadístico = {stat_s} · p = n/d\n"
            f"Resultado no concluyente"
        )

    if name.startswith("Lilliefors") and p <= LILLIEFORS_P_FLOOR:
        # Floor of the statsmodels table, not a measured value.
        ps = f"< {LILLIEFORS_P_FLOOR:g}"
    elif p <= 0.0:
        # An exact p = 0 is underflow, not a measurement.
        ps = "< 1e-300"
    elif p >= 1e-4:
        ps = f"= {p:.4f}"
    else:
        ps = f"= {p:.2e}"

    ok = bool(p > alpha)
    verdict = "NO se rechaza H0" if ok else "Se rechaza H0"
    return ok, (
        f"{head}\nestadístico = {stat_s} · p {ps}\nα = {alpha} → {verdict}"
    )


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def normality_report(
    df: pd.DataFrame,
    col: str,
    log_scale: bool = False,
    *,
    log_transformation: bool = False,
    neg_strategy: str = "shift",  # 'shift' | 'signed' | 'drop'
    test: str = "auto",  # 'auto' | 'shapiro' | 'dagostino' | 'ks'
    alpha: float = 0.05,
    color: str = C_BLUE,
    kde: bool = True,
    max_qq_points: int = 5_000,  # points actually drawn on the Q-Q
    max_fliers: int = 2_000,  # outliers actually drawn on the boxplot
    kde_max_n: int = 20_000,  # above this the KDE is subsampled
    figsize: tuple[float, float] = (17, 5.6),
    random_state: int = 0,
    show: bool = True,
) -> dict:
    """
    1x3 panel (histogram | boxplot | Q-Q plot) plus a normality test.

    ``log_scale`` and ``log_transformation`` are different things and are
    mutually exclusive:

    ==========================  ================  =====================
    \\                          ``log_scale``     ``log_transformation``
    ==========================  ================  =====================
    Data                        untouched (x)     log10(x)
    Axes                        log10 scale       linear
    mean/median/skew/kurtosis   on x              on log10(x)
    Normality tests             1, on x           2: log(x) and x
    ``neg_strategy``            only 'drop'       all three
    ==========================  ================  =====================

    Parameters
    ----------
    df : source DataFrame.
    col : name of the numeric column.
    log_scale : draw the axes on a log10 scale without touching the data.
        Values ``x <= 0`` cannot be rendered on a log axis and are
        dropped from the plot, the statistics and the test alike, so all
        three describe the same rows.
    log_transformation : replace the data with log10(x) before analysing
        it, following ``neg_strategy``.
    neg_strategy : how to treat ``x <= 0`` under ``log_transformation``.
        'shift'  -> log10(x - min + 1)  (keeps every row)
        'signed' -> sign(x)·log10(1+|x|) (symmetric log around 0)
        'drop'   -> discard x <= 0 (zeros included)
    test : normality test. ``'auto'`` uses Shapiro-Wilk for n <= 5000 and
        K² above it. Under ``log_transformation`` TWO independent tests
        run — one on x and one on the transform — and each resolves
        ``'auto'`` with its own n, so they may be different tests. The
        panel prints the name, n and statistic of each.

    Returns
    -------
    dict
        ``n``, ``n_dropped``, ``n_dropped_nonpositive``, ``mode``,
        ``transform``, ``mean``, ``median``, ``std``, ``skew``,
        ``kurtosis``, ``test``, ``statistic``, ``p_value``,
        ``is_normal``, ``n_test``, the ``*_raw`` counterparts of the test
        on untransformed x (``None``/``nan`` unless
        ``log_transformation``), and ``fig``.
    """
    if col not in df.columns:
        raise KeyError(f"'{col}' is not in the DataFrame.")
    if log_scale and log_transformation:
        raise ValueError(
            "log_scale and log_transformation are mutually exclusive: "
            "log_scale only rescales the axes while log_transformation "
            "replaces the data, so enabling both would apply the "
            "logarithm twice to the view. Pick one."
        )
    if test not in TESTS:
        raise ValueError(f"test must be one of {TESTS}, not {test!r}.")
    if (log_scale or log_transformation) and neg_strategy not in (
        NEG_STRATEGIES
    ):
        raise ValueError(
            f"neg_strategy must be one of {NEG_STRATEGIES}, "
            f"not {neg_strategy!r}."
        )

    if log_transformation:
        mode = "log_transformation"
    elif log_scale:
        mode = "log_scale"
    else:
        mode = "identity"

    # --- 0. DATA: single pass, contiguous float64 numpy ---
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    n_input = x.size
    x = x[np.isfinite(x)]
    n_dropped = n_input - x.size
    if x.size < MIN_N:
        raise ValueError(
            f"'{col}' has only {x.size} valid values (minimum {MIN_N})."
        )

    transform = "identidad"
    x_raw = None
    n_dropped_np = 0

    if mode == "log_transformation":
        x_raw = x
        x, transform, n_dropped_np = _log_transform(x, neg_strategy)
        x = x[np.isfinite(x)]
        if x.size < MIN_N:
            raise ValueError(
                f"'{col}' is left with {x.size} finite values after "
                f"'{transform}' (minimum {MIN_N})."
            )
    elif mode == "log_scale":
        # Nothing is transformed here, but a log axis cannot render
        # x <= 0. Those rows leave the plot, the statistics and the test
        # together, so all three describe the same set.
        positive = x[x > 0]
        n_dropped_np = int(x.size - positive.size)
        if n_dropped_np:
            if neg_strategy != "drop":
                warnings.warn(
                    f"neg_strategy={neg_strategy!r} does not apply under "
                    f"log_scale, which transforms nothing. The "
                    f"{n_dropped_np:,} non-positive values of '{col}' "
                    f"cannot be drawn on a log axis and were dropped "
                    f"from the plot, the statistics and the test.",
                    stacklevel=2,
                )
            if positive.size < MIN_N:
                raise ValueError(
                    f"'{col}' is left with {positive.size} positive "
                    f"values, which is what a log axis can show "
                    f"(minimum {MIN_N})."
                )
            x = positive

    log_axis = mode == "log_scale"

    # Sort ONCE: serves quantiles, boxplot and Q-Q (O(n log n) overall)
    xs = np.sort(x)
    n = xs.size
    q1, med, q3 = np.percentile(xs, [25, 50, 75], method="linear")
    iqr = q3 - q1
    mean, std = xs.mean(), xs.std(ddof=1)
    degenerate = not np.isfinite(std) or std <= 0  # constant column
    # Unbiased estimators; meaningless without variance.
    skew, kurt = (
        (np.nan, np.nan)
        if degenerate
        else (
            float(stats.skew(xs, bias=False)),
            float(stats.kurtosis(xs, bias=False)),
        )
    )

    # --- 1. CONFIGURATION (scoped: does not mutate global state) ---
    with plt.rc_context():
        sns.set_theme(style="white", context="notebook")
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["text.color"] = "#333333"

        # --- 2. BUILD THE FIGURE ---
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        ax_hist, ax_box, ax_qq = axes

        if mode == "log_transformation":
            unit = f"{col} [{transform}]"
        elif log_axis:
            unit = f"{col} (eje log)"
        else:
            unit = col

        fig.text(
            0.06,
            0.955,
            f"Diagnóstico de Distribución: {col}",
            fontsize=20,
            fontweight="bold",
            color=C_DARK,
        )
        if mode == "log_transformation":
            scale_txt = "Escala logarítmica"
        elif log_axis:
            scale_txt = "Escala logarítmica (solo eje)"
        else:
            scale_txt = "Escala original"
        sub = f"{scale_txt} · n = {n:,}"
        if n_dropped:
            sub += f" · {n_dropped:,} nulos descartados"
        if n_dropped_np:
            sub += f" · {n_dropped_np:,} no positivos descartados"
        if mode == "log_transformation":
            sub += f" · transformación: {transform}"
        fig.text(0.06, 0.905, sub, fontsize=12.5, color=C_SOFT)

        # ---------- PANEL 1: HISTOGRAM ----------
        ax_hist.grid(
            axis="y", color=C_GRID, linestyle="-", linewidth=1, zorder=0
        )

        # KDE: estimated on a subsample when n is large. Note that this
        # widens the bandwidth (Scott ∝ n^-1/5) and over-smooths slightly
        # against the histogram, which does use every point.
        kde_data, use_kde = xs, kde and not degenerate
        if use_kde and n > kde_max_n:
            rng = np.random.default_rng(random_state)
            kde_data = rng.choice(xs, kde_max_n, replace=False)

        # stairs takes edges and heights as they are, with no implicit
        # renormalisation, so non-uniform (log-spaced) bins stay correct.
        density, edges = _hist_density(xs, log_axis)
        ax_hist.stairs(
            density,
            edges,
            fill=True,
            facecolor=color,
            edgecolor="none",
            alpha=0.85,
            zorder=2,
        )
        if use_kde:
            sns.kdeplot(
                x=kde_data,
                ax=ax_hist,
                log_scale=log_axis,
                color=C_DARK,
                linewidth=1.6,
                zorder=3,
            )
        if log_axis:
            ax_hist.set_xscale("log")

        ax_hist.axvline(
            med,
            color=C_DARK,
            ls="--",
            lw=1.4,
            zorder=4,
            label=f"Mediana: {med:,.3g}",
        )
        ax_hist.axvline(
            mean,
            color=C_ORANGE,
            ls=":",
            lw=1.8,
            zorder=4,
            label=f"Media: {mean:,.3g}",
        )
        ax_hist.legend(frameon=False, fontsize=9.5, loc="upper right")

        ax_hist.set_title(
            "Histograma y Densidad",
            loc="left",
            fontsize=14,
            fontweight="bold",
            pad=15,
            color=C_MID,
        )
        ax_hist.set_xlabel(unit, fontsize=11, fontweight="medium")
        ax_hist.set_ylabel("Densidad", fontsize=11, fontweight="medium")

        # ---------- PANEL 2: BOXPLOT ----------
        ax_box.grid(
            axis="x", color=C_GRID, linestyle="-", linewidth=1, zorder=0
        )

        # Tukey fences stay in the original space even under log_scale:
        # the IQR and the outlier count printed here describe x, exactly
        # like the test does.
        # xs is sorted -> whiskers and outliers by binary search.
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        i_lo = int(np.searchsorted(xs, lo, side="left"))
        i_hi = int(np.searchsorted(xs, hi, side="right"))
        whislo = xs[i_lo] if i_lo < n else xs[0]
        whishi = xs[i_hi - 1] if i_hi > 0 else xs[-1]
        out = np.concatenate([xs[:i_lo], xs[i_hi:]])
        n_out = out.size

        # bxp draws from the summary already computed => O(1) in n
        ax_box.bxp(
            [
                dict(
                    label="",
                    med=med,
                    q1=q1,
                    q3=q3,
                    whislo=whislo,
                    whishi=whishi,
                    fliers=[],
                )
            ],
            positions=[0],
            orientation="horizontal",
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color, edgecolor="none", alpha=0.85),
            medianprops=dict(color=C_DARK, linewidth=2),
            whiskerprops=dict(color=C_GREY, linewidth=1.3),
            capprops=dict(color=C_GREY, linewidth=1.3),
            zorder=2,
        )
        ax_box.set_ylim(-0.6, 0.6)
        if log_axis:
            ax_box.set_xscale("log")
        if n_out:
            draw = out
            if n_out > max_fliers:
                rng = np.random.default_rng(random_state)
                draw = rng.choice(out, max_fliers, replace=False)
            jitter = np.random.default_rng(random_state).uniform(
                -0.06, 0.06, draw.size
            )
            ax_box.scatter(
                draw,
                jitter,
                s=12,
                color=C_GREY,
                alpha=0.45,
                edgecolors="none",
                zorder=3,
            )

        ax_box.scatter([mean], [0], marker="D", s=45, color=C_ORANGE, zorder=5)
        ax_box.set_title(
            "Dispersión y Outliers",
            loc="left",
            fontsize=14,
            fontweight="bold",
            pad=15,
            color=C_MID,
        )
        ax_box.set_xlabel(unit, fontsize=11, fontweight="medium")
        ax_box.set_yticks([])

        box_txt = f"IQR {iqr:,.4g}\nOutliers  {n_out:,} ({n_out / n:.1%})"
        ax_box.text(
            0.02,
            0.04,
            box_txt,
            transform=ax_box.transAxes,
            fontsize=9.5,
            va="bottom",
            ha="left",
            color=C_MID,
            linespacing=1.5,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=C_BOX,
                edgecolor="#dddddd",
            ),
        )

        # ---------- PANEL 3: Q-Q PLOT ----------
        ax_qq.grid(color=C_GRID, linestyle="-", linewidth=1, zorder=0)

        # Quantile subsampling: keeps the exact shape of the tails and
        # honours max_qq_points (drawing n points on a 10^5-row dataset
        # adds nothing and costs seconds).
        if n > max_qq_points:
            idx = np.unique(
                np.linspace(0, n - 1, max_qq_points).astype(np.int64)
            )
        else:
            idx = np.arange(n)
        sample_q = xs[idx]
        probs = (idx + 0.5) / n  # Hazen plotting positions
        theor_q = stats.norm.ppf(probs)

        ax_qq.scatter(
            theor_q,
            sample_q,
            s=11,
            color=color,
            alpha=0.6,
            edgecolors="none",
            zorder=2,
        )
        if log_axis:
            ax_qq.set_yscale("log")

        # Reference line through the quartiles (same idea as line="q"):
        # robust to the tails, unlike a mean/sd line. Under a log axis it
        # is fitted on log10(x) so that it comes out straight, which is
        # why it then means lognormality rather than normality.
        tq1, tq3 = stats.norm.ppf([0.25, 0.75])
        if iqr > 0:
            ref_x = np.array([theor_q[0], theor_q[-1]])
            if log_axis:
                lq1, lq3 = np.percentile(np.log10(xs), [25, 75])
                slope = (lq3 - lq1) / (tq3 - tq1)
                ref_y = 10 ** (lq1 + slope * (ref_x - tq1))
                ref_label = "Referencia lognormal"
            else:
                slope = iqr / (tq3 - tq1)
                ref_y = q1 + slope * (ref_x - tq1)
                ref_label = "Referencia normal"
            ax_qq.plot(
                ref_x,
                ref_y,
                color=C_ORANGE,
                linewidth=1.8,
                zorder=3,
                label=ref_label,
            )
            ax_qq.legend(frameon=False, fontsize=9.5, loc="upper left")

        ax_qq.set_title(
            "Q-Q Plot Lognormal" if log_axis else "Q-Q Plot Normal",
            loc="left",
            fontsize=14,
            fontweight="bold",
            pad=15,
            color=C_MID,
        )
        ax_qq.set_xlabel(
            "Cuantiles teóricos", fontsize=11, fontweight="medium"
        )
        ax_qq.set_ylabel(
            f"Cuantiles observados · {unit}",
            fontsize=11,
            fontweight="medium",
        )

        # --- HYPOTHESIS TESTS ---
        # H0: the sample comes from a normal distribution.
        name_raw: str | None = None
        statistic_raw = p_value_raw = np.nan
        n_test_raw = 0
        is_normal_raw: bool | None = None
        info_raw = ""

        if degenerate:
            name, statistic, p_value = "sin varianza", np.nan, np.nan
            n_test, is_normal = n, None
            info = "Columna constante\nEl test de normalidad no es aplicable"
        else:
            name, statistic, p_value, n_test = _normality_test(xs, test)
            label = (
                f"H0: {transform} es normal"
                if mode == "log_transformation"
                else "H0: los datos son normales"
            )
            is_normal, info = _fmt_block(
                name, statistic, p_value, n_test, label, alpha
            )

            if mode == "log_transformation":
                # Independent test on untransformed x: it picks its own
                # test from ITS n, which under 'drop' is not even the
                # same n. That is why each block carries its own.
                (
                    name_raw,
                    statistic_raw,
                    p_value_raw,
                    n_test_raw,
                ) = _normality_test(x_raw, test)
                is_normal_raw, info_raw = _fmt_block(
                    name_raw,
                    statistic_raw,
                    p_value_raw,
                    n_test_raw,
                    "H0: x (sin transformar) es normal",
                    alpha,
                )

        # Under log_scale the reference line and the verdict answer
        # different questions, and the figure has to say so rather than
        # leaving it to the documentation.
        caveat = (
            "La recta evalúa lognormalidad de x.\n"
            "El test contrasta normalidad de x."
            if log_axis and not degenerate
            else ""
        )
        blocks = [b for b in (info, info_raw, caveat) if b]

        edge = (
            C_GREY
            if is_normal is None
            else (C_BLUE if is_normal else C_ORANGE)
        )
        ax_qq.text(
            0.97,
            0.05,
            "\n\n".join(blocks),
            transform=ax_qq.transAxes,
            fontsize=8.5,
            va="bottom",
            ha="right",
            color=C_MID,
            linespacing=1.25,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=C_BOX,
                edgecolor=edge,
                linewidth=1.4,
            ),
        )

        # --- FINAL STYLING ---
        for a in axes:
            sns.despine(ax=a, left=True, bottom=True)
            a.tick_params(colors=C_SOFT, labelsize=9.5)

        plt.subplots_adjust(
            top=0.78,
            left=0.06,
            right=0.97,
            bottom=0.13,
            wspace=0.25,
        )
        if show:
            plt.show()

    # Always closed: the notebook loops over columns and matplotlib does
    # not release figures on its own. `fig` still accepts savefig().
    plt.close(fig)

    return {
        "n": int(n),
        "n_dropped": int(n_dropped),
        "n_dropped_nonpositive": int(n_dropped_np),
        "mode": mode,
        "transform": transform,
        "mean": float(mean),
        "median": float(med),
        "std": float(std),
        "skew": skew,
        "kurtosis": kurt,
        "test": name,
        "statistic": statistic,
        "p_value": p_value,
        "is_normal": is_normal,
        "n_test": int(n_test),
        "test_raw": name_raw,
        "statistic_raw": statistic_raw,
        "p_value_raw": p_value_raw,
        "is_normal_raw": is_normal_raw,
        "n_raw": int(n_test_raw),
        "fig": fig,
    }
