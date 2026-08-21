"""Kind-aware EDA plotting built on matplotlib and seaborn.

The module exposes a single class, :class:`EDAPlotter`. Every method
routes through one variable-kind resolver
(:meth:`EDAPlotter.resolve_kind`) and one style system
(:class:`StyleConfig`), so charts come out consistent without the caller
passing styling arguments.

Importing this module has no side effects: no theme is applied, no
figure is created and the global matplotlib state is left untouched.
Styling is scoped to a ``matplotlib.rc_context`` opened inside each
method.

Examples
--------
>>> import pandas as pd
>>> from utils.plotting import EDAPlotter
>>> df = pd.DataFrame({"price": [1.5, 2.25, 3.75, 4.1]})
>>> plotter = EDAPlotter(df)
>>> plotter.resolve_kind(df["price"])
'numeric'
"""

from __future__ import annotations

import textwrap
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels  # noqa: F401
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.ticker import FuncFormatter
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)

Kind = Literal["numeric", "categorical", "datetime"]
Nonpositive = Literal["raise", "mask", "clip", "shift"]

__all__ = ["EDAPlotter", "StyleConfig", "TransformMeta"]


# ---------------------------------------------------------------------
# Interpretation strings. Keyed by language then phrase id so that no
# user-facing sentence is an f-string buried in a method body.
# ---------------------------------------------------------------------
_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "panel_hist": "Distribution",
        "panel_box": "Spread & outliers",
        "panel_qq": "Normal Q-Q",
        "panel_stats": "Summary",
        "n_missing": "n = {n:,}, missing = {missing:,} ({pct:.1f}%)",
        "stat_n": "n",
        "stat_missing": "missing",
        "stat_mean": "mean",
        "stat_median": "median",
        "stat_std": "std",
        "stat_iqr": "IQR",
        "stat_min": "min",
        "stat_max": "max",
        "stat_skew": "skewness",
        "stat_kurtosis": "excess kurtosis",
        "test_reject": (
            "{test}: statistic = {stat:.4f}, {p} (alpha = {alpha})\n"
            "{p_cmp} < {alpha} -> the normality hypothesis is rejected."
        ),
        "test_keep": (
            "{test}: statistic = {stat:.4f}, {p} (alpha = {alpha})\n"
            "{p_cmp} >= {alpha} -> no evidence against normality at "
            "alpha = {alpha}."
        ),
        "test_anderson_reject": (
            "{test}: statistic = {stat:.4f}, critical value at "
            "{alpha_pct:.0f}% = {crit:.4f}\n"
            "statistic > critical value -> the normality hypothesis is "
            "rejected."
        ),
        "test_anderson_keep": (
            "{test}: statistic = {stat:.4f}, critical value at "
            "{alpha_pct:.0f}% = {crit:.4f}\n"
            "statistic <= critical value -> no evidence against "
            "normality at alpha = {alpha}."
        ),
        "shape_right": "right-skewed",
        "shape_left": "left-skewed",
        "shape_sym": "roughly symmetric",
        "tail_heavy": "heavy-tailed",
        "tail_light": "light-tailed",
        "tail_meso": "mesokurtic",
        "shape_line": (
            "Shape: {shape} (skew = {skew:.2f}) and {tail} "
            "(excess kurtosis = {kurt:.2f})."
        ),
        "caveat_large_n": (
            "With n = {n:,}, virtually any real sample rejects; read the "
            "Q-Q panel rather than the p-value."
        ),
        "caveat_small_n": (
            "With n = {n:,}, the test has little power to detect "
            "departures from normality."
        ),
        "caveat_ties": (
            "Only {k} distinct values: heavy ties make this test "
            "unreliable and produce a staircase Q-Q pattern."
        ),
        "suggest_transform": (
            "Right-skewed and strictly positive: try transform='{name}'."
        ),
        "raw_vs_transformed": "{test} p: {raw} raw -> {new} after {name}",
        "log_scale_note": "log scale (base {base})",
        "symlog_note": "symlog scale (base {base}, linthresh = {thr:g})",
        "whisker_note": "whiskers: {whis} x IQR - outliers: {n_out:,}",
        "lambda_note": "{name} lambda = {lam:.4f}",
        "auto_transform": "auto-selected transform '{name}': {why}",
        "why_right_pos": "right-skewed and strictly positive",
        "why_right_zero": "right-skewed with zeros present",
        "why_negative": "skewed with non-positive values present",
        "why_left": "left-skewed",
        "legend_mean": "mean",
        "legend_median": "median",
        "legend_normal": "normal fit",
        "legend_band": "{level:.0f}% confidence band",
        "axis_theoretical": "Theoretical quantiles",
        "axis_sample": "Sample quantiles",
        "axis_count": "Count",
        "n_label": "n = {n:,}",
    },
    "es": {
        "panel_hist": "Distribucion",
        "panel_box": "Dispersion y atipicos",
        "panel_qq": "Q-Q normal",
        "panel_stats": "Resumen",
        "n_missing": "n = {n:,}, faltantes = {missing:,} ({pct:.1f}%)",
        "stat_n": "n",
        "stat_missing": "faltantes",
        "stat_mean": "media",
        "stat_median": "mediana",
        "stat_std": "desv. est.",
        "stat_iqr": "RIC",
        "stat_min": "min",
        "stat_max": "max",
        "stat_skew": "asimetria",
        "stat_kurtosis": "curtosis en exceso",
        "test_reject": (
            "{test}: estadistico = {stat:.4f}, {p} (alfa = {alpha})\n"
            "{p_cmp} < {alpha} -> se rechaza la hipotesis de normalidad."
        ),
        "test_keep": (
            "{test}: estadistico = {stat:.4f}, {p} (alfa = {alpha})\n"
            "{p_cmp} >= {alpha} -> no hay evidencia contra la normalidad "
            "con alfa = {alpha}."
        ),
        "test_anderson_reject": (
            "{test}: estadistico = {stat:.4f}, valor critico al "
            "{alpha_pct:.0f}% = {crit:.4f}\n"
            "estadistico > valor critico -> se rechaza la hipotesis de "
            "normalidad."
        ),
        "test_anderson_keep": (
            "{test}: estadistico = {stat:.4f}, valor critico al "
            "{alpha_pct:.0f}% = {crit:.4f}\n"
            "estadistico <= valor critico -> no hay evidencia contra la "
            "normalidad con alfa = {alpha}."
        ),
        "shape_right": "asimetrica a la derecha",
        "shape_left": "asimetrica a la izquierda",
        "shape_sym": "aproximadamente simetrica",
        "tail_heavy": "de colas pesadas",
        "tail_light": "de colas ligeras",
        "tail_meso": "mesocurtica",
        "shape_line": (
            "Forma: {shape} (asimetria = {skew:.2f}) y {tail} "
            "(curtosis en exceso = {kurt:.2f})."
        ),
        "caveat_large_n": (
            "Con n = {n:,}, casi cualquier muestra real se rechaza; lea "
            "el panel Q-Q en lugar del valor p."
        ),
        "caveat_small_n": (
            "Con n = {n:,}, la prueba tiene poca potencia para detectar "
            "desviaciones de la normalidad."
        ),
        "caveat_ties": (
            "Solo {k} valores distintos: los empates hacen la prueba poco "
            "confiable y producen un patron escalonado en el Q-Q."
        ),
        "suggest_transform": (
            "Asimetrica a la derecha y estrictamente positiva: pruebe "
            "transform='{name}'."
        ),
        "raw_vs_transformed": (
            "{test} p: {raw} crudo -> {new} despues de {name}"
        ),
        "log_scale_note": "escala log (base {base})",
        "symlog_note": "escala symlog (base {base}, linthresh = {thr:g})",
        "whisker_note": "bigotes: {whis} x RIC - atipicos: {n_out:,}",
        "lambda_note": "{name} lambda = {lam:.4f}",
        "auto_transform": (
            "transformacion '{name}' elegida automaticamente: {why}"
        ),
        "why_right_pos": "asimetrica a la derecha y estrictamente positiva",
        "why_right_zero": "asimetrica a la derecha con ceros presentes",
        "why_negative": "asimetrica con valores no positivos presentes",
        "why_left": "asimetrica a la izquierda",
        "legend_mean": "media",
        "legend_median": "mediana",
        "legend_normal": "ajuste normal",
        "legend_band": "banda de confianza {level:.0f}%",
        "axis_theoretical": "Cuantiles teoricos",
        "axis_sample": "Cuantiles muestrales",
        "axis_count": "Frecuencia",
        "n_label": "n = {n:,}",
    },
}


def _t(lang: str, key: str, **kwargs: Any) -> str:
    """Look up an interpretation string and format it.

    Falls back to English when a phrase is missing from the requested
    language, so a partially translated catalog never raises.

    Parameters
    ----------
    lang : str
        Language code, ``"en"`` or ``"es"``.
    key : str
        Phrase id within :data:`_STRINGS`.
    **kwargs
        Formatting arguments for the phrase.

    Returns
    -------
    str
        The formatted phrase.

    Examples
    --------
    >>> _t("en", "stat_mean")
    'mean'
    """
    table = _STRINGS.get(lang, _STRINGS["en"])
    template = table.get(key) or _STRINGS["en"][key]
    return template.format(**kwargs) if kwargs else template


# ---------------------------------------------------------------------
# Style tokens
# ---------------------------------------------------------------------
_CATEGORICAL_COLORS = (
    "#5dade2",
    "#95a5a6",
    "#58d68d",
    "#f5b041",
    "#af7ac5",
    "#ec7063",
    "#48c9b0",
    "#f7dc6f",
)

#: Level sets that trigger the two-colour ``binary`` palette. Compared
#: case-insensitively against the string form of the levels.
_BOOLEAN_LEVEL_SETS = (
    frozenset({"false", "true"}),
    frozenset({"0", "1"}),
    frozenset({"no", "yes"}),
    frozenset({"n", "y"}),
    frozenset({"no", "si"}),
)

_FALSEY_TOKENS = frozenset({"false", "0", "no", "n", "0.0"})


@dataclass
class StyleConfig:
    """Visual tokens shared by every :class:`EDAPlotter` method.

    Holds every colour, size and layout constant the class uses. No
    plotting method hard-codes a hex value; each reads it from here, so
    restyling the whole utility means replacing one object.

    Attributes
    ----------
    enabled : bool
        When ``False`` the whole style system is bypassed and plain
        matplotlib defaults are used.
    figsize : tuple of float
        Default size of a single panel, in inches.
    margins : dict
        Default ``subplots_adjust`` margins for a single panel.

    Examples
    --------
    >>> StyleConfig().grid_color
    '#eeeeee'
    """

    enabled: bool = True

    font_family: str = "sans-serif"
    text_color: str = "#333333"
    title_color: str = "#2c3e50"
    subtitle_color: str = "#7f8c8d"
    panel_title_color: str = "#34495e"
    grid_color: str = "#eeeeee"

    title_size: float = 20.0
    subtitle_size: float = 13.0
    panel_title_size: float = 14.0
    label_size: float = 11.0
    tick_size: float = 10.0
    annot_size: float = 10.0

    panel_title_pad: float = 15.0
    panel_title_loc: str = "left"
    label_weight: str = "medium"

    figsize: tuple[float, float] = (8.0, 6.0)
    grid_zorder: int = 0
    data_zorder: int = 2
    grid_linewidth: float = 1.0
    line_width: float = 2.0
    marker_edge_width: float = 0.5

    margins: dict[str, float] = field(
        default_factory=lambda: {
            "top": 0.85,
            "left": 0.15,
            "right": 0.90,
            "bottom": 0.15,
        }
    )

    binary_true: str = "#5dade2"
    binary_false: str = "#95a5a6"
    categorical: tuple[str, ...] = _CATEGORICAL_COLORS
    sequential_base: str = "#5dade2"
    diverging_low: str = "#ec7063"
    diverging_high: str = "#5dade2"
    positive: str = "#58d68d"
    negative: str = "#ec7063"

    def replace(self, **changes: Any) -> StyleConfig:
        """Return a copy of this config with ``changes`` applied.

        Parameters
        ----------
        **changes
            Attribute names and their new values.

        Returns
        -------
        StyleConfig
            A new instance; the original is not mutated.

        Examples
        --------
        >>> StyleConfig().replace(tick_size=8).tick_size
        8
        """
        return replace(self, **changes)

    def rc_params(self) -> dict[str, Any]:
        """Build the ``rcParams`` mapping for a scoped context.

        Returns
        -------
        dict
            Keys accepted by :func:`matplotlib.rc_context`.

        Examples
        --------
        >>> StyleConfig().rc_params()["text.color"]
        '#333333'
        """
        if not self.enabled:
            return {}
        return {
            "font.family": self.font_family,
            "text.color": self.text_color,
            "axes.labelcolor": self.text_color,
            "axes.edgecolor": self.text_color,
            "axes.labelsize": self.label_size,
            "axes.titlesize": self.panel_title_size,
            "xtick.color": self.text_color,
            "ytick.color": self.text_color,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "legend.fontsize": self.tick_size,
            "legend.frameon": False,
            "figure.figsize": self.figsize,
            "grid.color": self.grid_color,
            "grid.linewidth": self.grid_linewidth,
            "savefig.facecolor": "white",
        }

    def sequential_cmap(self) -> LinearSegmentedColormap:
        """Single-hue ramp from white to :attr:`sequential_base`.

        Returns
        -------
        matplotlib.colors.LinearSegmentedColormap
            A colormap usable anywhere matplotlib accepts one.

        Examples
        --------
        >>> StyleConfig().sequential_cmap().name
        'eda_sequential'
        """
        return LinearSegmentedColormap.from_list(
            "eda_sequential", ["#ffffff", self.sequential_base]
        )

    def diverging_cmap(self) -> LinearSegmentedColormap:
        """Ramp running low -> white -> high, centred on white.

        Returns
        -------
        matplotlib.colors.LinearSegmentedColormap
            Colormap for correlation-style matrices centred at zero.

        Examples
        --------
        >>> StyleConfig().diverging_cmap().name
        'eda_diverging'
        """
        return LinearSegmentedColormap.from_list(
            "eda_diverging",
            [self.diverging_low, "#ffffff", self.diverging_high],
        )


# ---------------------------------------------------------------------
# Transform machinery
# ---------------------------------------------------------------------
@dataclass
class TransformMeta:
    """What :func:`_apply_transform` did to a vector of values.

    Carried alongside the transformed data so callers can label axes,
    invert an aggregate and report how many values were altered without
    recomputing anything.

    Attributes
    ----------
    name : str
        Transform name, ``"identity"`` when nothing was applied.
    label : str
        Axis label fragment, e.g. ``"log10(price)"``.
    inverse : callable or None
        Maps transformed values back to the original units. ``None``
        when the transform is not invertible.

    Examples
    --------
    >>> import numpy as np
    >>> _, meta = _apply_transform(np.array([1.0, 10.0]), "log10", "raise")
    >>> meta.name
    'log10'
    """

    name: str = "identity"
    base: float | None = None
    lam: float | None = None
    offset: float = 0.0
    n_dropped: int = 0
    n_clipped: int = 0
    label: str = ""
    inverse: Callable[[np.ndarray], np.ndarray] | None = None
    note: str | None = None

    def label_for(self, column: str) -> str:
        """Render the axis label for ``column`` under this transform.

        Parameters
        ----------
        column : str
            Original variable name.

        Returns
        -------
        str
            ``column`` itself for the identity transform, otherwise the
            transformed name including any applied offset.

        Examples
        --------
        >>> TransformMeta().label_for("price")
        'price'
        """
        if self.name == "identity":
            return column
        inner = column
        if self.offset:
            inner = f"{column} + {self.offset:.4g}"
        if self.lam is not None:
            return f"{self.label}({inner}), lambda = {self.lam:.4g}"
        return f"{self.label}({inner})"


#: Transform name -> (domain, axis-label fragment). The domain drives
#: the :func:`_guard_domain` check and therefore the ``nonpositive``
#: policy; ``None`` means the transform accepts the whole real line.
_TRANSFORM_DOMAIN: dict[str, str | None] = {
    "log": "positive",
    "log10": "positive",
    "log2": "positive",
    "log1p": "ge_minus_one",
    "sqrt": "nonnegative",
    "cbrt": None,
    "reciprocal": "nonzero",
    "square": None,
    "boxcox": "positive",
    "yeojohnson": None,
    "zscore": None,
    "robust": None,
    "minmax": None,
    "rank": None,
    "quantile": None,
    "winsorize": None,
}

_DOMAIN_LIMIT: dict[str, float] = {
    "positive": 0.0,
    "nonnegative": 0.0,
    "ge_minus_one": -1.0,
    "nonzero": 0.0,
}


def _domain_violations(values: np.ndarray, domain: str) -> np.ndarray:
    """Boolean mask of the finite values outside ``domain``."""
    finite = np.isfinite(values)
    if domain == "positive":
        bad = values <= 0
    elif domain == "nonnegative":
        bad = values < 0
    elif domain == "ge_minus_one":
        bad = values < -1
    elif domain == "nonzero":
        bad = values == 0
    else:
        bad = np.zeros_like(values, dtype=bool)
    return bad & finite


def _guard_domain(
    values: np.ndarray,
    domain: str | None,
    nonpositive: Nonpositive,
    *,
    column: str,
    operation: str,
) -> tuple[np.ndarray, float, int, int]:
    """Enforce a transform's domain under the ``nonpositive`` policy.

    Offending values are never dropped from the array; ``"mask"``
    replaces them with ``NaN`` so the result stays aligned with the rows
    it came from.

    Parameters
    ----------
    values : numpy.ndarray
        Input values, possibly containing ``NaN``.
    domain : str or None
        Key of :data:`_DOMAIN_LIMIT`, or ``None`` for no restriction.
    nonpositive : {"raise", "mask", "clip", "shift"}
        What to do with values outside the domain.
    column : str
        Variable name, used in the error and warning text.
    operation : str
        What is being attempted, e.g. ``"transform='log'"``.

    Returns
    -------
    values : numpy.ndarray
        Values after the policy has been applied.
    offset : float
        Amount added by the ``"shift"`` policy, else ``0.0``.
    n_dropped : int
        Count replaced with ``NaN`` by the ``"mask"`` policy.
    n_clipped : int
        Count moved by the ``"clip"`` or ``"shift"`` policies.

    Raises
    ------
    ValueError
        Under the default ``"raise"`` policy when any value violates the
        domain.

    Examples
    --------
    >>> import numpy as np
    >>> out, off, n, _ = _guard_domain(
    ...     np.array([0.0, 1.0]),
    ...     "positive",
    ...     "mask",
    ...     column="x",
    ...     operation="log",
    ... )
    >>> n
    1
    """
    if domain is None:
        return values, 0.0, 0, 0
    bad = _domain_violations(values, domain)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return values, 0.0, 0, 0

    finite = values[np.isfinite(values)]
    vmin = float(finite.min()) if finite.size else float("nan")
    limit = _DOMAIN_LIMIT[domain]
    hint = "log1p" if vmin >= limit else "yeojohnson' or symlog"
    if nonpositive == "raise":
        raise ValueError(
            f"{operation} on {column!r} requires values "
            f"{'> ' if domain == 'positive' else '>= '}{limit:g}, but "
            f"{n_bad:,} of {values.size:,} violate it (min = {vmin:g}). "
            f"Use nonpositive='mask'/'clip'/'shift', or try '{hint}'."
        )

    values = values.astype(float, copy=True)
    if nonpositive == "mask":
        values[bad] = np.nan
        warnings.warn(
            f"{operation} on {column!r}: masked {n_bad:,} value(s) "
            f"outside the domain; n = {values.size - n_bad:,} remain.",
            UserWarning,
            stacklevel=3,
        )
        return values, 0.0, n_bad, 0

    if nonpositive == "clip":
        ok = values[~bad & np.isfinite(values)]
        floor = float(ok.min()) if ok.size else limit + 1e-9
        values[bad] = floor
        warnings.warn(
            f"{operation} on {column!r}: clipped {n_bad:,} value(s) to "
            f"{floor:g}. This fabricates a floor; the figure is "
            f"annotated accordingly.",
            UserWarning,
            stacklevel=3,
        )
        return values, 0.0, 0, n_bad

    offset = float(abs(vmin) + abs(limit) + 1e-9)
    warnings.warn(
        f"{operation} on {column!r}: shifted all values by "
        f"{offset:.4g} to enter the domain. A shifted transform is a "
        f"different variable; the axis label reports the offset.",
        UserWarning,
        stacklevel=3,
    )
    return values + offset, offset, 0, n_bad


def _auto_transform_name(values: np.ndarray, lang: str) -> tuple[str, str]:
    """Pick a transform from skew sign and positivity."""
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return "identity", ""
    skew = float(pd.Series(finite).skew())
    vmin = float(finite.min())
    if skew > 0.5:
        if vmin > 0:
            return "log", _t(lang, "why_right_pos")
        if vmin >= 0:
            return "log1p", _t(lang, "why_right_zero")
        return "yeojohnson", _t(lang, "why_negative")
    if skew < -0.5:
        return "square", _t(lang, "why_left")
    return "identity", ""


def _apply_transform(
    values: np.ndarray | pd.Series,
    spec: str | Callable[[np.ndarray], np.ndarray] | None,
    nonpositive: Nonpositive = "raise",
    *,
    column: str = "value",
    lang: str = "en",
    winsor_limits: tuple[float, float] = (0.01, 0.01),
) -> tuple[np.ndarray, TransformMeta]:
    """Transform ``values`` and describe what was done.

    The single implementation of every transform in the catalog. No
    plotting method calls :func:`numpy.log` directly; they all route
    here so that domain guards, axis labels, fitted parameters and the
    inverse mapping stay consistent.

    Parameters
    ----------
    values : numpy.ndarray or pandas.Series
        Values to transform.
    spec : str, callable or None
        Transform name, a vectorised callable, ``"auto"``, or ``None``
        for the identity.
    nonpositive : {"raise", "mask", "clip", "shift"}, default "raise"
        Domain-violation policy, see :func:`_guard_domain`.
    column : str, default "value"
        Variable name used in messages and in the axis label.
    lang : str, default "en"
        Language for any generated note.
    winsor_limits : tuple of float, default (0.01, 0.01)
        Lower and upper fractions clipped by ``"winsorize"``.

    Returns
    -------
    transformed : numpy.ndarray
        The transformed values, same length as the input.
    meta : TransformMeta
        Name, label, fitted lambda, offset, counts and inverse.

    Raises
    ------
    ValueError
        For an unknown transform name, or a domain violation under the
        default ``nonpositive="raise"``.

    Examples
    --------
    >>> import numpy as np
    >>> out, meta = _apply_transform(np.array([1.0, 100.0]), "log10")
    >>> out.tolist(), meta.label
    ([0.0, 2.0], 'log10')
    """
    arr = np.asarray(
        values.to_numpy() if isinstance(values, pd.Series) else values,
        dtype=float,
    )
    if spec is None:
        return arr, TransformMeta(inverse=lambda a: a)

    note = None
    if callable(spec):
        out = np.asarray(spec(arr), dtype=float)
        meta = TransformMeta(name="custom", label="f", inverse=None)
        return out, meta

    name = str(spec).lower()
    if name == "auto":
        name, why = _auto_transform_name(arr, lang)
        if name == "identity":
            return arr, TransformMeta(inverse=lambda a: a)
        note = _t(lang, "auto_transform", name=name, why=why)
        warnings.warn(note, UserWarning, stacklevel=2)
    if name in {"identity", "none"}:
        return arr, TransformMeta(inverse=lambda a: a, note=note)
    if name not in _TRANSFORM_DOMAIN:
        known = ", ".join(sorted(_TRANSFORM_DOMAIN))
        raise ValueError(
            f"Unknown transform {spec!r}. Known transforms: {known}, "
            f"'auto', or a vectorised callable."
        )

    arr, offset, n_drop, n_clip = _guard_domain(
        arr,
        _TRANSFORM_DOMAIN[name],
        nonpositive,
        column=column,
        operation=f"transform={name!r}",
    )
    out, meta = _compute_transform(arr, name, winsor_limits)
    meta.offset = offset
    meta.n_dropped = n_drop
    meta.n_clipped = n_clip
    meta.note = note
    if offset and meta.inverse is not None:
        inner = meta.inverse
        meta.inverse = lambda a, f=inner, o=offset: f(a) - o
    return out, meta


def _compute_transform(
    arr: np.ndarray,
    name: str,
    winsor_limits: tuple[float, float],
) -> tuple[np.ndarray, TransformMeta]:
    """Apply a catalog transform to already domain-checked values."""
    from scipy import stats as sps

    if name in {"log", "log10", "log2"}:
        base = {"log": float(np.e), "log10": 10.0, "log2": 2.0}[name]
        fn = {"log": np.log, "log10": np.log10, "log2": np.log2}[name]
        label = {"log": "ln", "log10": "log10", "log2": "log2"}[name]
        return fn(arr), TransformMeta(
            name=name,
            base=base,
            label=label,
            inverse=lambda a, b=base: np.power(b, a),
        )
    if name == "log1p":
        return np.log1p(arr), TransformMeta(
            name=name, label="log1p", inverse=np.expm1
        )
    if name == "sqrt":
        return np.sqrt(arr), TransformMeta(
            name=name, label="sqrt", inverse=lambda a: np.power(a, 2)
        )
    if name == "cbrt":
        return np.cbrt(arr), TransformMeta(
            name=name, label="cbrt", inverse=lambda a: np.power(a, 3)
        )
    if name == "reciprocal":
        warnings.warn(
            "transform='reciprocal' reverses the order of the values: "
            "the largest becomes the smallest, so the chart reads "
            "backwards.",
            UserWarning,
            stacklevel=4,
        )
        return 1.0 / arr, TransformMeta(
            name=name, label="1/", inverse=lambda a: 1.0 / a
        )
    if name == "square":
        return np.power(arr, 2), TransformMeta(
            name=name, label="square", inverse=np.sqrt
        )
    if name in {"boxcox", "yeojohnson"}:
        finite = arr[np.isfinite(arr)]
        if name == "boxcox":
            _, lam = sps.boxcox(finite)
            out = sps.boxcox(arr, lmbda=lam)
            inv = _boxcox_inverse(lam)
            label = "Box-Cox"
        else:
            _, lam = sps.yeojohnson(finite)
            out = sps.yeojohnson(arr, lmbda=lam)
            inv = _yeojohnson_inverse(lam)
            label = "Yeo-Johnson"
        return out, TransformMeta(
            name=name, lam=float(lam), label=label, inverse=inv
        )
    if name in {"zscore", "robust", "minmax"}:
        return _rescale(arr, name)
    if name == "rank":
        out = pd.Series(arr).rank().to_numpy()
        return out, TransformMeta(name=name, label="rank", inverse=None)
    if name == "quantile":
        ranks = pd.Series(arr).rank()
        n = int(ranks.notna().sum())
        out = sps.norm.ppf((ranks - 0.375) / (n + 0.25)).to_numpy()
        warnings.warn(
            "transform='quantile' maps the data onto a normal by rank, "
            "which forces normality; any normality test run afterwards "
            "is meaningless.",
            UserWarning,
            stacklevel=4,
        )
        return out, TransformMeta(name=name, label="quantile", inverse=None)
    lo, hi = winsor_limits
    finite = arr[np.isfinite(arr)]
    low = float(np.quantile(finite, lo)) if finite.size else np.nan
    high = float(np.quantile(finite, 1 - hi)) if finite.size else np.nan
    clipped = int(((arr < low) | (arr > high)).sum())
    warnings.warn(
        f"transform='winsorize' clipped {clipped:,} value(s) to "
        f"[{low:g}, {high:g}].",
        UserWarning,
        stacklevel=4,
    )
    return np.clip(arr, low, high), TransformMeta(
        name="winsorize",
        label="winsorized",
        n_clipped=clipped,
        inverse=lambda a: a,
    )


def _boxcox_inverse(lam: float) -> Callable[[np.ndarray], np.ndarray]:
    """Inverse of the Box-Cox transform at ``lam``."""
    from scipy.special import inv_boxcox

    return lambda a, lm=lam: inv_boxcox(a, lm)


def _yeojohnson_inverse(lam: float) -> Callable[[np.ndarray], np.ndarray]:
    """Inverse of the Yeo-Johnson transform at ``lam``.

    scipy ships no inverse, so both branches are written out.
    """

    def inverse(a: np.ndarray, lm: float = lam) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        out = np.empty_like(a)
        pos = a >= 0
        if abs(lm) < 1e-12:
            out[pos] = np.expm1(a[pos])
        else:
            out[pos] = np.power(a[pos] * lm + 1.0, 1.0 / lm) - 1.0
        neg = ~pos
        if abs(lm - 2.0) < 1e-12:
            out[neg] = -np.expm1(-a[neg])
        else:
            base = 1.0 - a[neg] * (2.0 - lm)
            out[neg] = 1.0 - np.power(base, 1.0 / (2.0 - lm))
        return out

    return inverse


def _rescale(arr: np.ndarray, name: str) -> tuple[np.ndarray, TransformMeta]:
    """Centre-and-scale transforms; shape is left unchanged."""
    finite = arr[np.isfinite(arr)]
    if name == "zscore":
        centre = float(np.mean(finite))
        scale = float(np.std(finite, ddof=1)) or 1.0
        label = "z"
    elif name == "robust":
        centre = float(np.median(finite))
        q1, q3 = np.quantile(finite, [0.25, 0.75])
        scale = float(q3 - q1) or 1.0
        label = "robust"
    else:
        centre = float(finite.min())
        scale = float(finite.max() - finite.min()) or 1.0
        label = "minmax"
    out = (arr - centre) / scale
    return out, TransformMeta(
        name=name,
        label=label,
        inverse=lambda a, c=centre, s=scale: a * s + c,
    )


def _compact_formatter(value: float, _pos: int = 0) -> str:
    """Format an axis tick compactly (``1.2M`` rather than offset text)."""
    magnitude = abs(value)
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if magnitude >= limit:
            return f"{value / limit:,.1f}{suffix}"
    if magnitude and magnitude < 0.01:
        return f"{value:.2e}"
    if float(value).is_integer():
        return f"{int(value):,d}"
    return f"{value:,.2f}"


def _format_p(p: float) -> str:
    """Render a p-value, avoiding a misleading ``0.0000``."""
    if not np.isfinite(p):
        return "p = n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.4f}"


def _p_compact(p: float) -> str:
    """Bare p-value keeping the ``<`` so tiny values stay distinct.

    ``_format_p`` output split on whitespace would render both 1e-14 and
    0.0009 as ``0.001``, which reads as "unchanged" in a raw-versus-
    transformed comparison.
    """
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.4g}"


class EDAPlotter:
    """Kind-aware plotting for exploratory data analysis.

    Every method resolves each variable to ``"numeric"``,
    ``"categorical"`` or ``"datetime"`` through :meth:`resolve_kind`,
    then renders it with the shared :class:`StyleConfig` look. Styling
    is applied inside a scoped ``rc_context``, so the caller's global
    matplotlib state is never mutated.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Default frame. Every method also accepts ``df=``; if both are
        ``None`` the method raises.
    cat_max_cardinality : int, default 12
        A numeric column with at most this many distinct values may
        resolve to ``"categorical"`` (rule 4 of :meth:`resolve_kind`).
    low_unique_ratio : float, default 0.05
        Alternative rule-4 trigger: distinct values over row count.
    as_categorical, as_numeric : list of str, optional
        Constructor-level kind overrides, beaten by a method's
        ``treat_as``.
    palette : str, list or dict, default "tab10"
        Fallback palette; the :class:`StyleConfig` palettes take
        precedence for categorical and boolean levels.
    figsize : tuple of float, optional
        Default panel size. ``None`` uses the style token ``(8, 6)``.
    save_dir : str or pathlib.Path, optional
        Root that ``save_as`` paths resolve against.
    show : bool, default False
        Whether methods call ``plt.show()`` by default.
    random_state : int, default 42
        Seeds every bootstrap, jitter, sample and swarm layout.
    lang : {"en", "es"}, default "en"
        Language of generated interpretation text.
    style_overrides : dict, optional
        Per-instance :class:`StyleConfig` field overrides.

    Attributes
    ----------
    DEFAULT_STYLE : StyleConfig
        Class-level defaults; instances copy and override it.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"grade": [1, 2, 2, 3], "score": [9.1, 8.2, 7.3, 6.4]}
    ... )
    >>> plotter = EDAPlotter(df)
    >>> plotter.is_numeric(df["score"])
    True
    """

    DEFAULT_STYLE: StyleConfig = StyleConfig()

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        *,
        cat_max_cardinality: int = 12,
        low_unique_ratio: float = 0.05,
        as_categorical: list[str] | None = None,
        as_numeric: list[str] | None = None,
        palette: str | list | dict = "tab10",
        style: str = "whitegrid",
        context: str = "notebook",
        figsize: tuple[float, float] | None = None,
        dpi: int = 110,
        save_dir: str | Path | None = None,
        show: bool = False,
        random_state: int = 42,
        lang: str = "en",
        min_group_size: int = 1,
        max_levels_warn: int = 20,
        max_levels_error: int = 50,
        style_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.df = df
        self.cat_max_cardinality = int(cat_max_cardinality)
        self.low_unique_ratio = float(low_unique_ratio)
        self.as_categorical = set(as_categorical or ())
        self.as_numeric = set(as_numeric or ())
        self.palette = palette
        self.style_name = style
        self.context = context
        self.dpi = int(dpi)
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.show = bool(show)
        self.random_state = int(random_state)
        self.lang = lang if lang in _STRINGS else "en"
        self.min_group_size = int(min_group_size)
        self.max_levels_warn = int(max_levels_warn)
        self.max_levels_error = int(max_levels_error)

        overrides = dict(style_overrides or {})
        self.style = self.DEFAULT_STYLE.replace(**overrides)
        if figsize is not None:
            self.style = self.style.replace(figsize=tuple(figsize))

        self._color_cache: dict[str, dict[Any, str]] = {}
        self._warned_kind: set[str] = set()
        self._warned_once: set[str] = set()
        self._last_transform: TransformMeta | None = None

    # -- warnings ----------------------------------------------------
    def _warn_once(self, key: str, message: str) -> None:
        """Emit ``message`` at most once per key for this instance."""
        if key in self._warned_once:
            return
        self._warned_once.add(key)
        warnings.warn(message, UserWarning, stacklevel=3)

    # -- kind resolution --------------------------------------------
    def resolve_kind(
        self,
        series: pd.Series | Sequence[Any],
        override: str | None = None,
    ) -> Kind:
        """Classify a variable as numeric, categorical or datetime.

        The backbone of the class: every method routes its behaviour
        through this one resolver. Resolution order is (1) explicit
        override, (2) object/string/category/bool dtypes, (3) datetime,
        period and timedelta dtypes, (4) numeric with low cardinality,
        (5) numeric.

        Rule 4 exists so that ``1/2/3/4/5`` ratings, ``0/1`` flags and
        encoded class labels behave as classes rather than as a
        continuous axis, while ``4.7, 4.71, 4.9`` stays numeric even in
        a tiny sample. It fires when the distinct count is at most
        ``cat_max_cardinality`` **and** the values are integer-like or
        the distinct-to-row ratio is below ``low_unique_ratio``.

        Parameters
        ----------
        series : pandas.Series or sequence
            The variable to classify.
        override : {"numeric", "categorical", "datetime"}, optional
            Per-call override; beats the constructor's
            ``as_categorical`` / ``as_numeric``.

        Returns
        -------
        {"numeric", "categorical", "datetime"}
            The resolved kind.

        Warns
        -----
        UserWarning
            Once per column, when rule 4 reclassifies a numeric column
            as categorical.

        Examples
        --------
        >>> import pandas as pd
        >>> p = EDAPlotter()
        >>> p.resolve_kind(pd.Series([1, 2, 3, 1, 2], name="grade"))
        'categorical'
        """
        series = self._as_series(series)
        name = str(series.name) if series.name is not None else ""

        if override is not None:
            if override not in {"numeric", "categorical", "datetime"}:
                raise ValueError(
                    f"treat_as for {name!r} must be 'numeric', "
                    f"'categorical' or 'datetime', got {override!r}."
                )
            return override  # type: ignore[return-value]
        if name and name in self.as_categorical:
            return "categorical"
        if name and name in self.as_numeric:
            return "numeric"

        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            return "categorical"
        if is_bool_dtype(dtype):
            return "categorical"
        if is_datetime64_any_dtype(dtype) or is_timedelta64_dtype(dtype):
            return "datetime"
        if isinstance(dtype, pd.PeriodDtype):
            return "datetime"
        if not is_numeric_dtype(dtype):
            # str (pandas 3 default for text), object, and anything
            # else non-numeric all group rather than measure.
            return "categorical"

        n_unique = int(series.nunique(dropna=True))
        n_rows = int(len(series))
        if n_unique <= self.cat_max_cardinality:
            ratio = n_unique / n_rows if n_rows else 1.0
            if self._is_integer_like(series) or ratio < self.low_unique_ratio:
                if name and name not in self._warned_kind:
                    self._warned_kind.add(name)
                    warnings.warn(
                        f"Column {name!r} is numeric but has only "
                        f"{n_unique} distinct value(s), so it is treated "
                        f"as categorical. Override with "
                        f"treat_as={{{name!r}: 'numeric'}} or the "
                        f"constructor's as_numeric=[{name!r}].",
                        UserWarning,
                        stacklevel=2,
                    )
                return "categorical"
        return "numeric"

    @staticmethod
    def _is_integer_like(series: pd.Series) -> bool:
        """True when every non-null value is a whole number."""
        if is_integer_dtype(series.dtype):
            return True
        values = series.dropna().to_numpy(dtype=float, na_value=np.nan)
        if values.size == 0:
            return False
        finite = values[np.isfinite(values)]
        return bool(finite.size) and bool(np.all(np.mod(finite, 1) == 0))

    def is_categorical(
        self,
        series: pd.Series | Sequence[Any],
        override: str | None = None,
    ) -> bool:
        """Whether :meth:`resolve_kind` calls this variable categorical.

        Parameters
        ----------
        series : pandas.Series or sequence
            The variable to test.
        override : str, optional
            Per-call kind override.

        Returns
        -------
        bool
            ``True`` for a categorical variable.

        Examples
        --------
        >>> import pandas as pd
        >>> EDAPlotter().is_categorical(pd.Series(["a", "b"]))
        True
        """
        return self.resolve_kind(series, override) == "categorical"

    def is_numeric(
        self,
        series: pd.Series | Sequence[Any],
        override: str | None = None,
    ) -> bool:
        """Whether :meth:`resolve_kind` calls this variable numeric.

        Parameters
        ----------
        series : pandas.Series or sequence
            The variable to test.
        override : str, optional
            Per-call kind override.

        Returns
        -------
        bool
            ``True`` for a continuous numeric variable.

        Examples
        --------
        >>> import pandas as pd
        >>> EDAPlotter().is_numeric(pd.Series([1.5, 2.5, 3.5, 4.25]))
        True
        """
        return self.resolve_kind(series, override) == "numeric"

    # -- frame and column plumbing ----------------------------------
    @staticmethod
    def _as_series(values: Any, name: str | None = None) -> pd.Series:
        """Coerce array-likes to a Series, preserving any name."""
        if isinstance(values, pd.Series):
            return values if name is None else values.rename(name)
        return pd.Series(values, name=name)

    def _frame(self, df: pd.DataFrame | None) -> pd.DataFrame:
        """Return the frame to use, preferring the per-call one."""
        frame = df if df is not None else self.df
        if frame is None:
            raise ValueError(
                "No DataFrame available: pass df= to this method or "
                "construct EDAPlotter(df=...)."
            )
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"df must be a pandas DataFrame, got {type(frame).__name__}."
            )
        return frame

    def _column(
        self,
        df: pd.DataFrame,
        spec: str | pd.Series | Sequence[Any] | None,
        role: str,
    ) -> pd.Series | None:
        """Resolve a column name or array-like into a named Series."""
        if spec is None:
            return None
        if isinstance(spec, str):
            if spec not in df.columns:
                close = get_close_matches(spec, list(df.columns), n=3)
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                raise ValueError(
                    f"{role}={spec!r} is not a column of the frame.{hint}"
                )
            return df[spec]
        series = self._as_series(spec)
        if series.name is None:
            series = series.rename(role)
        return series

    def _require_kind(
        self,
        series: pd.Series,
        expected: Kind,
        role: str,
        treat_as: Mapping[str, str] | None,
    ) -> None:
        """Raise unless ``series`` resolves to ``expected``."""
        name = str(series.name)
        kind = self.resolve_kind(series, (treat_as or {}).get(name))
        if kind != expected:
            raise ValueError(
                f"{role}={name!r} resolves to {kind!r}, but this plot "
                f"needs {expected!r}. Override with "
                f"treat_as={{{name!r}: {expected!r}}}."
            )

    # -- categorical level handling ---------------------------------
    def ordered_levels(
        self,
        series: pd.Series,
        order: Sequence[Any] | None = None,
        sort: str | None = None,
        values: pd.Series | None = None,
        ascending: bool = False,
    ) -> list[Any]:
        """Order the levels of a categorical variable.

        Numeric-coded categoricals are ordered by numeric value, never
        lexicographically, so the axis reads ``1, 2, 10`` rather than
        ``1, 10, 2``. An ordered ``CategoricalDtype`` keeps its own
        order by default.

        Parameters
        ----------
        series : pandas.Series
            The grouping variable.
        order : sequence, optional
            Explicit order; wins over ``sort`` and may be a subset,
            which filters the data.
        sort : {"value", "alpha", "natural", "median", None}, optional
            Ordering rule. ``"value"`` and ``"median"`` require
            ``values``.
        values : pandas.Series, optional
            Numeric companion used by ``"value"`` and ``"median"``.
        ascending : bool, default False
            Direction for statistic-based ordering.

        Returns
        -------
        list
            Levels in plotting order.

        Examples
        --------
        >>> import pandas as pd
        >>> lv = EDAPlotter().ordered_levels(pd.Series([10, 2, 1]))
        >>> [int(v) for v in lv]
        [1, 2, 10]
        """
        if order is not None:
            return list(order)

        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype) and dtype.ordered:
            present = set(series.dropna().unique())
            return [c for c in dtype.categories if c in present]

        levels = list(pd.unique(series.dropna()))
        numeric_coded = is_numeric_dtype(series.dtype) or is_bool_dtype(
            series.dtype
        )

        if sort in {"value", "median"} and values is not None:
            stat = "median" if sort == "median" else "mean"
            grouped = values.groupby(series, observed=True).agg(stat)
            return list(grouped.sort_values(ascending=ascending).index)
        if sort == "alpha":
            return sorted(levels, key=str)
        if sort == "natural" or (sort is None and numeric_coded):
            try:
                return sorted(levels)
            except TypeError:
                return sorted(levels, key=str)
        if sort is None:
            return levels
        return sorted(levels, key=str)

    def _limit_levels(
        self,
        series: pd.Series,
        top_n: int | None = None,
        other_label: str = "Other",
    ) -> pd.Series:
        """Collapse rare levels of a grouping variable.

        Parameters
        ----------
        series : pandas.Series
            The grouping variable.
        top_n : int, optional
            Keep the ``top_n`` most frequent levels and collapse the
            rest into ``other_label``. ``None`` keeps every level.
        other_label : str, default "Other"
            Name of the collapsed bucket, always sorted last.

        Returns
        -------
        pandas.Series
            The grouping variable, possibly collapsed.

        Raises
        ------
        ValueError
            When the cardinality exceeds ``max_levels_error``.

        Warns
        -----
        UserWarning
            When the cardinality exceeds ``max_levels_warn`` and
            ``top_n`` is ``None``.

        Examples
        --------
        >>> import pandas as pd
        >>> s = pd.Series(list("aaabbc"), name="g")
        >>> sorted(EDAPlotter()._limit_levels(s, top_n=2).unique())
        ['Other', 'a', 'b']
        """
        name = str(series.name)
        n_levels = int(series.nunique(dropna=True))
        if top_n is None:
            if n_levels > self.max_levels_error:
                raise ValueError(
                    f"{name!r} has {n_levels} levels, above the "
                    f"max_levels_error limit of "
                    f"{self.max_levels_error}. Pass top_n=... to keep "
                    f"only the most frequent levels."
                )
            if n_levels > self.max_levels_warn:
                self._warn_once(
                    f"levels:{name}",
                    f"{name!r} has {n_levels} levels, which will be hard "
                    f"to read. Consider top_n=... to keep only the most "
                    f"frequent ones.",
                )
            return series
        if n_levels <= top_n:
            return series
        keep = list(series.value_counts().head(top_n).index)
        self._warn_once(
            f"top_n:{name}",
            f"{name!r} has {n_levels} levels; keeping the top {top_n} "
            f"and collapsing {n_levels - top_n} into {other_label!r}.",
        )
        return series.where(series.isin(keep), other_label)

    # -- colours -----------------------------------------------------
    def reset_color_cache(self) -> None:
        """Forget every cached level-to-colour mapping.

        Examples
        --------
        >>> p = EDAPlotter()
        >>> p.reset_color_cache()
        """
        self._color_cache.clear()

    def set_color_map(self, column: str, mapping: Mapping[Any, str]) -> None:
        """Pin the colours used for one column's levels.

        Parameters
        ----------
        column : str
            Column whose levels are being pinned.
        mapping : mapping
            Level to matplotlib colour.

        Examples
        --------
        >>> p = EDAPlotter()
        >>> p.set_color_map("flag", {True: "#5dade2", False: "#95a5a6"})
        >>> p.color_map("flag", [True])[True]
        '#5dade2'
        """
        self._color_cache[column] = dict(mapping)

    @staticmethod
    def _looks_boolean(levels: Sequence[Any]) -> bool:
        """Whether two levels read as a false/true pair."""
        if len(levels) != 2:
            return False
        tokens = frozenset(str(v).strip().lower() for v in levels)
        return tokens in _BOOLEAN_LEVEL_SETS

    def color_map(
        self,
        column: str,
        levels: Sequence[Any],
        palette: str | list | dict | None = None,
    ) -> dict[Any, str]:
        """Stable level-to-colour mapping for one column.

        The mapping is cached on the instance, so a level keeps the same
        colour across every figure this plotter produces. Two levels
        that read as a false/true pair automatically get the flat
        two-colour ``binary`` palette.

        Parameters
        ----------
        column : str
            Column name, used as the cache key.
        levels : sequence
            Levels to assign colours to, in plotting order.
        palette : str, list or dict, optional
            Explicit palette overriding the style defaults.

        Returns
        -------
        dict
            Level to colour.

        Examples
        --------
        >>> EDAPlotter().color_map("flag", [False, True])[True]
        '#5dade2'
        """
        cached = self._color_cache.setdefault(column, {})
        if palette is None and self._looks_boolean(levels):
            for level in levels:
                token = str(level).strip().lower()
                cached.setdefault(
                    level,
                    self.style.binary_false
                    if token in _FALSEY_TOKENS
                    else self.style.binary_true,
                )
            return {level: cached[level] for level in levels}

        if isinstance(palette, Mapping):
            for level in levels:
                if level in palette:
                    cached[level] = palette[level]
        missing = [lv for lv in levels if lv not in cached]
        if missing:
            colors = self._palette_colors(len(cached) + len(missing), palette)
            for index, level in enumerate(missing, start=len(cached)):
                cached[level] = colors[index % len(colors)]
        return {level: cached[level] for level in levels}

    def _palette_colors(
        self, n: int, palette: str | list | dict | None
    ) -> list[str]:
        """Produce ``n`` colours from the configured palette."""
        if isinstance(palette, (list, tuple)):
            base = list(palette)
        elif isinstance(palette, str):
            base = [
                mpl.colors.to_hex(c) for c in sns.color_palette(palette, n)
            ]
        elif self.style.enabled:
            base = list(self.style.categorical)
        else:
            base = [
                mpl.colors.to_hex(c)
                for c in sns.color_palette(self.palette, n)
            ]
        if n > len(base):
            self._warn_once(
                f"palette-cycle:{len(base)}",
                f"{n} levels exceed the {len(base)}-colour categorical "
                f"palette; colours will repeat. Consider top_n=... or "
                f"faceting.",
            )
        return base

    # -- style scope and figure construction ------------------------
    def _resolve_style(
        self, style: Mapping[str, Any] | str | None
    ) -> StyleConfig:
        """Merge a per-call style override onto the instance style."""
        if style is None:
            return self.style
        if isinstance(style, str):
            if style == "none":
                return self.style.replace(enabled=False)
            raise ValueError(
                f"style={style!r} is not recognised; pass 'none' to "
                f"disable styling or a mapping of StyleConfig fields."
            )
        return self.style.replace(**dict(style))

    def _style_scope(self, style: StyleConfig) -> Any:
        """Context manager applying the style without global mutation."""
        return mpl.rc_context(style.rc_params())

    def _new_figure(
        self,
        style: StyleConfig,
        figsize: tuple[float, float] | None,
        nrows: int = 1,
        ncols: int = 1,
        **kwargs: Any,
    ) -> tuple[Figure, np.ndarray]:
        """Create a figure and its axes grid at the configured dpi."""
        size = figsize or style.figsize
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=size,
            dpi=self.dpi,
            squeeze=False,
            **kwargs,
        )
        return fig, axes

    # -- three-tier text hierarchy ----------------------------------
    def _wrap(self, text: str, fig: Figure, size: float) -> list[str]:
        """Wrap ``text`` to the figure width at the given font size."""
        width_pt = fig.get_figwidth() * 72.0
        # 0.55 em is a reasonable mean advance width for sans-serif.
        chars = max(20, int(width_pt / (size * 0.55)))
        return textwrap.wrap(text, width=chars) or [""]

    def _place_header(
        self,
        fig: Figure,
        style: StyleConfig,
        title: str | None,
        subtitle: str | None,
        margins: Mapping[str, float],
    ) -> float:
        """Draw the figure title and subtitle, return the free top.

        Positions are computed in points and converted to figure
        coordinates, so the header holds its spacing on any figure size
        rather than sitting at a hard-coded fraction. When there is no
        subtitle the title moves down and the reserved band shrinks.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to draw on.
        style : StyleConfig
            Active style tokens.
        title, subtitle : str or None
            Header text; ``None`` omits that tier.
        margins : mapping
            Requested ``subplots_adjust`` margins.

        Returns
        -------
        float
            The ``top`` margin the axes may occupy.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure(figsize=(8, 6))
        >>> p = EDAPlotter()
        >>> round(
        ...     p._place_header(fig, p.style, "T", "s", p.style.margins), 2
        ... ) <= 0.9
        True
        >>> plt.close(fig)
        """
        top = float(margins.get("top", style.margins["top"]))
        if not title and not subtitle:
            return top

        height_pt = fig.get_figheight() * 72.0
        x = max(0.02, float(margins.get("left", 0.15)) - 0.05)
        pad_pt = 14.0 if subtitle else 20.0
        gap_pt = 6.0

        cursor = 1.0 - pad_pt / height_pt
        if title:
            lines = self._wrap(title, fig, style.title_size)
            fig.text(
                x,
                cursor,
                "\n".join(lines),
                ha="left",
                va="top",
                fontsize=style.title_size,
                fontweight="bold",
                color=style.title_color,
            )
            used = style.title_size * 1.2 * len(lines)
            cursor -= (used + gap_pt) / height_pt
        if subtitle:
            lines = self._wrap(subtitle, fig, style.subtitle_size)
            fig.text(
                x,
                cursor,
                "\n".join(lines),
                ha="left",
                va="top",
                fontsize=style.subtitle_size,
                color=style.subtitle_color,
            )
            used = style.subtitle_size * 1.2 * len(lines)
            cursor -= used / height_pt

        free = cursor - gap_pt * 1.6 / height_pt
        return float(min(top, max(0.35, free)))

    def _panel_title(
        self, ax: Axes, style: StyleConfig, text: str | None
    ) -> None:
        """Set the third-tier per-axes title."""
        if not text:
            return
        ax.set_title(
            text,
            loc=style.panel_title_loc,
            fontsize=style.panel_title_size,
            fontweight="bold",
            pad=style.panel_title_pad,
            color=style.panel_title_color,
        )

    # -- axis furniture ---------------------------------------------
    def _apply_grid(
        self, ax: Axes, style: StyleConfig, axis: str | None
    ) -> None:
        """Draw the value-axis grid beneath the data."""
        if not style.enabled or axis is None:
            return
        ax.set_axisbelow(True)
        ax.grid(
            axis=axis,
            color=style.grid_color,
            linestyle="-",
            linewidth=style.grid_linewidth,
            zorder=style.grid_zorder,
        )

    def _despine(
        self, ax: Axes, style: StyleConfig, keep_spines: bool = False
    ) -> None:
        """Flatten the frame and drop orphaned tick marks."""
        if not style.enabled or keep_spines:
            return
        sns.despine(ax=ax, left=True, bottom=True)
        ax.tick_params(length=0)

    def _format_value_axis(
        self,
        ax: Axes,
        which: str,
        style: StyleConfig,
        percent: bool = False,
    ) -> None:
        """Apply thousands separators or a percent formatter."""
        if not style.enabled:
            return
        axis = ax.yaxis if which == "y" else ax.xaxis
        if percent:
            axis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}%"))
        else:
            axis.set_major_formatter(FuncFormatter(_compact_formatter))

    def _rotate_ticks(
        self,
        ax: Axes,
        labels: Sequence[Any],
        rotate: int | str | None,
        axis: str = "x",
    ) -> None:
        """Rotate long or numerous categorical tick labels."""
        if rotate is None:
            return
        if rotate == "auto":
            texts = [str(v) for v in labels]
            long_label = any(len(t) > 10 for t in texts)
            if not (long_label or len(texts) > 6):
                return
            angle = 45
        else:
            angle = int(rotate)
        if not angle:
            return
        target = (
            ax.get_xticklabels() if axis == "x" else (ax.get_yticklabels())
        )
        for label in target:
            label.set_rotation(angle)
            label.set_horizontalalignment("right")

    def _relabel_ticks(
        self,
        ax: Axes,
        labels: Mapping[Any, str] | Sequence[str] | None,
        levels: Sequence[Any],
        axis: str = "x",
    ) -> None:
        """Apply ``xticklabels`` / ``yticklabels`` safely.

        matplotlib warns when ``set_*ticklabels`` runs without a fixed
        locator, so the ticks are pinned first. The mapping form is
        preferred because a positional list silently mislabels when the
        category order changes.
        """
        if labels is None:
            return
        if isinstance(labels, Mapping):
            text = [str(labels.get(level, level)) for level in levels]
        else:
            text = [str(v) for v in labels]
            if len(text) != len(levels):
                raise ValueError(
                    f"{axis}ticklabels has {len(text)} entries but there "
                    f"are {len(levels)} levels. Pass a "
                    f"{{old: new}} mapping to avoid positional "
                    f"mislabelling."
                )
        ticks = list(range(len(levels)))
        if axis == "x":
            ax.set_xticks(ticks)
            ax.set_xticklabels(text)
        else:
            ax.set_yticks(ticks)
            ax.set_yticklabels(text)

    def _style_legend(
        self,
        ax: Axes,
        style: StyleConfig,
        title: str | None,
        legend: bool,
        loc: str = "best",
    ) -> None:
        """Apply the frameless legend rules, moving it out if crowded."""
        handles, labels = ax.get_legend_handles_labels()
        existing = ax.get_legend()
        if not legend or not handles:
            if existing is not None:
                existing.remove()
            return
        outside = len(labels) > 4
        kwargs: dict[str, Any] = {"frameon": False, "title": title}
        if outside and loc == "best":
            kwargs["loc"] = "upper left"
            kwargs["bbox_to_anchor"] = (1.02, 1.0)
        else:
            kwargs["loc"] = loc
        drawn = ax.legend(handles, labels, **kwargs)
        if drawn.get_title() is not None:
            drawn.get_title().set_fontweight("bold")
            drawn.get_title().set_fontsize(style.tick_size)

    def _axis_note(
        self, ax: Axes, style: StyleConfig, text: str, which: str = "x"
    ) -> None:
        """Write a small note next to an axis (log-scale marker etc.)."""
        if not text:
            return
        xy = (1.0, -0.14) if which == "x" else (-0.12, 1.02)
        ha = "right" if which == "x" else "left"
        ax.annotate(
            text,
            xy=xy,
            xycoords="axes fraction",
            ha=ha,
            va="top",
            fontsize=style.annot_size * 0.9,
            color=style.subtitle_color,
        )

    # -- log axis ----------------------------------------------------
    def _normalise_log_scale(
        self,
        log_scale: Any,
        value_axis: str,
        log_x: bool = False,
        log_y: bool = False,
    ) -> dict[str, float | str]:
        """Turn every accepted ``log_scale`` form into an axis->base map.

        ``log_x`` / ``log_y`` are accepted aliases mapping onto
        ``{"x": 10}`` / ``{"y": 10}``.
        """
        spec: dict[str, float | str] = {}
        if log_x:
            spec["x"] = 10.0
        if log_y:
            spec["y"] = 10.0
        if log_scale is None or log_scale is False:
            return spec
        if log_scale is True:
            spec[value_axis] = 10.0
        elif isinstance(log_scale, str):
            if log_scale == "symlog":
                spec[value_axis] = "symlog"
            elif log_scale in {"x", "y"}:
                spec[log_scale] = 10.0
            elif log_scale == "both":
                spec["x"] = 10.0
                spec["y"] = 10.0
            else:
                raise ValueError(
                    f"log_scale={log_scale!r} is not recognised. Use "
                    f"True, 'x', 'y', 'both', 'symlog', a base, or a "
                    f"{{axis: base}} mapping."
                )
        elif isinstance(log_scale, Mapping):
            spec.update({str(k): v for k, v in log_scale.items()})
        elif isinstance(log_scale, (int, float)):
            spec[value_axis] = float(log_scale)
        else:
            raise ValueError(
                f"log_scale={log_scale!r} is not a recognised form."
            )
        return spec

    def _apply_log_axis(
        self,
        ax: Axes,
        spec: Mapping[str, float | str],
        style: StyleConfig,
        values: Mapping[str, np.ndarray] | None = None,
        column: Mapping[str, str] | None = None,
        nonpositive: Nonpositive = "raise",
    ) -> None:
        """Set log axes, guarding the domain the same way transforms do.

        A log axis makes matplotlib silently drop non-positive points,
        so the same ``nonpositive`` guard runs here and reports a count
        rather than letting rows vanish unannounced.
        """
        for axis_name, base in spec.items():
            data = (values or {}).get(axis_name)
            name = (column or {}).get(axis_name, axis_name)
            if base == "symlog":
                thresh = self._linthresh(data)
                setter = ax.set_xscale if axis_name == "x" else (ax.set_yscale)
                setter("symlog", linthresh=thresh)
                self._axis_note(
                    ax,
                    style,
                    _t(self.lang, "symlog_note", base=10, thr=thresh),
                    axis_name,
                )
                continue
            if data is not None:
                _guard_domain(
                    np.asarray(data, dtype=float),
                    "positive",
                    nonpositive,
                    column=name,
                    operation=f"log_scale on axis {axis_name!r}",
                )
            setter = ax.set_xscale if axis_name == "x" else ax.set_yscale
            setter("log", base=float(base))
            self._axis_note(
                ax,
                style,
                _t(self.lang, "log_scale_note", base=f"{float(base):g}"),
                axis_name,
            )

    @staticmethod
    def _linthresh(data: np.ndarray | None) -> float:
        """Smallest non-zero magnitude, rounded down to a power of ten."""
        if data is None:
            return 1.0
        finite = np.abs(np.asarray(data, dtype=float))
        finite = finite[np.isfinite(finite) & (finite > 0)]
        if not finite.size:
            return 1.0
        return float(10 ** np.floor(np.log10(finite.min())))

    def _check_scale_conflict(
        self,
        log_spec: Mapping[str, Any],
        transform_spec: Any,
        axis: str,
        column: str,
    ) -> None:
        """Refuse a log axis stacked on top of a log transform."""
        if axis in log_spec and transform_spec not in (None, "identity"):
            raise ValueError(
                f"Both log_scale and transform={transform_spec!r} target "
                f"the {axis!r} axis for {column!r}, which would apply a "
                f"double log. Use one or the other: log_scale rescales "
                f"the axis and keeps statistics in raw units, transform "
                f"changes the data and the statistics with it."
            )

    # -- finishing ---------------------------------------------------
    def _save(self, fig: Figure, save_as: str | Path | None) -> Path | None:
        """Write the figure out, creating parent directories."""
        if save_as is None:
            return None
        path = Path(save_as)
        if not path.is_absolute() and self.save_dir is not None:
            path = self.save_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path,
            dpi=self.dpi,
            facecolor="white",
            bbox_inches="tight",
        )
        return path

    def _finish(
        self,
        fig: Figure,
        style: StyleConfig,
        margins: Mapping[str, float],
        top: float,
        save_as: str | Path | None,
        show: bool | None,
    ) -> None:
        """Reserve the header band, then save and optionally show.

        ``tight_layout`` is deliberately not called here: it would
        reflow the axes on top of the ``fig.text`` header. The top
        margin computed by :meth:`_place_header` is applied instead.
        """
        if style.enabled:
            adjust = dict(style.margins)
            adjust.update(margins)
            adjust["top"] = top
            fig.subplots_adjust(**adjust)
        self._save(fig, save_as)
        if self.show if show is None else show:
            plt.show()

    # -- shared signature -------------------------------------------
    def _resolve_common(
        self,
        df: pd.DataFrame | None,
        *,
        style: Mapping[str, Any] | str | None = None,
        figsize: tuple[float, float] | None = None,
        treat_as: Mapping[str, str] | None = None,
        ax: Axes | None = None,
        facet: Any = None,
        ax_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, StyleConfig, dict[str, str], dict[str, Any]]:
        """Parse the parameters every plotting method shares.

        Parameters
        ----------
        df : pandas.DataFrame or None
            Per-call frame, falling back to ``self.df``.
        style : mapping or str, optional
            Per-call style override, or ``"none"`` to disable styling.
        figsize : tuple of float, optional
            Per-call panel size.
        treat_as : mapping, optional
            Per-call kind overrides.
        ax : matplotlib.axes.Axes, optional
            Target axes; mutually exclusive with ``facet``.
        facet : optional
            Faceting variable.
        ax_kwargs : mapping, optional
            Extra keyword arguments applied via ``Axes.set``.

        Returns
        -------
        frame, style, treat_as, ax_kwargs
            Normalised values for the calling method.

        Raises
        ------
        ValueError
            When both ``ax`` and ``facet`` are supplied.

        Examples
        --------
        >>> import pandas as pd
        >>> p = EDAPlotter(pd.DataFrame({"a": [1.0, 2.0]}))
        >>> frame, _, _, _ = p._resolve_common(None)
        >>> list(frame.columns)
        ['a']
        """
        if ax is not None and facet is not None:
            raise ValueError(
                "ax and facet are mutually exclusive: faceting creates "
                "its own grid of axes, so it cannot draw into one you "
                "supply."
            )
        frame = self._frame(df)
        resolved = self._resolve_style(style)
        if figsize is not None:
            resolved = resolved.replace(figsize=tuple(figsize))
        return frame, resolved, dict(treat_as or {}), dict(ax_kwargs or {})

    def _dropna_subset(
        self,
        df: pd.DataFrame,
        columns: Sequence[str],
        dropna: bool,
        label: str,
    ) -> pd.DataFrame:
        """Drop rows null in the columns this plot actually uses."""
        used = [c for c in columns if c and c in df.columns]
        if not used or not dropna:
            return df
        before = len(df)
        out = df.dropna(subset=used)
        removed = before - len(out)
        if removed and before and removed / before > 0.01:
            warnings.warn(
                f"{label}: dropped {removed:,} row(s) "
                f"({removed / before:.1%}) with nulls in "
                f"{', '.join(used)}; n = {len(out):,} remain.",
                UserWarning,
                stacklevel=3,
            )
        return out

    def _filter_small_groups(
        self, df: pd.DataFrame, column: str | None, label: str
    ) -> pd.DataFrame:
        """Drop groups smaller than ``min_group_size``."""
        if column is None or self.min_group_size <= 1:
            return df
        counts = df[column].value_counts()
        small = counts[counts < self.min_group_size]
        if small.empty:
            return df
        warnings.warn(
            f"{label}: dropped {len(small)} group(s) of {column!r} with "
            f"fewer than {self.min_group_size} rows "
            f"({', '.join(map(str, small.index[:5]))}).",
            UserWarning,
            stacklevel=3,
        )
        return df[~df[column].isin(small.index)]

    # -- faceting ----------------------------------------------------
    def _facet(
        self,
        df: pd.DataFrame,
        facet: str,
        draw: Callable[[Axes, pd.DataFrame, Any], None],
        *,
        style: StyleConfig,
        col_wrap: int = 3,
        sharex: bool = True,
        sharey: bool = True,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        subtitle: str | None = None,
        top_n: int | None = None,
        panel_title_fmt: str = "{level}",
        min_rows: int = 0,
        save_as: str | Path | None = None,
        show: bool | None = None,
    ) -> Figure:
        """Lay out one panel per level of ``facet``.

        Written once and delegated to by every method, so grid
        construction, blanking of unused cells and header placement are
        never reimplemented.

        Parameters
        ----------
        df : pandas.DataFrame
            Data to split.
        facet : str
            Categorical column defining the panels.
        draw : callable
            ``draw(ax, sub_df, level)``, called once per panel.
        style : StyleConfig
            Active style tokens.
        col_wrap : int, default 3
            Panels per row.
        sharex, sharey : bool, default True
            Whether panels share scales.
        figsize : tuple of float, optional
            Overall figure size; scaled from the grid when ``None``.
        title, subtitle : str, optional
            Figure-level header.
        top_n : int, optional
            Level cap, applied through :meth:`_limit_levels`.

        Returns
        -------
        matplotlib.figure.Figure
            The faceted figure.

        Examples
        --------
        >>> import matplotlib, pandas as pd
        >>> matplotlib.use("Agg")
        >>> df = pd.DataFrame({"g": ["a", "b"], "v": [1.0, 2.0]})
        >>> fig = EDAPlotter()._facet(
        ...     df,
        ...     "g",
        ...     lambda ax, d, lv: ax.plot(d["v"]),
        ...     style=EDAPlotter().style,
        ... )
        >>> len(fig.axes) >= 2
        True
        """
        series = self._limit_levels(df[facet], top_n)
        df = df.assign(**{facet: series})
        levels = self.ordered_levels(series)
        if min_rows > 0:
            counts = series.value_counts()
            too_small = [
                lv for lv in levels if int(counts.get(lv, 0)) < min_rows
            ]
            if too_small:
                warnings.warn(
                    f"facet={facet!r}: skipped {len(too_small)} panel(s) "
                    f"with fewer than {min_rows} rows "
                    f"({', '.join(map(str, too_small[:5]))}).",
                    UserWarning,
                    stacklevel=3,
                )
                levels = [lv for lv in levels if lv not in too_small]
        n = len(levels)
        if n == 0:
            raise ValueError(
                f"facet={facet!r} has no level with at least "
                f"{max(min_rows, 1)} row(s) to plot."
            )
        ncols = min(col_wrap, n)
        nrows = int(np.ceil(n / ncols))
        size = figsize or (
            min(6.0 * ncols, 22.0),
            max(3.6 * nrows, 3.6),
        )
        fig, axes = self._new_figure(
            style, size, nrows, ncols, sharex=sharex, sharey=sharey
        )
        flat = axes.ravel()
        for index, level in enumerate(levels):
            ax = flat[index]
            subset = df[df[facet] == level]
            draw(ax, subset, level)
            self._panel_title(ax, style, panel_title_fmt.format(level=level))
        for spare in flat[n:]:
            spare.set_visible(False)
        top = self._place_header(fig, style, title, subtitle, style.margins)
        self._finish(fig, style, {}, top, save_as, show)
        return fig

    # -- binning -----------------------------------------------------
    def _bin_edges(
        self,
        values: np.ndarray,
        bins: Any,
        binwidth: float | None,
        binrange: tuple[float, float] | None,
        discrete: bool,
        log_base: float | None,
    ) -> np.ndarray:
        """Compute bin edges once, on pooled data.

        With a log axis the edges are **geometric** (equally spaced in
        log space). Equal-width bins drawn on a log axis render at
        varying widths and misrepresent density, so rescaling the axis
        alone is not enough.
        """
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("No finite values left to bin.")
        lo, hi = (
            binrange
            if binrange is not None
            else (float(finite.min()), float(finite.max()))
        )
        if hi <= lo:
            hi = lo + 1.0
        if discrete:
            return np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1.0)
        if log_base:
            positive = finite[finite > 0]
            if positive.size == 0:
                raise ValueError(
                    "A log-scaled histogram needs positive values."
                )
            lo = max(lo, float(positive.min()))
            count = bins if isinstance(bins, int) else 30
            return np.logspace(np.log10(lo), np.log10(hi), int(count) + 1)
        if binwidth:
            return np.arange(lo, hi + binwidth, binwidth)
        if isinstance(bins, (Sequence, np.ndarray)) and not isinstance(
            bins, str
        ):
            return np.asarray(bins, dtype=float)
        if isinstance(bins, int):
            return np.linspace(lo, hi, bins + 1)
        return np.histogram_bin_edges(finite, bins=str(bins), range=(lo, hi))

    # -- histplot ----------------------------------------------------
    def histplot(
        self,
        df: pd.DataFrame | None = None,
        x: str | pd.Series | None = None,
        y: str | pd.Series | None = None,
        hue: str | pd.Series | None = None,
        *,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | bool | None = None,
        subtitle: str | None = None,
        panel_title: str | bool | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        palette: str | list | dict | None = None,
        legend: bool = True,
        legend_loc: str = "best",
        order: Sequence[Any] | None = None,
        hue_order: Sequence[Any] | None = None,
        top_n: int | None = None,
        log_scale: Any = None,
        log_x: bool = False,
        log_y: bool = False,
        transform: Any = None,
        back_transform: bool = False,
        nonpositive: Nonpositive = "raise",
        facet: str | None = None,
        facet_col_wrap: int = 3,
        sharex: bool = True,
        sharey: bool = True,
        dropna: bool = True,
        treat_as: Mapping[str, str] | None = None,
        bins: Any = "auto",
        binwidth: float | None = None,
        binrange: tuple[float, float] | None = None,
        stat: str = "count",
        common_norm: bool = False,
        kde: bool = False,
        kde_kwargs: Mapping[str, Any] | None = None,
        cumulative: bool = False,
        multiple: str = "layer",
        element: str = "bars",
        rug: bool = False,
        reference_lines: Any = None,
        alpha: float | None = None,
        discrete: bool | None = None,
        show_n: bool = False,
        rotate_xticks: int | str | None = "auto",
        xticklabels: Mapping[Any, str] | Sequence[str] | None = None,
        yticklabels: Mapping[Any, str] | Sequence[str] | None = None,
        style: Mapping[str, Any] | str | None = None,
        save_as: str | Path | None = None,
        show: bool | None = None,
        ax_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Axes | Figure:
        """Plot the distribution of one variable.

        Throwing any column at this method produces something sensible:
        a numeric column gives a histogram, a categorical one (including
        a numeric column with few distinct values, per
        :meth:`resolve_kind`) falls back to one bar per level rather
        than a broken histogram, and a datetime column bins on a time
        frequency.

        When ``hue`` is set the bin edges are computed **once on the
        pooled data**, so every group shares identical bins and the
        groups stay comparable.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Frame to plot; falls back to the instance frame.
        x, y : str or pandas.Series, optional
            The variable to bin. Supply exactly one; ``y`` draws a
            horizontal histogram.
        hue : str or pandas.Series, optional
            Categorical split. Resolved through :meth:`resolve_kind`, so
            a low-cardinality numeric column works as classes.
        bins : int, sequence or str, default "auto"
            Bin count, explicit edges, or a rule name accepted by
            :func:`numpy.histogram_bin_edges` (``"fd"``, ``"sturges"``,
            ``"scott"``, ``"sqrt"``, ``"doane"``).
        binwidth : float, optional
            Overrides ``bins``.
        binrange : tuple of float, optional
            Clip the binned domain.
        stat : {"count", "frequency", "probability", "percent", \
"density"}, default "count"
            What the bar heights measure.
        common_norm : bool, default False
            Normalise over the pooled data rather than per group. The
            default is per group so that groups of different sizes stay
            visually comparable.
        multiple : {"layer", "dodge", "stack", "fill", "facet"}, \
default "layer"
            How ``hue`` groups are combined. ``"facet"`` delegates to
            the shared faceting machinery.
        element : {"bars", "step", "poly"}, default "bars"
            Bar rendering. ``"step"`` is the readable choice for four or
            more overlaid groups.
        kde : bool, default False
            Overlay a Gaussian KDE per group, scaled to ``stat``.
        rug : bool, default False
            Add a rug; auto-disabled above n = 5000.
        reference_lines : str, sequence, mapping or float, optional
            Any of ``"mean"``, ``"median"``, ``"mode"``, or explicit
            values, drawn per group in the group colour.
        discrete : bool, optional
            One bin per integer value, edges offset by 0.5.
        log_scale : bool, str, float or mapping, optional
            Rescale the axis; bin edges become **geometric** so the bars
            keep equal visual width. Statistics stay in raw units.
        transform : str or callable, optional
            Transform the data before binning, so the statistics are
            computed in the transformed space.
        show_n : bool, default False
            Annotate the sample size.

        Returns
        -------
        matplotlib.axes.Axes or matplotlib.figure.Figure
            The axes for a single panel, the figure when faceting.

        Raises
        ------
        ValueError
            When both or neither of ``x`` and ``y`` are given, or when
            ``transform`` and ``log_scale`` target the same axis.

        Examples
        --------
        >>> import matplotlib, pandas as pd
        >>> matplotlib.use("Agg")
        >>> df = pd.DataFrame({"v": [1.0, 2.0, 2.5, 3.0, 9.0]})
        >>> ax = EDAPlotter(df).histplot(x="v")
        >>> ax.get_xlabel()
        'v'
        """
        frame, style_cfg, treat, extra_ax = self._resolve_common(
            df,
            style=style,
            figsize=figsize,
            treat_as=treat_as,
            ax=ax,
            facet=facet,
            ax_kwargs=ax_kwargs,
        )
        if (x is None) == (y is None):
            raise ValueError(
                "histplot needs exactly one of x= or y=; pass x for a "
                "vertical histogram or y for a horizontal one."
            )
        orient_axis = "x" if x is not None else "y"
        target = self._column(frame, x if x is not None else y, orient_axis)
        assert target is not None
        name = str(target.name)
        hue_series = self._column(frame, hue, "hue")
        hue_name = str(hue_series.name) if hue_series is not None else None

        work = frame.copy()
        work[name] = target.to_numpy()
        if hue_series is not None:
            work[hue_name] = hue_series.to_numpy()
        work = self._dropna_subset(
            work,
            [name, hue_name, facet],
            dropna,
            f"histplot({name!r})",
        )
        if hue_name is not None:
            work[hue_name] = self._limit_levels(work[hue_name], top_n)

        kind = self.resolve_kind(work[name], treat.get(name))
        if kind == "categorical":
            self._warn_once(
                f"hist-cat:{name}",
                f"{name!r} resolves to categorical, so histplot is "
                f"drawing one bar per level instead of a histogram. "
                f"Override with treat_as={{{name!r}: 'numeric'}}.",
            )
        if multiple == "facet" and hue_name is not None and facet is None:
            facet, multiple = hue_name, "layer"
            hue_name, hue_series = None, None

        if facet is not None:
            return self._facet(
                work,
                facet,
                lambda panel_ax, sub, level: self.histplot(
                    sub,
                    x=name if orient_axis == "x" else None,
                    y=None if orient_axis == "x" else name,
                    hue=hue_name,
                    ax=panel_ax,
                    bins=bins,
                    binwidth=binwidth,
                    binrange=binrange or self._pooled_range(work[name]),
                    stat=stat,
                    kde=kde,
                    element=element,
                    multiple=multiple,
                    discrete=discrete,
                    legend=legend,
                    treat_as=treat,
                    title=False,
                    panel_title=False,
                    style=style,
                    **kwargs,
                ),
                style=style_cfg,
                col_wrap=facet_col_wrap,
                sharex=sharex,
                sharey=sharey,
                figsize=figsize,
                title=self._auto_title(title, f"Distribution of {name}"),
                subtitle=subtitle,
                save_as=save_as,
                show=show,
            )

        owns_figure = ax is None
        with self._style_scope(style_cfg):
            if owns_figure:
                fig, axes = self._new_figure(style_cfg, figsize)
                ax = axes[0, 0]
            else:
                fig = ax.figure
            drawn = self._draw_hist(
                ax=ax,
                data=work,
                name=name,
                hue_name=hue_name,
                kind=kind,
                orient_axis=orient_axis,
                style=style_cfg,
                treat=treat,
                bins=bins,
                binwidth=binwidth,
                binrange=binrange,
                stat=stat,
                common_norm=common_norm,
                kde=kde,
                kde_kwargs=kde_kwargs,
                cumulative=cumulative,
                multiple=multiple,
                element=element,
                rug=rug,
                reference_lines=reference_lines,
                alpha=alpha,
                discrete=discrete,
                palette=palette,
                hue_order=hue_order,
                order=order,
                log_scale=log_scale,
                log_x=log_x,
                log_y=log_y,
                transform=transform,
                nonpositive=nonpositive,
                show_n=show_n,
                xlabel=xlabel,
                ylabel=ylabel,
                legend=legend,
                legend_loc=legend_loc,
                rotate_xticks=rotate_xticks,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                kwargs=kwargs,
            )
            self._panel_title(
                ax,
                style_cfg,
                self._auto_title(
                    panel_title,
                    None if owns_figure else f"Distribution of {name}",
                ),
            )
            if extra_ax:
                ax.set(**extra_ax)
            if owns_figure:
                top = self._place_header(
                    fig,
                    style_cfg,
                    self._auto_title(title, f"Distribution of {name}"),
                    subtitle,
                    style_cfg.margins,
                )
                self._finish(fig, style_cfg, {}, top, save_as, show)
        return drawn

    @staticmethod
    def _pooled_range(series: pd.Series) -> tuple[float, float] | None:
        """Domain of a numeric series, for shared bins across facets."""
        values = pd.to_numeric(series, errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        return float(finite.min()), float(finite.max())

    @staticmethod
    def _auto_title(
        title: str | bool | None, default: str | None
    ) -> str | None:
        """Resolve the ``title=False`` disable convention."""
        if title is False:
            return None
        if title is None or title is True:
            return default
        return str(title)

    def _draw_hist(
        self,
        *,
        ax: Axes,
        data: pd.DataFrame,
        name: str,
        hue_name: str | None,
        kind: Kind,
        orient_axis: str,
        style: StyleConfig,
        treat: Mapping[str, str],
        bins: Any,
        binwidth: float | None,
        binrange: tuple[float, float] | None,
        stat: str,
        common_norm: bool,
        kde: bool,
        kde_kwargs: Mapping[str, Any] | None,
        cumulative: bool,
        multiple: str,
        element: str,
        rug: bool,
        reference_lines: Any,
        alpha: float | None,
        discrete: bool | None,
        palette: Any,
        hue_order: Sequence[Any] | None,
        order: Sequence[Any] | None,
        log_scale: Any,
        log_x: bool,
        log_y: bool,
        transform: Any,
        nonpositive: Nonpositive,
        show_n: bool,
        xlabel: str | None,
        ylabel: str | None,
        legend: bool,
        legend_loc: str,
        rotate_xticks: int | str | None,
        xticklabels: Any,
        yticklabels: Any,
        kwargs: Mapping[str, Any],
    ) -> Axes:
        """Draw a histogram into ``ax`` without touching the figure."""
        value_axis = orient_axis
        log_spec = self._normalise_log_scale(
            log_scale, value_axis, log_x, log_y
        )
        self._check_scale_conflict(log_spec, transform, value_axis, name)
        if kind == "categorical" and (log_spec.get(value_axis) or transform):
            raise ValueError(
                f"{name!r} is categorical, so log_scale and transform "
                f"cannot apply to the {value_axis!r} axis. Override with "
                f"treat_as={{{name!r}: 'numeric'}} if it is continuous."
            )

        levels: list[Any] = []
        if kind == "categorical":
            self._draw_hist_categorical(
                ax=ax,
                data=data,
                name=name,
                hue_name=hue_name,
                orient_axis=orient_axis,
                style=style,
                palette=palette,
                order=order,
                stat=stat,
                kwargs=kwargs,
            )
            levels = self.ordered_levels(data[name], order)
            label = name
        else:
            label = self._draw_hist_numeric(
                ax=ax,
                data=data,
                name=name,
                hue_name=hue_name,
                orient_axis=orient_axis,
                style=style,
                bins=bins,
                binwidth=binwidth,
                binrange=binrange,
                stat=stat,
                common_norm=common_norm,
                kde=kde,
                kde_kwargs=kde_kwargs,
                cumulative=cumulative,
                multiple=multiple,
                element=element,
                alpha=alpha,
                discrete=discrete,
                palette=palette,
                hue_order=hue_order,
                log_spec=log_spec,
                transform=transform,
                nonpositive=nonpositive,
                rug=rug,
                reference_lines=reference_lines,
                kwargs=kwargs,
            )

        count_axis = "y" if orient_axis == "x" else "x"
        self._apply_grid(ax, style, count_axis)
        self._despine(ax, style)
        stat_label = ylabel if orient_axis == "x" else xlabel
        default_stat = (
            stat.capitalize()
            if stat != "count"
            else _t(self.lang, "axis_count")
        )
        if orient_axis == "x":
            ax.set_xlabel(xlabel if xlabel is not None else label)
            ax.set_ylabel(
                stat_label if stat_label is not None else default_stat
            )
        else:
            ax.set_ylabel(ylabel if ylabel is not None else label)
            ax.set_xlabel(
                stat_label if stat_label is not None else default_stat
            )
        self._format_value_axis(
            ax, count_axis, style, percent=stat == "percent"
        )
        if kind == "categorical":
            self._relabel_ticks(ax, xticklabels, levels, "x")
            self._rotate_ticks(ax, levels, rotate_xticks)
        elif xticklabels is not None or yticklabels is not None:
            self._warn_once(
                f"ticklabels:{name}",
                f"xticklabels/yticklabels are ignored for the continuous "
                f"axis of histplot({name!r}); they apply to categorical "
                f"axes.",
            )
        if show_n:
            ax.annotate(
                _t(self.lang, "n_label", n=int(len(data))),
                xy=(0.98, 0.96),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=style.annot_size,
                color=style.subtitle_color,
            )
        self._style_legend(ax, style, hue_name, legend, legend_loc)
        return ax

    def _draw_hist_categorical(
        self,
        *,
        ax: Axes,
        data: pd.DataFrame,
        name: str,
        hue_name: str | None,
        orient_axis: str,
        style: StyleConfig,
        palette: Any,
        order: Sequence[Any] | None,
        stat: str,
        kwargs: Mapping[str, Any],
    ) -> None:
        """One bar per level, the fallback for categorical input."""
        levels = self.ordered_levels(data[name], order)
        colour_key = hue_name or name
        colour_levels = (
            self.ordered_levels(data[hue_name]) if hue_name else levels
        )
        mapping = self.color_map(colour_key, colour_levels, palette)
        # seaborn 0.13 deprecates palette without hue, so the categorical
        # variable is assigned to hue and the redundant legend removed.
        sns.countplot(
            data=data,
            **{orient_axis: name},
            hue=hue_name or name,
            order=levels,
            hue_order=colour_levels,
            palette=mapping,
            legend=hue_name is not None,
            edgecolor="none",
            zorder=style.data_zorder,
            ax=ax,
            **dict(kwargs),
        )
        if stat != "count":
            self._warn_once(
                f"hist-cat-stat:{name}",
                f"stat={stat!r} is ignored for the categorical fallback "
                f"of histplot({name!r}), which always shows counts.",
            )

    def _draw_hist_numeric(
        self,
        *,
        ax: Axes,
        data: pd.DataFrame,
        name: str,
        hue_name: str | None,
        orient_axis: str,
        style: StyleConfig,
        bins: Any,
        binwidth: float | None,
        binrange: tuple[float, float] | None,
        stat: str,
        common_norm: bool,
        kde: bool,
        kde_kwargs: Mapping[str, Any] | None,
        cumulative: bool,
        multiple: str,
        element: str,
        alpha: float | None,
        discrete: bool | None,
        palette: Any,
        hue_order: Sequence[Any] | None,
        log_spec: Mapping[str, Any],
        transform: Any,
        nonpositive: Nonpositive,
        rug: bool,
        reference_lines: Any,
        kwargs: Mapping[str, Any],
    ) -> str:
        """Bin and draw a true histogram; returns the axis label."""
        raw = pd.to_numeric(data[name], errors="coerce").to_numpy(float)
        values, meta = _apply_transform(
            raw,
            transform,
            nonpositive,
            column=name,
            lang=self.lang,
        )
        self._last_transform = meta
        plot_data = data.assign(**{name: values})

        log_base = log_spec.get(orient_axis)
        if log_base == "symlog":
            log_base = None
        if log_base:
            _guard_domain(
                values,
                "positive",
                nonpositive,
                column=name,
                operation="log_scale on a histogram",
            )
        is_discrete = (
            discrete
            if discrete is not None
            else bool(
                self._is_integer_like(pd.Series(values))
                and len(np.unique(values[np.isfinite(values)])) <= 25
            )
        )
        edges = self._bin_edges(
            values,
            bins,
            binwidth,
            binrange,
            is_discrete,
            float(log_base) if log_base else None,
        )
        n_levels = int(plot_data[hue_name].nunique()) if hue_name else 1
        if alpha is None:
            alpha = (
                max(0.35, 0.9 / n_levels)
                if multiple == "layer" and n_levels > 1
                else 0.9
            )
        levels = (
            self.ordered_levels(plot_data[hue_name], hue_order)
            if hue_name
            else []
        )
        mapping = (
            self.color_map(hue_name, levels, palette) if hue_name else None
        )
        if rug and len(plot_data) > 5000:
            self._warn_once(
                f"rug:{name}",
                f"rug=True is disabled for n = {len(plot_data):,} "
                f"(above 5000) because the ticks would be solid.",
            )
            rug = False

        common: dict[str, Any] = {
            "data": plot_data,
            orient_axis: name,
            "bins": edges,
            "stat": stat,
            "element": element,
            "cumulative": cumulative,
            "kde": kde,
            "ax": ax,
            "zorder": style.data_zorder,
            "edgecolor": "none",
            "alpha": alpha,
        }
        if kde and kde_kwargs:
            common["kde_kws"] = dict(kde_kwargs)
        if hue_name:
            common.update(
                hue=hue_name,
                hue_order=levels,
                palette=mapping,
                multiple=multiple,
                common_norm=common_norm,
                common_bins=True,
            )
        else:
            common["color"] = style.categorical[0]
        sns.histplot(**common, **dict(kwargs))
        if rug:
            sns.rugplot(
                data=plot_data,
                **{orient_axis: name},
                hue=hue_name,
                palette=mapping if hue_name else None,
                color=None if hue_name else style.categorical[0],
                ax=ax,
                legend=False,
                height=0.03,
            )
        if log_base:
            setter = ax.set_xscale if orient_axis == "x" else ax.set_yscale
            setter("log", base=float(log_base))
            self._axis_note(
                ax,
                style,
                _t(
                    self.lang,
                    "log_scale_note",
                    base=f"{float(log_base):g}",
                ),
                orient_axis,
            )
        self._draw_reference_lines(
            ax=ax,
            data=plot_data,
            name=name,
            hue_name=hue_name,
            levels=levels,
            mapping=mapping,
            spec=reference_lines,
            orient_axis=orient_axis,
            style=style,
        )
        return meta.label_for(name)

    def _draw_reference_lines(
        self,
        *,
        ax: Axes,
        data: pd.DataFrame,
        name: str,
        hue_name: str | None,
        levels: Sequence[Any],
        mapping: Mapping[Any, str] | None,
        spec: Any,
        orient_axis: str,
        style: StyleConfig,
    ) -> None:
        """Draw mean/median/mode or explicit reference lines."""
        if spec is None:
            return
        wanted = [spec] if isinstance(spec, (str, float, int)) else spec
        drawer = ax.axvline if orient_axis == "x" else ax.axhline
        groups: list[tuple[Any, pd.Series, str]] = []
        if hue_name:
            for level in levels:
                subset = data.loc[data[hue_name] == level, name]
                colour = (mapping or {}).get(level, style.text_color)
                groups.append((level, subset, colour))
        else:
            groups.append((None, data[name], style.text_color))

        for level, series, colour in groups:
            items = (
                wanted.items()
                if isinstance(wanted, Mapping)
                else [(w, w) for w in wanted]
            )
            for key, value in items:
                if isinstance(value, str):
                    stat_value = {
                        "mean": series.mean,
                        "median": series.median,
                        "mode": lambda s=series: (
                            s.mode().iloc[0] if not s.mode().empty else np.nan
                        ),
                    }[value]()
                    text = (
                        _t(self.lang, f"legend_{value}")
                        if value
                        in {
                            "mean",
                            "median",
                        }
                        else value
                    )
                else:
                    stat_value = float(value)
                    text = str(key)
                if not np.isfinite(stat_value):
                    continue
                suffix = f" ({level})" if level is not None else ""
                drawer(
                    stat_value,
                    color=colour,
                    linestyle="--",
                    linewidth=1.4,
                    zorder=style.data_zorder + 1,
                    label=f"{text}{suffix} = {stat_value:,.4g}",
                )

    # -- boxplot -----------------------------------------------------
    def boxplot(
        self,
        df: pd.DataFrame | None = None,
        x: str | pd.Series | None = None,
        y: str | pd.Series | Sequence[str] | None = None,
        hue: str | pd.Series | None = None,
        *,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | bool | None = None,
        subtitle: str | None = None,
        panel_title: str | bool | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        palette: str | list | dict | None = None,
        legend: bool = True,
        legend_loc: str = "best",
        order: Sequence[Any] | None = None,
        hue_order: Sequence[Any] | None = None,
        top_n: int | None = None,
        log_scale: Any = None,
        log_x: bool = False,
        log_y: bool = False,
        transform: Any = None,
        nonpositive: Nonpositive = "raise",
        facet: str | None = None,
        facet_col_wrap: int = 3,
        sharex: bool = True,
        sharey: bool = True,
        dropna: bool = True,
        treat_as: Mapping[str, str] | None = None,
        orientation: str | None = None,
        whis: float | tuple[float, float] = 1.5,
        showfliers: bool | None = None,
        notch: bool = False,
        showmeans: bool = False,
        overlay: str | None = None,
        overlay_kwargs: Mapping[str, Any] | None = None,
        sort: str | None = None,
        ascending: bool = False,
        annotate_n: bool = False,
        widths: str = "uniform",
        violin: bool = False,
        rotate_xticks: int | str | None = "auto",
        xticklabels: Mapping[Any, str] | Sequence[str] | None = None,
        yticklabels: Mapping[Any, str] | Sequence[str] | None = None,
        style: Mapping[str, Any] | str | None = None,
        save_as: str | Path | None = None,
        show: bool | None = None,
        ax_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Axes | Figure:
        """Compare a numeric distribution across categorical levels.

        Input resolution is symmetric: whichever of ``x`` and ``y``
        resolves to categorical becomes the grouping axis and the
        orientation follows automatically. A numeric column with few
        distinct values counts as a valid grouping axis, so boxes of
        ``price`` per ``stratum`` work without any override.

        Quartiles and whiskers are computed on the raw data. Drawing
        them on a log axis is safe, because quantiles are invariant
        under any monotone transform: the median of the logs is the log
        of the median. Means are not, so ``showmeans=True`` on a log
        axis puts the marker off-centre relative to a log-space mean.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Frame to plot.
        x, y : str, pandas.Series or list of str, optional
            The grouping and value variables in either order. ``y`` may
            be a list of numeric columns for wide-format input, giving
            one box per column.
        hue : str or pandas.Series, optional
            Second-level split, dodged within each group.
        orientation : {"vertical", "horizontal"}, optional
            Forced orientation; inferred from which axis is categorical
            when omitted. Horizontal reads better beyond ~8 groups or
            with long labels.
        whis : float or tuple of float, default 1.5
            IQR multiplier, or an explicit ``(low, high)`` percentile
            pair such as ``(5, 95)``.
        notch : bool, default False
            Draw median confidence notches; warns when a group has
            fewer than 10 rows, since notches can invert.
        showmeans : bool, default False
            Add a mean marker so skew is visible against the median.
        overlay : {"strip", "swarm", "none"}, optional
            Raw-point overlay. ``"swarm"` falls back to ``"strip"``
            above 1000 points per group. When active, ``showfliers``
            defaults to ``False`` so outliers are not drawn twice.
        sort : {"median", "alpha", "natural", None}, optional
            Group ordering. Defaults to ``"median"``, except for an
            ordinal grouper (numeric-low-cardinality or an ordered
            ``CategoricalDtype``) where it defaults to ``"natural"`` —
            an ordinal scale must never be reordered by median.
        widths : {"uniform", "count"}, default "uniform"
            ``"count"`` scales box width by ``sqrt(n)``.
        violin : bool, default False
            Replace boxes with violins, keeping an inner mini-box.
        annotate_n : bool, default False
            Write ``n = ...`` beside each box.

        Returns
        -------
        matplotlib.axes.Axes or matplotlib.figure.Figure
            The axes for a single panel, the figure when faceting.

        Raises
        ------
        ValueError
            For a categorical-only request, or two continuous numeric
            variables with no valid grouping axis.

        Examples
        --------
        >>> import matplotlib, pandas as pd
        >>> matplotlib.use("Agg")
        >>> df = pd.DataFrame({"g": list("aabb"), "v": [1.5, 2.25, 3.75, 4.1]})
        >>> ax = EDAPlotter(df).boxplot(x="g", y="v")
        >>> ax.get_ylabel()
        'v'
        """
        frame, style_cfg, treat, extra_ax = self._resolve_common(
            df,
            style=style,
            figsize=figsize,
            treat_as=treat_as,
            ax=ax,
            facet=facet,
            ax_kwargs=ax_kwargs,
        )
        if isinstance(y, (list, tuple)) and not isinstance(y, str):
            frame, x, y = self._melt_wide(frame, list(y))
        group, value, orient = self._resolve_box_axes(
            frame, x, y, orientation, treat
        )
        value_name = str(value.name)
        group_name = str(group.name) if group is not None else None

        work = frame.copy()
        work[value_name] = value.to_numpy()
        if group_name:
            work[group_name] = group.to_numpy()
        hue_series = self._column(work, hue, "hue")
        hue_name = str(hue_series.name) if hue_series is not None else None
        work = self._dropna_subset(
            work,
            [value_name, group_name, hue_name, facet],
            dropna,
            f"boxplot({value_name!r})",
        )
        if group_name:
            work[group_name] = self._limit_levels(work[group_name], top_n)
            work = self._filter_small_groups(
                work, group_name, f"boxplot({value_name!r})"
            )
        if hue_name:
            work[hue_name] = self._limit_levels(work[hue_name], None)
            if work[hue_name].nunique() > 4:
                self._warn_once(
                    f"box-hue:{hue_name}",
                    f"hue={hue_name!r} has "
                    f"{work[hue_name].nunique()} levels; dodged boxes "
                    f"get thin. Consider facet={hue_name!r} instead.",
                )

        if facet is not None:
            return self._facet(
                work,
                facet,
                lambda panel_ax, sub, level: self.boxplot(
                    sub,
                    x=x if not isinstance(x, pd.Series) else None,
                    y=y if not isinstance(y, pd.Series) else None,
                    hue=hue_name,
                    ax=panel_ax,
                    orientation=orient,
                    whis=whis,
                    showfliers=showfliers,
                    showmeans=showmeans,
                    overlay=overlay,
                    violin=violin,
                    sort=sort,
                    legend=legend,
                    treat_as=treat,
                    title=False,
                    panel_title=False,
                    style=style,
                    **kwargs,
                ),
                style=style_cfg,
                col_wrap=facet_col_wrap,
                sharex=sharex,
                sharey=sharey,
                figsize=figsize,
                title=self._auto_title(
                    title, self._box_title(value_name, group_name)
                ),
                subtitle=subtitle,
                save_as=save_as,
                show=show,
            )

        owns_figure = ax is None
        with self._style_scope(style_cfg):
            if owns_figure:
                fig, axes = self._new_figure(style_cfg, figsize)
                ax = axes[0, 0]
            else:
                fig = ax.figure
            self._draw_box(
                ax=ax,
                data=work,
                value_name=value_name,
                group_name=group_name,
                hue_name=hue_name,
                orient=orient,
                style=style_cfg,
                whis=whis,
                showfliers=showfliers,
                notch=notch,
                showmeans=showmeans,
                overlay=overlay,
                overlay_kwargs=overlay_kwargs,
                sort=sort,
                ascending=ascending,
                order=order,
                hue_order=hue_order,
                palette=palette,
                widths=widths,
                violin=violin,
                annotate_n=annotate_n,
                log_scale=log_scale,
                log_x=log_x,
                log_y=log_y,
                transform=transform,
                nonpositive=nonpositive,
                xlabel=xlabel,
                ylabel=ylabel,
                legend=legend,
                legend_loc=legend_loc,
                rotate_xticks=rotate_xticks,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                treat=treat,
                kwargs=kwargs,
            )
            self._panel_title(
                ax,
                style_cfg,
                self._auto_title(
                    panel_title,
                    None
                    if owns_figure
                    else self._box_title(value_name, group_name),
                ),
            )
            if extra_ax:
                ax.set(**extra_ax)
            if owns_figure:
                top = self._place_header(
                    fig,
                    style_cfg,
                    self._auto_title(
                        title, self._box_title(value_name, group_name)
                    ),
                    subtitle,
                    style_cfg.margins,
                )
                self._finish(fig, style_cfg, {}, top, save_as, show)
        return ax

    @staticmethod
    def _box_title(value_name: str, group_name: str | None) -> str:
        """Default figure title for a boxplot."""
        if group_name:
            return f"Distribution of {value_name} by {group_name}"
        return f"Distribution of {value_name}"

    @staticmethod
    def _melt_wide(
        frame: pd.DataFrame, columns: Sequence[str]
    ) -> tuple[pd.DataFrame, str, str]:
        """Reshape wide-format input into one box per column."""
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"y={missing} are not columns of the frame.")
        long = frame[list(columns)].melt(
            var_name="variable", value_name="value"
        )
        return long, "variable", "value"

    def _resolve_box_axes(
        self,
        frame: pd.DataFrame,
        x: Any,
        y: Any,
        orientation: str | None,
        treat: Mapping[str, str],
    ) -> tuple[pd.Series | None, pd.Series, str]:
        """Work out which axis groups and which one holds the values."""
        xs = self._column(frame, x, "x")
        ys = self._column(frame, y, "y")
        if xs is None and ys is None:
            raise ValueError(
                "boxplot needs at least one variable: pass x=, y=, or both."
            )
        kinds = {
            "x": None
            if xs is None
            else self.resolve_kind(xs, treat.get(str(xs.name))),
            "y": None
            if ys is None
            else self.resolve_kind(ys, treat.get(str(ys.name))),
        }
        if xs is not None and ys is None:
            if kinds["x"] == "categorical":
                raise ValueError(
                    f"x={xs.name!r} is categorical and there is nothing "
                    f"to summarise. Pass a numeric y=, or use barplot "
                    f"for level counts."
                )
            return None, xs, orientation or "vertical"
        if ys is not None and xs is None:
            if kinds["y"] == "categorical":
                raise ValueError(
                    f"y={ys.name!r} is categorical and there is nothing "
                    f"to summarise. Pass a numeric x=, or use barplot "
                    f"for level counts."
                )
            return None, ys, orientation or "vertical"

        assert xs is not None and ys is not None
        if kinds["x"] == "categorical" and kinds["y"] == "numeric":
            return xs, ys, orientation or "vertical"
        if kinds["y"] == "categorical" and kinds["x"] == "numeric":
            if orientation is None:
                self._warn_once(
                    f"box-orient:{xs.name}",
                    f"y={ys.name!r} is the categorical axis, so the "
                    f"boxes are drawn horizontally. Pass "
                    f"orientation='vertical' to override.",
                )
            return ys, xs, orientation or "horizontal"
        if kinds["x"] == "categorical" and kinds["y"] == "categorical":
            raise ValueError(
                f"Both x={xs.name!r} and y={ys.name!r} are categorical, "
                f"so there is no distribution to summarise. Use barplot "
                f"for a cross-tab."
            )
        raise ValueError(
            f"x={xs.name!r} and y={ys.name!r} are both continuous "
            f"numeric, so neither can group the other. Use "
            f"scatterplot, or bin one of them, or pass "
            f"treat_as={{{str(xs.name)!r}: 'categorical'}}."
        )

    def _draw_box(
        self,
        *,
        ax: Axes,
        data: pd.DataFrame,
        value_name: str,
        group_name: str | None,
        hue_name: str | None,
        orient: str,
        style: StyleConfig,
        whis: float | tuple[float, float],
        showfliers: bool | None,
        notch: bool,
        showmeans: bool,
        overlay: str | None,
        overlay_kwargs: Mapping[str, Any] | None,
        sort: str | None,
        ascending: bool,
        order: Sequence[Any] | None,
        hue_order: Sequence[Any] | None,
        palette: Any,
        widths: str,
        violin: bool,
        annotate_n: bool,
        log_scale: Any,
        log_x: bool,
        log_y: bool,
        transform: Any,
        nonpositive: Nonpositive,
        xlabel: str | None,
        ylabel: str | None,
        legend: bool,
        legend_loc: str,
        rotate_xticks: int | str | None,
        xticklabels: Any,
        yticklabels: Any,
        treat: Mapping[str, str],
        kwargs: Mapping[str, Any],
    ) -> Axes:
        """Draw boxes into ``ax`` without touching the figure."""
        value_axis = "y" if orient == "vertical" else "x"
        cat_axis = "x" if orient == "vertical" else "y"
        log_spec = self._normalise_log_scale(
            log_scale, value_axis, log_x, log_y
        )
        self._check_scale_conflict(log_spec, transform, value_axis, value_name)
        if cat_axis in log_spec:
            raise ValueError(
                f"log_scale targets the {cat_axis!r} axis, which holds "
                f"the categorical grouping variable. A log scale only "
                f"applies to the continuous axis."
            )

        raw = pd.to_numeric(data[value_name], errors="coerce")
        values, meta = _apply_transform(
            raw, transform, nonpositive, column=value_name, lang=self.lang
        )
        self._last_transform = meta
        plot_data = data.assign(**{value_name: values})

        levels: list[Any] = []
        if group_name:
            effective_sort = sort
            if effective_sort is None and order is None:
                effective_sort = (
                    "natural"
                    if self._is_ordinal(plot_data[group_name])
                    else "median"
                )
            levels = self.ordered_levels(
                plot_data[group_name],
                order,
                effective_sort,
                plot_data[value_name],
                ascending,
            )
            if orient == "horizontal" and effective_sort in {
                "median",
                "value",
            }:
                levels = list(reversed(levels))
            plot_data = plot_data[plot_data[group_name].isin(levels)]

        if overlay in {"strip", "swarm"} and showfliers is None:
            showfliers = False
        elif showfliers is None:
            showfliers = True
        if notch and group_name:
            small = plot_data.groupby(group_name, observed=True).size().lt(10)
            if bool(small.any()):
                self._warn_once(
                    f"notch:{value_name}",
                    f"notch=True with {int(small.sum())} group(s) of "
                    f"fewer than 10 rows: the notches can invert and "
                    f"become unreadable.",
                )

        colour_key = hue_name or group_name or value_name
        colour_levels = (
            self.ordered_levels(plot_data[hue_name], hue_order)
            if hue_name
            else levels
        )
        mapping = (
            self.color_map(colour_key, colour_levels, palette)
            if colour_levels
            else None
        )

        common: dict[str, Any] = {
            "data": plot_data,
            value_axis: value_name,
            "ax": ax,
            "zorder": style.data_zorder,
        }
        if group_name:
            common[cat_axis] = group_name
            common["order"] = levels
        if hue_name:
            common.update(
                hue=hue_name, hue_order=colour_levels, palette=mapping
            )
        elif group_name and mapping:
            # seaborn 0.13: palette requires hue, so the grouping
            # variable doubles as hue and its legend is suppressed.
            common.update(
                hue=group_name,
                hue_order=levels,
                palette=mapping,
                legend=False,
            )
        else:
            common["color"] = style.categorical[0]

        if violin:
            sns.violinplot(
                **common,
                inner="box",
                linewidth=1.0,
                **dict(kwargs),
            )
        else:
            box_kwargs: dict[str, Any] = {
                "whis": whis,
                "showfliers": showfliers,
                "notch": notch,
                "showmeans": showmeans,
                "linewidth": 1.0,
                "flierprops": {
                    "marker": "o",
                    "markersize": 3,
                    "markerfacecolor": style.subtitle_color,
                    "markeredgecolor": "none",
                    "alpha": 0.5,
                },
            }
            if showmeans:
                box_kwargs["meanprops"] = {
                    "marker": "D",
                    "markerfacecolor": style.negative,
                    "markeredgecolor": "none",
                    "markersize": 5,
                }
            sns.boxplot(**common, **box_kwargs, **dict(kwargs))
            if widths == "count" and group_name:
                counts = (
                    plot_data.groupby(group_name, observed=True)
                    .size()
                    .reindex(levels)
                    .fillna(0)
                )
                self._scale_box_widths(ax, counts.to_numpy(float), cat_axis)

        if overlay in {"strip", "swarm"}:
            self._draw_overlay(
                ax=ax,
                data=plot_data,
                value_name=value_name,
                group_name=group_name,
                hue_name=hue_name,
                levels=levels,
                colour_levels=colour_levels,
                mapping=mapping,
                value_axis=value_axis,
                cat_axis=cat_axis,
                overlay=overlay,
                overlay_kwargs=overlay_kwargs,
                style=style,
            )

        if log_spec:
            self._apply_log_axis(
                ax,
                log_spec,
                style,
                {value_axis: raw.to_numpy(float)},
                {value_axis: value_name},
                nonpositive,
            )
            if showmeans:
                self._warn_once(
                    f"box-logmean:{value_name}",
                    "showmeans=True on a log axis: quantiles are "
                    "invariant under the transform but the mean is "
                    "not, so the marker sits off-centre relative to a "
                    "log-space mean.",
                )
        self._apply_grid(ax, style, value_axis)
        self._despine(ax, style)

        value_label = meta.label_for(value_name)
        if orient == "vertical":
            ax.set_ylabel(ylabel if ylabel is not None else value_label)
            ax.set_xlabel(xlabel if xlabel is not None else (group_name or ""))
        else:
            ax.set_xlabel(xlabel if xlabel is not None else value_label)
            ax.set_ylabel(ylabel if ylabel is not None else (group_name or ""))
        if value_axis not in log_spec:
            self._format_value_axis(ax, value_axis, style)
        if levels:
            self._relabel_ticks(
                ax,
                xticklabels if cat_axis == "x" else yticklabels,
                levels,
                cat_axis,
            )
            if cat_axis == "x":
                self._rotate_ticks(ax, levels, rotate_xticks)
        if annotate_n and group_name:
            self._annotate_group_n(
                ax, plot_data, group_name, levels, cat_axis, style
            )
        self._style_legend(ax, style, hue_name, legend, legend_loc)
        return ax

    def _scale_box_widths(
        self, ax: Axes, counts: np.ndarray, cat_axis: str
    ) -> None:
        """Shrink each box by ``sqrt(n)`` so small groups recede.

        seaborn only accepts a ``widths`` list when no ``hue`` is
        assigned, and the palette workaround always assigns one, so the
        drawn artists are rescaled in place instead.
        """
        boxes = [p for p in ax.patches if isinstance(p, PathPatch)]
        if len(boxes) != len(counts) or not len(counts):
            self._warn_once(
                "box-widths",
                f"widths='count' expected {len(counts)} box artists but "
                f"found {len(boxes)}; leaving widths uniform.",
            )
            return
        root = np.sqrt(np.asarray(counts, dtype=float))
        peak = float(root.max())
        if peak <= 0:
            return
        factors = 0.25 + 0.75 * root / peak
        index = 0 if cat_axis == "x" else 1
        centres: list[tuple[float, float]] = []
        for box, factor in zip(boxes, factors):
            verts = box.get_path().vertices
            lo = float(verts[:, index].min())
            hi = float(verts[:, index].max())
            centre = (lo + hi) / 2.0
            verts[:, index] = centre + (verts[:, index] - centre) * factor
            centres.append((centre, factor))
        for line in ax.lines:
            data = line.get_xdata() if cat_axis == "x" else line.get_ydata()
            values = np.asarray(data, dtype=float)
            if values.size < 2 or np.ptp(values) == 0:
                continue
            mid = (values.min() + values.max()) / 2.0
            for centre, factor in centres:
                if abs(mid - centre) < 1e-9:
                    scaled = centre + (values - centre) * factor
                    if cat_axis == "x":
                        line.set_xdata(scaled)
                    else:
                        line.set_ydata(scaled)
                    break

    def _is_ordinal(self, series: pd.Series) -> bool:
        """Whether reordering this grouper by median would be wrong."""
        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            return bool(dtype.ordered)
        return bool(is_numeric_dtype(dtype) or is_bool_dtype(dtype))

    def _draw_overlay(
        self,
        *,
        ax: Axes,
        data: pd.DataFrame,
        value_name: str,
        group_name: str | None,
        hue_name: str | None,
        levels: Sequence[Any],
        colour_levels: Sequence[Any],
        mapping: Mapping[Any, str] | None,
        value_axis: str,
        cat_axis: str,
        overlay: str,
        overlay_kwargs: Mapping[str, Any] | None,
        style: StyleConfig,
    ) -> None:
        """Add jittered or swarmed raw points on top of the boxes."""
        per_group = (
            int(data.groupby(group_name, observed=True).size().max())
            if group_name
            else len(data)
        )
        if overlay == "swarm" and per_group > 1000:
            self._warn_once(
                f"swarm:{value_name}",
                f"overlay='swarm' with {per_group:,} points in the "
                f"largest group is too dense to lay out; falling back "
                f"to 'strip'.",
            )
            overlay = "strip"
        alpha = float(np.clip(600.0 / max(per_group, 1), 0.12, 0.75))
        opts: dict[str, Any] = {
            "data": data,
            value_axis: value_name,
            "ax": ax,
            "size": 2.6,
            "alpha": alpha,
            "linewidth": 0,
            "zorder": style.data_zorder + 1,
            "legend": False,
        }
        if group_name:
            opts[cat_axis] = group_name
            opts["order"] = list(levels)
        if hue_name:
            opts.update(
                hue=hue_name,
                hue_order=list(colour_levels),
                palette=mapping,
                dodge=True,
            )
        else:
            opts["color"] = style.text_color
        opts.update(dict(overlay_kwargs or {}))
        if overlay == "swarm":
            sns.swarmplot(**opts)
        else:
            sns.stripplot(**opts, jitter=0.25)

    def _annotate_group_n(
        self,
        ax: Axes,
        data: pd.DataFrame,
        group_name: str,
        levels: Sequence[Any],
        cat_axis: str,
        style: StyleConfig,
    ) -> None:
        """Write ``n = ...`` beside each box."""
        counts = (
            data.groupby(group_name, observed=True)
            .size()
            .reindex(levels)
            .fillna(0)
            .astype(int)
        )
        for index, level in enumerate(levels):
            text = _t(self.lang, "n_label", n=int(counts.loc[level]))
            if cat_axis == "x":
                ax.annotate(
                    text,
                    xy=(index, 0.01),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="bottom",
                    fontsize=style.annot_size * 0.85,
                    color=style.subtitle_color,
                )
            else:
                ax.annotate(
                    text,
                    xy=(0.99, index),
                    xycoords=("axes fraction", "data"),
                    ha="right",
                    va="center",
                    fontsize=style.annot_size * 0.85,
                    color=style.subtitle_color,
                )

    # -- normality testing -------------------------------------------
    def run_normality_test(
        self,
        values: np.ndarray | pd.Series,
        test: str = "auto",
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Run a normality test and describe the outcome.

        ``test="auto"`` picks Shapiro-Wilk up to n = 5000 and
        D'Agostino-Pearson K squared above it, because Shapiro becomes
        unreliable and over-powered on large samples.

        Anderson-Darling is handled specially: it returns a statistic
        and a table of critical values rather than a p-value, so the
        result carries ``critical`` and ``p`` stays ``nan``. No p-value
        is ever fabricated for it.

        Parameters
        ----------
        values : numpy.ndarray or pandas.Series
            Sample to test; nulls are dropped.
        test : str, default "auto"
            One of ``"auto"``, ``"shapiro"``, ``"dagostino"``,
            ``"jarque_bera"``, ``"anderson"``, ``"ks"``,
            ``"lilliefors"``.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        dict
            ``test``, ``statistic``, ``p``, ``critical``, ``reject``,
            ``alpha``, ``n`` and ``switched`` keys.

        Examples
        --------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> out = EDAPlotter().run_normality_test(rng.normal(size=200))
        >>> out["test"]
        'Shapiro-Wilk'
        """
        from scipy import stats as sps

        data = np.asarray(pd.Series(values).dropna().to_numpy(), dtype=float)
        data = data[np.isfinite(data)]
        n = int(data.size)
        result: dict[str, Any] = {
            "n": n,
            "alpha": float(alpha),
            "statistic": float("nan"),
            "p": float("nan"),
            "critical": None,
            "reject": None,
            "switched": False,
            "test": test,
        }
        if n < 3:
            result["test"] = "n/a"
            return result

        chosen = test
        if test == "auto":
            chosen = "shapiro" if n <= 5000 else "dagostino"
            result["switched"] = n > 5000
        if chosen == "shapiro" and n > 5000:
            self._warn_once(
                "shapiro-large-n",
                f"Shapiro-Wilk with n = {n:,} is unreliable and "
                f"over-powered; consider test='anderson' or "
                f"test='dagostino'.",
            )
        if chosen == "ks":
            self._warn_once(
                "ks-fitted",
                "A Kolmogorov-Smirnov test against an MLE-fitted normal "
                "is anti-conservative. Use test='lilliefors', which "
                "corrects for the estimated parameters.",
            )

        if chosen == "shapiro":
            stat, p = sps.shapiro(data)
            label = "Shapiro-Wilk"
        elif chosen == "dagostino":
            stat, p = sps.normaltest(data)
            label = "D'Agostino-Pearson K2"
        elif chosen == "jarque_bera":
            stat, p = sps.jarque_bera(data)
            label = "Jarque-Bera"
        elif chosen == "anderson":
            result.update(self._anderson(data, alpha))
            return result
        elif chosen == "lilliefors":
            stat, p = self._lilliefors(data)
            label = "Lilliefors"
        elif chosen == "ks":
            loc, scale = float(np.mean(data)), float(np.std(data, ddof=1))
            stat, p = sps.kstest(data, "norm", args=(loc, scale))
            label = "Kolmogorov-Smirnov"
        else:
            raise ValueError(
                f"test={test!r} is not recognised. Use 'auto', "
                f"'shapiro', 'dagostino', 'jarque_bera', 'anderson', "
                f"'ks' or 'lilliefors'."
            )
        result.update(
            test=label,
            statistic=float(stat),
            p=float(p),
            reject=bool(p < alpha),
        )
        return result

    @staticmethod
    def _anderson(data: np.ndarray, alpha: float) -> dict[str, Any]:
        """Anderson-Darling, preferring critical values over a p-value.

        The test yields a statistic and a table of critical values, not
        a p-value, and the verdict is phrased against the critical value
        at ``alpha``. scipy 1.17 deprecated that API in favour of an
        interpolated p-value and will drop the table in 1.19, so the
        legacy path is used while it exists and the p-value path is the
        fallback.
        """
        from scipy import stats as sps

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            outcome = sps.anderson(data, dist="norm")
        stat = float(outcome.statistic)
        crits = getattr(outcome, "critical_values", None)
        levels = getattr(outcome, "significance_level", None)
        if crits is not None and levels is not None:
            levels = np.asarray(levels, dtype=float)
            crits = np.asarray(crits, dtype=float)
            index = int(np.argmin(np.abs(levels - alpha * 100)))
            return {
                "test": "Anderson-Darling",
                "statistic": stat,
                "critical": float(crits[index]),
                "critical_level": float(levels[index]),
                "reject": bool(stat > crits[index]),
            }
        outcome = sps.anderson(data, dist="norm", method="interpolate")
        p = float(outcome.pvalue)
        return {
            "test": "Anderson-Darling",
            "statistic": stat,
            "p": p,
            "reject": bool(p < alpha),
        }

    @staticmethod
    def _lilliefors(data: np.ndarray) -> tuple[float, float]:
        """Lilliefors test, falling back to a Monte-Carlo p-value."""
        from scipy import stats as sps

        try:
            from statsmodels.stats.diagnostic import lilliefors

            stat, p = lilliefors(data, dist="norm")
            return float(stat), float(p)
        except ImportError:
            pass
        n = data.size
        loc, scale = float(np.mean(data)), float(np.std(data, ddof=1))
        stat = float(sps.kstest(data, "norm", args=(loc, scale)).statistic)
        rng = np.random.default_rng(0)
        sims = rng.normal(size=(400, n))
        centred = (sims - sims.mean(1, keepdims=True)) / sims.std(
            1, ddof=1, keepdims=True
        )
        null = np.array(
            [
                sps.kstest(row, "norm", args=(0.0, 1.0)).statistic
                for row in centred
            ]
        )
        return stat, float((null >= stat).mean())

    def describe_distribution(
        self,
        values: np.ndarray | pd.Series,
        test: str = "auto",
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Summary statistics plus a normality test for one variable.

        Parameters
        ----------
        values : numpy.ndarray or pandas.Series
            The variable to summarise.
        test : str, default "auto"
            Passed to :meth:`run_normality_test`.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        dict
            ``n``, ``missing``, ``mean``, ``median``, ``std``, ``iqr``,
            ``min``, ``max``, ``skew``, ``kurtosis``, ``n_unique`` and
            the flattened test result.

        Examples
        --------
        >>> import numpy as np
        >>> stats = EDAPlotter().describe_distribution(np.arange(50.0))
        >>> stats["n"]
        50
        """
        series = pd.Series(values, dtype="float64")
        clean = series.dropna()
        q1, q3 = (
            (float(clean.quantile(0.25)), float(clean.quantile(0.75)))
            if len(clean)
            else (np.nan, np.nan)
        )
        outcome = self.run_normality_test(clean, test, alpha)
        return {
            "n": int(len(clean)),
            "missing": int(series.isna().sum()),
            "missing_pct": (
                100.0 * series.isna().sum() / len(series)
                if len(series)
                else 0.0
            ),
            "mean": float(clean.mean()) if len(clean) else np.nan,
            "median": float(clean.median()) if len(clean) else np.nan,
            "std": float(clean.std()) if len(clean) > 1 else np.nan,
            "iqr": q3 - q1,
            "min": float(clean.min()) if len(clean) else np.nan,
            "max": float(clean.max()) if len(clean) else np.nan,
            "skew": float(clean.skew()) if len(clean) > 2 else np.nan,
            "kurtosis": float(clean.kurt()) if len(clean) > 3 else np.nan,
            "n_unique": int(clean.nunique()),
            "test": outcome["test"],
            "statistic": outcome["statistic"],
            "p_value": outcome["p"],
            "critical": outcome.get("critical"),
            "reject_normality": outcome["reject"],
            "alpha": outcome["alpha"],
        }

    def verdict_text(
        self, stats: Mapping[str, Any], lang: str | None = None
    ) -> tuple[str, bool]:
        """Render the four-part plain-language normality verdict.

        The verdict always carries the numbers, the decision, the
        direction of the deviation read from skew and excess kurtosis
        rather than from the p-value, and a sample-size caveat where one
        applies. Failing to reject is phrased as *no evidence against
        normality*, never as evidence of normality.

        Parameters
        ----------
        stats : mapping
            Output of :meth:`describe_distribution`.
        lang : str, optional
            Language override.

        Returns
        -------
        text, reject
            The rendered verdict and whether normality was rejected.

        Examples
        --------
        >>> import numpy as np
        >>> p = EDAPlotter()
        >>> s = p.describe_distribution(np.arange(100.0))
        >>> isinstance(p.verdict_text(s)[0], str)
        True
        """
        lang = lang or self.lang
        alpha = float(stats.get("alpha", 0.05))
        reject = bool(stats.get("reject_normality"))
        lines: list[str] = []

        if stats.get("critical") is not None:
            key = "test_anderson_reject" if reject else "test_anderson_keep"
            lines.append(
                _t(
                    lang,
                    key,
                    test=stats["test"],
                    stat=stats["statistic"],
                    crit=stats["critical"],
                    alpha=alpha,
                    alpha_pct=alpha * 100,
                )
            )
        else:
            p = float(stats.get("p_value", np.nan))
            key = "test_reject" if reject else "test_keep"
            lines.append(
                _t(
                    lang,
                    key,
                    test=stats["test"],
                    stat=stats["statistic"],
                    p=_format_p(p),
                    p_cmp=_format_p(p),
                    alpha=alpha,
                )
            )

        skew = float(stats.get("skew", np.nan))
        kurt = float(stats.get("kurtosis", np.nan))
        if np.isfinite(skew) and np.isfinite(kurt):
            shape = (
                "shape_sym"
                if abs(skew) < 0.5
                else ("shape_right" if skew > 0 else "shape_left")
            )
            tail = (
                "tail_meso"
                if abs(kurt) < 0.5
                else ("tail_heavy" if kurt > 0 else "tail_light")
            )
            lines.append(
                _t(
                    lang,
                    "shape_line",
                    shape=_t(lang, shape),
                    skew=skew,
                    tail=_t(lang, tail),
                    kurt=kurt,
                )
            )

        n = int(stats.get("n", 0))
        if n > 5000:
            lines.append(_t(lang, "caveat_large_n", n=n))
        elif n < 20:
            lines.append(_t(lang, "caveat_small_n", n=n))
        n_unique = int(stats.get("n_unique", 0))
        if n_unique and n_unique < 10:
            lines.append(_t(lang, "caveat_ties", k=n_unique))
        if (
            np.isfinite(skew)
            and skew > 0.5
            and float(stats.get("min", -1)) > 0
        ):
            lines.append(_t(lang, "suggest_transform", name="log"))
        return "\n".join(lines), reject

    # -- qqplot ------------------------------------------------------
    def qqplot(
        self,
        df: pd.DataFrame | None = None,
        x: str | pd.Series | None = None,
        hue: str | pd.Series | None = None,
        *,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | bool | None = None,
        subtitle: str | None = None,
        panel_title: str | bool | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        palette: str | list | dict | None = None,
        legend: bool = True,
        legend_loc: str = "best",
        hue_order: Sequence[Any] | None = None,
        top_n: int | None = None,
        log_scale: Any = None,
        transform: Any = None,
        nonpositive: Nonpositive = "raise",
        facet: str | None = None,
        facet_col_wrap: int = 3,
        sharex: bool = False,
        sharey: bool = False,
        dropna: bool = True,
        treat_as: Mapping[str, str] | None = None,
        dist: Any = "norm",
        dist_params: Mapping[str, Any] | None = None,
        fit: bool = True,
        standardize: bool = False,
        line: str | None = "q",
        conf_band: float | None = None,
        annotate_test: str | Sequence[str] | None = None,
        plotting_positions: str = "blom",
        marker_kwargs: Mapping[str, Any] | None = None,
        compare: str | pd.Series | None = None,
        alpha: float = 0.05,
        style: Mapping[str, Any] | str | None = None,
        save_as: str | Path | None = None,
        show: bool | None = None,
        ax_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Axes | Figure:
        """Assess whether a variable follows a theoretical distribution.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Frame to plot.
        x : str or pandas.Series
            The numeric variable to assess.
        hue : str or pandas.Series, optional
            Categorical split. Each level is fitted **separately** and
            gets its own reference line, because one shared line across
            groups with different scales is misleading.
        dist : str or scipy distribution, default "norm"
            ``"norm"``, ``"t"``, ``"lognorm"``, ``"expon"``,
            ``"uniform"``, ``"gamma"``, ``"beta"``, ``"chi2"``,
            ``"weibull_min"``, or any frozen / ``rv_continuous`` object.
        dist_params : mapping, optional
            Shape parameters, e.g. ``{"df": 5}`` for ``"t"``.
        fit : bool, default True
            Estimate ``loc``, ``scale`` and any shape parameters by MLE.
        standardize : bool, default False
            z-score the sample so the reference is the 45 degree line.
        line : {"q", "45", "s", "r", None}, default "q"
            Reference line: through the quartiles, the identity, a
            standardized line, or an OLS fit.
        conf_band : float, optional
            Pointwise band level, e.g. ``0.95``. Uses ``statsmodels``
            when available, otherwise a parametric bootstrap.
        annotate_test : str or sequence, optional
            Any of ``"shapiro"``, ``"ks"``, ``"anderson"``,
            ``"jarque_bera"``, rendered in a corner box.
        plotting_positions : {"blom", "median", "hazen", "weibull"}, \
default "blom"
            Plotting-position convention.
        compare : str or pandas.Series, optional
            Second sample for a two-sample empirical QQ plot.
        facet : str, optional
            One panel per level; ``sharex``/``sharey`` default to
            ``False`` here because group scales usually differ.

        Returns
        -------
        matplotlib.axes.Axes or matplotlib.figure.Figure
            The axes for a single panel, the figure when faceting.

        Raises
        ------
        ValueError
            When ``x`` is categorical, or when ``log_scale`` is given —
            a log axis is not meaningful on a QQ plot.

        Examples
        --------
        >>> import matplotlib, numpy as np, pandas as pd
        >>> matplotlib.use("Agg")
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"v": rng.normal(size=80)})
        >>> ax = EDAPlotter(df).qqplot(x="v")
        >>> ax.get_xlabel().startswith("Theoretical")
        True
        """
        if log_scale is not None and log_scale is not False:
            raise ValueError(
                "log_scale is not meaningful on a QQ plot, because both "
                "axes are already quantiles. Either test against a "
                "log-normal with dist='lognorm', or test the logged "
                "variable against a normal with transform='log'."
            )
        frame, style_cfg, treat, extra_ax = self._resolve_common(
            df,
            style=style,
            figsize=figsize,
            treat_as=treat_as,
            ax=ax,
            facet=facet,
            ax_kwargs=ax_kwargs,
        )
        target = self._column(frame, x, "x")
        if target is None:
            raise ValueError("qqplot needs x=<numeric column>.")
        name = str(target.name)
        kind = self.resolve_kind(target, treat.get(name))
        if kind == "categorical" and not self._is_integer_like(target):
            raise ValueError(
                f"x={name!r} resolves to categorical, so a QQ plot of it "
                f"is meaningless. Use barplot for level counts, or "
                f"treat_as={{{name!r}: 'numeric'}} if it is continuous."
            )
        if target.nunique(dropna=True) <= self.cat_max_cardinality:
            self._warn_once(
                f"qq-ties:{name}",
                f"{name!r} has only {target.nunique()} distinct values; "
                f"ties will show up as a staircase pattern in the QQ "
                f"plot.",
            )

        hue_series = self._column(frame, hue, "hue")
        hue_name = str(hue_series.name) if hue_series is not None else None
        work = frame.copy()
        work[name] = target.to_numpy()
        if hue_name:
            work[hue_name] = hue_series.to_numpy()
        work = self._dropna_subset(
            work, [name, hue_name, facet], dropna, f"qqplot({name!r})"
        )
        if hue_name:
            work[hue_name] = self._limit_levels(work[hue_name], top_n)

        if facet is not None:
            return self._facet(
                work,
                facet,
                lambda panel_ax, sub, level: self.qqplot(
                    sub,
                    x=name,
                    hue=hue_name,
                    ax=panel_ax,
                    dist=dist,
                    dist_params=dist_params,
                    fit=fit,
                    standardize=standardize,
                    line=line,
                    conf_band=conf_band,
                    annotate_test=annotate_test,
                    plotting_positions=plotting_positions,
                    transform=transform,
                    nonpositive=nonpositive,
                    treat_as=treat,
                    legend=legend,
                    title=False,
                    panel_title=False,
                    style=style,
                    **kwargs,
                ),
                style=style_cfg,
                col_wrap=facet_col_wrap,
                sharex=sharex,
                sharey=sharey,
                figsize=figsize,
                title=self._auto_title(title, f"Q-Q plot of {name}"),
                subtitle=subtitle,
                min_rows=8,
                save_as=save_as,
                show=show,
            )

        owns_figure = ax is None
        with self._style_scope(style_cfg):
            if owns_figure:
                fig, axes = self._new_figure(style_cfg, figsize)
                ax = axes[0, 0]
            else:
                fig = ax.figure
            self._draw_qq(
                ax=ax,
                data=work,
                name=name,
                hue_name=hue_name,
                style=style_cfg,
                dist=dist,
                dist_params=dist_params,
                fit=fit,
                standardize=standardize,
                line=line,
                conf_band=conf_band,
                annotate_test=annotate_test,
                positions=plotting_positions,
                marker_kwargs=marker_kwargs,
                compare=self._column(frame, compare, "compare"),
                palette=palette,
                hue_order=hue_order,
                transform=transform,
                nonpositive=nonpositive,
                alpha=alpha,
                xlabel=xlabel,
                ylabel=ylabel,
                legend=legend,
                legend_loc=legend_loc,
                kwargs=kwargs,
            )
            self._panel_title(
                ax,
                style_cfg,
                self._auto_title(
                    panel_title,
                    None if owns_figure else f"Q-Q plot of {name}",
                ),
            )
            if extra_ax:
                ax.set(**extra_ax)
            if owns_figure:
                top = self._place_header(
                    fig,
                    style_cfg,
                    self._auto_title(title, f"Q-Q plot of {name}"),
                    subtitle,
                    style_cfg.margins,
                )
                self._finish(fig, style_cfg, {}, top, save_as, show)
        return ax

    @staticmethod
    def _plotting_position_a(convention: str) -> float:
        """Map a plotting-position convention onto its ``a`` value."""
        table = {
            "blom": 0.375,
            "median": 0.3175,
            "hazen": 0.5,
            "weibull": 0.0,
        }
        if convention not in table:
            raise ValueError(
                f"plotting_positions={convention!r} is not recognised. "
                f"Use one of {', '.join(sorted(table))}."
            )
        return table[convention]

    def _resolve_dist(
        self,
        dist: Any,
        data: np.ndarray,
        fit: bool,
        dist_params: Mapping[str, Any] | None,
    ) -> tuple[Any, tuple[Any, ...], str]:
        """Return a frozen distribution plus a description of its fit."""
        from scipy import stats as sps

        if isinstance(dist, str):
            if not hasattr(sps, dist):
                raise ValueError(
                    f"dist={dist!r} is not a scipy.stats distribution."
                )
            family = getattr(sps, dist)
            label = dist
        else:
            family = dist
            label = getattr(dist, "name", str(dist))
        if hasattr(family, "dist") and not hasattr(family, "fit"):
            return family, (), label  # already frozen

        shapes = tuple((dist_params or {}).values())
        if fit:
            params = family.fit(data, *shapes)
        elif shapes:
            params = (*shapes, 0.0, 1.0)
        else:
            params = (0.0, 1.0)
        detail = ", ".join(f"{p:.3g}" for p in params)
        return family(*params), params, f"{label}({detail})"

    def _draw_qq(
        self,
        *,
        ax: Axes,
        data: pd.DataFrame,
        name: str,
        hue_name: str | None,
        style: StyleConfig,
        dist: Any,
        dist_params: Mapping[str, Any] | None,
        fit: bool,
        standardize: bool,
        line: str | None,
        conf_band: float | None,
        annotate_test: str | Sequence[str] | None,
        positions: str,
        marker_kwargs: Mapping[str, Any] | None,
        compare: pd.Series | None,
        palette: Any,
        hue_order: Sequence[Any] | None,
        transform: Any,
        nonpositive: Nonpositive,
        alpha: float,
        xlabel: str | None,
        ylabel: str | None,
        legend: bool,
        legend_loc: str,
        kwargs: Mapping[str, Any],
    ) -> Axes:
        """Draw the QQ scatter, reference line and optional band."""
        a = self._plotting_position_a(positions)
        raw = pd.to_numeric(data[name], errors="coerce")
        values, meta = _apply_transform(
            raw, transform, nonpositive, column=name, lang=self.lang
        )
        self._last_transform = meta
        frame = data.assign(**{name: values})

        if compare is not None:
            return self._draw_qq_two_sample(
                ax=ax,
                sample=frame[name].dropna().to_numpy(float),
                other=pd.to_numeric(compare, errors="coerce")
                .dropna()
                .to_numpy(float),
                names=(meta.label_for(name), str(compare.name)),
                style=style,
                line=line,
                marker_kwargs=marker_kwargs,
                kwargs=kwargs,
            )

        groups: list[tuple[Any, np.ndarray, str]] = []
        if hue_name:
            levels = self.ordered_levels(frame[hue_name], hue_order)
            mapping = self.color_map(hue_name, levels, palette)
            for level in levels:
                sample = (
                    frame.loc[frame[hue_name] == level, name]
                    .dropna()
                    .to_numpy(float)
                )
                if sample.size < 8:
                    self._warn_once(
                        f"qq-small:{hue_name}:{level}",
                        f"group {level!r} has n = {sample.size} (< 8), "
                        f"too few points for a meaningful QQ line; it "
                        f"is skipped.",
                    )
                    continue
                groups.append((level, sample, mapping[level]))
        else:
            sample = frame[name].dropna().to_numpy(float)
            if sample.size < 8:
                raise ValueError(
                    f"qqplot needs at least 8 finite values, got "
                    f"{sample.size} for {name!r}."
                )
            groups.append((None, sample, style.categorical[0]))

        detail = ""
        for level, sample, colour in groups:
            if standardize:
                sample = (sample - sample.mean()) / (sample.std(ddof=1) or 1.0)
            ordered = np.sort(sample)
            n = ordered.size
            probs = (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)
            frozen, _params, detail = self._resolve_dist(
                dist, ordered, fit and not standardize, dist_params
            )
            theoretical = frozen.ppf(probs)
            label = f"{level} (n = {n:,})" if level is not None else None
            marker_opts: dict[str, Any] = {
                "s": 14,
                "color": colour,
                "edgecolor": "white",
                "linewidth": style.marker_edge_width,
                "zorder": style.data_zorder + 1,
                "alpha": 0.85,
            }
            marker_opts.update(dict(marker_kwargs or {}))
            if conf_band:
                lower, upper = self._qq_band(frozen, n, probs, conf_band)
                ax.fill_between(
                    theoretical,
                    lower,
                    upper,
                    color=colour,
                    alpha=0.12,
                    linewidth=0,
                    zorder=style.grid_zorder + 1,
                    label=_t(
                        self.lang,
                        "legend_band",
                        level=conf_band * 100,
                    )
                    if level is None
                    else None,
                )
                outside = (ordered < lower) | (ordered > upper)
                if outside.any():
                    marker_colors = np.where(outside, style.negative, colour)
                    marker_opts["color"] = list(marker_colors)
            ax.scatter(
                theoretical,
                ordered,
                label=label,
                **marker_opts,
                **dict(kwargs),
            )
            self._draw_qq_line(ax, theoretical, ordered, line, colour, style)

        ax.set_xlabel(
            xlabel
            if xlabel is not None
            else f"{_t(self.lang, 'axis_theoretical')} - {detail}"
        )
        ax.set_ylabel(
            ylabel
            if ylabel is not None
            else _t(self.lang, "axis_sample") + f" - {meta.label_for(name)}"
        )
        self._apply_grid(ax, style, "both")
        self._despine(ax, style)
        if annotate_test:
            self._annotate_tests(ax, groups, annotate_test, alpha, style)
        self._style_legend(ax, style, hue_name, legend, legend_loc)
        return ax

    def _draw_qq_line(
        self,
        ax: Axes,
        theoretical: np.ndarray,
        ordered: np.ndarray,
        line: str | None,
        colour: str,
        style: StyleConfig,
    ) -> None:
        """Add the requested reference line."""
        if line is None:
            return
        if line == "q":
            tq = np.quantile(theoretical, [0.25, 0.75])
            sq = np.quantile(ordered, [0.25, 0.75])
            slope = (sq[1] - sq[0]) / (tq[1] - tq[0] or 1.0)
            intercept = sq[0] - slope * tq[0]
        elif line == "45":
            slope, intercept = 1.0, 0.0
        elif line == "s":
            slope = float(ordered.std(ddof=1))
            intercept = float(ordered.mean())
        elif line == "r":
            slope, intercept = np.polyfit(theoretical, ordered, 1)
        else:
            raise ValueError(
                f"line={line!r} is not recognised. Use 'q', '45', 's', "
                f"'r' or None."
            )
        xs = np.array([theoretical.min(), theoretical.max()])
        ax.plot(
            xs,
            intercept + slope * xs,
            color=colour,
            linewidth=1.3,
            linestyle="-",
            zorder=style.data_zorder,
            alpha=0.9,
        )

    def _qq_band(
        self,
        frozen: Any,
        n: int,
        probs: np.ndarray,
        level: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise band around the reference line.

        Uses the order-statistic beta distribution, which is exactly the
        analytic band ``statsmodels`` draws, so no optional dependency
        is needed. Falls back to a parametric bootstrap if that fails.
        """
        from scipy import stats as sps

        tail = (1.0 - level) / 2.0
        try:
            ranks = np.arange(1, n + 1)
            lower_p = sps.beta.ppf(tail, ranks, n - ranks + 1)
            upper_p = sps.beta.ppf(1 - tail, ranks, n - ranks + 1)
            return frozen.ppf(lower_p), frozen.ppf(upper_p)
        except Exception:
            rng = np.random.default_rng(self.random_state)
            sims = np.sort(
                frozen.rvs(size=(1000, n), random_state=rng), axis=1
            )
            return (
                np.quantile(sims, tail, axis=0),
                np.quantile(sims, 1 - tail, axis=0),
            )

    def _draw_qq_two_sample(
        self,
        *,
        ax: Axes,
        sample: np.ndarray,
        other: np.ndarray,
        names: tuple[str, str],
        style: StyleConfig,
        line: str | None,
        marker_kwargs: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> Axes:
        """Empirical-versus-empirical QQ plot for two samples."""
        size = min(sample.size, other.size)
        probs = (np.arange(1, size + 1) - 0.5) / size
        left = np.quantile(other, probs)
        right = np.quantile(sample, probs)
        opts: dict[str, Any] = {
            "s": 14,
            "color": style.categorical[0],
            "edgecolor": "white",
            "linewidth": style.marker_edge_width,
            "zorder": style.data_zorder + 1,
        }
        opts.update(dict(marker_kwargs or {}))
        ax.scatter(left, right, **opts, **dict(kwargs))
        self._draw_qq_line(
            ax, left, right, line or "45", style.categorical[0], style
        )
        ax.set_xlabel(names[1])
        ax.set_ylabel(names[0])
        self._apply_grid(ax, style, "both")
        self._despine(ax, style)
        return ax

    def _annotate_tests(
        self,
        ax: Axes,
        groups: Sequence[tuple[Any, np.ndarray, str]],
        annotate_test: str | Sequence[str],
        alpha: float,
        style: StyleConfig,
    ) -> None:
        """Render test statistics in a frameless corner box."""
        wanted = (
            [annotate_test]
            if isinstance(annotate_test, str)
            else list(annotate_test)
        )
        lines: list[str] = []
        for level, sample, _colour in groups:
            prefix = f"{level}: " if level is not None else ""
            for test in wanted:
                outcome = self.run_normality_test(sample, test, alpha)
                if outcome.get("critical") is not None:
                    body = (
                        f"{outcome['test']} A2 = "
                        f"{outcome['statistic']:.3f} vs "
                        f"{outcome['critical']:.3f}"
                    )
                else:
                    body = (
                        f"{outcome['test']} = "
                        f"{outcome['statistic']:.3f}, "
                        f"{_format_p(outcome['p'])}"
                    )
                lines.append(prefix + body)
        ax.annotate(
            "\n".join(lines),
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=style.annot_size * 0.9,
            color=style.text_color,
        )

    # -- triple_plot -------------------------------------------------
    _LAYOUT_FIGSIZE: dict[str, tuple[float, float]] = {
        "row": (16.0, 5.0),
        "stacked": (14.0, 8.0),
        "grid": (14.0, 8.0),
        "column": (7.0, 14.0),
    }

    def triple_plot(
        self,
        df: pd.DataFrame | None = None,
        x: str | Sequence[str] | pd.Series | None = None,
        *,
        layout: str = "row",
        hue: str | pd.Series | None = None,
        transform: Any = None,
        nonpositive: Nonpositive = "raise",
        test: str = "auto",
        alpha: float = 0.05,
        lang: str | None = None,
        bins: Any = "fd",
        kde: bool = True,
        show_normal_fit: bool = True,
        conf_band: float | None = 0.95,
        show_stats: bool = True,
        stats_loc: str = "panel",
        title: str | bool | None = None,
        subtitle: str | None = None,
        panel_titles: Sequence[str] | None = None,
        annot_fmt: str = "{:,.4g}",
        return_stats: bool = False,
        figsize: tuple[float, float] | None = None,
        treat_as: Mapping[str, str] | None = None,
        style: Mapping[str, Any] | str | None = None,
        hist_kwargs: Mapping[str, Any] | None = None,
        box_kwargs: Mapping[str, Any] | None = None,
        qq_kwargs: Mapping[str, Any] | None = None,
        save_as: str | Path | None = None,
        show: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """One-call distribution diagnostic for a numeric variable.

        Draws a histogram, a boxplot and a normal Q-Q plot side by side,
        then reports a normality test with a plain-language verdict. The
        three panels are produced by calling :meth:`histplot`,
        :meth:`boxplot` and :meth:`qqplot` with an explicit ``ax=``, so
        there is no duplicated plotting logic and every option those
        methods expose stays reachable through ``hist_kwargs``,
        ``box_kwargs`` and ``qq_kwargs``.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Frame to plot.
        x : str, sequence of str or pandas.Series
            Numeric variable, or a list of them, in which case a list of
            figures is returned.
        layout : {"row", "stacked", "grid", "column"}, default "row"
            ``"stacked"`` puts the histogram above a horizontal boxplot
            sharing its x-axis, so outliers line up with the tail; it is
            the most informative arrangement.
        hue : str or pandas.Series, optional
            Categorical split, giving one row of three panels per level
            with shared x-limits. Capped at 4 levels.
        transform : str or callable, optional
            Plot the transformed variable in all three panels and report
            the normality test on **both** raw and transformed data, so
            it is visible whether the transform actually helped.
        test : str, default "auto"
            Normality test; ``"auto"`` switches from Shapiro-Wilk to
            D'Agostino-Pearson above n = 5000.
        alpha : float, default 0.05
            Significance level for the verdict.
        stats_loc : {"panel", "corner", "subtitle"}, default "panel"
            Where the statistics block goes.
        return_stats : bool, default False
            Return ``(figure, stats)`` instead of just the figure.
        lang : {"en", "es"}, optional
            Language of the interpretation text.

        Returns
        -------
        matplotlib.figure.Figure or tuple or list
            The figure; ``(figure, stats)`` when ``return_stats``; a
            list when ``x`` is a sequence.

        Raises
        ------
        ValueError
            When ``x`` is categorical, when n < 8, or when ``hue`` has
            more than 4 levels.

        Examples
        --------
        >>> import matplotlib, numpy as np, pandas as pd
        >>> matplotlib.use("Agg")
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"v": rng.normal(size=200)})
        >>> fig, stats = EDAPlotter(df).triple_plot(x="v", return_stats=True)
        >>> stats["n"]
        200
        """
        frame, style_cfg, treat, _ = self._resolve_common(
            df, style=style, figsize=figsize, treat_as=treat_as
        )
        if isinstance(x, (list, tuple)) and not isinstance(x, str):
            return [
                self.triple_plot(
                    frame,
                    x=single,
                    layout=layout,
                    hue=hue,
                    transform=transform,
                    nonpositive=nonpositive,
                    test=test,
                    alpha=alpha,
                    lang=lang,
                    bins=bins,
                    kde=kde,
                    conf_band=conf_band,
                    show_stats=show_stats,
                    stats_loc=stats_loc,
                    return_stats=return_stats,
                    figsize=figsize,
                    treat_as=treat,
                    style=style,
                    save_as=None,
                    show=show,
                    **kwargs,
                )
                for single in x
            ]

        target = self._column(frame, x, "x")
        if target is None:
            raise ValueError("triple_plot needs x=<numeric column>.")
        name = str(target.name)
        lang = lang or self.lang
        kind = self.resolve_kind(target, treat.get(name))
        numeric = pd.to_numeric(target, errors="coerce")
        if kind == "categorical" and not self._is_integer_like(numeric):
            raise ValueError(
                f"x={name!r} resolves to categorical, so a distribution "
                f"diagnostic does not apply. Use barplot for level "
                f"counts, or treat_as={{{name!r}: 'numeric'}}."
            )

        work = frame.copy()
        work[name] = numeric.to_numpy()
        hue_series = self._column(work, hue, "hue")
        hue_name = str(hue_series.name) if hue_series is not None else None
        if hue_name:
            work[hue_name] = hue_series.to_numpy()
            work = work.dropna(subset=[hue_name])
            levels = self.ordered_levels(work[hue_name])
            if len(levels) > 4:
                raise ValueError(
                    f"hue={hue_name!r} has {len(levels)} levels; "
                    f"triple_plot draws one row of three panels per "
                    f"level and caps at 4. Loop over the levels, or "
                    f"pass top_n through a filtered frame."
                )
        else:
            levels = [None]

        n_valid = int(work[name].notna().sum())
        if n_valid < 8:
            raise ValueError(
                f"triple_plot needs at least 8 non-null values, got "
                f"{n_valid} for {name!r}."
            )

        discrete = int(work[name].nunique(dropna=True)) <= 25 and bool(
            self._is_integer_like(work[name].dropna())
        )
        if discrete:
            self._warn_once(
                f"triple-discrete:{name}",
                f"{name!r} is discrete "
                f"({work[name].nunique()} distinct values): the "
                f"histogram switches to one bin per value, the KDE "
                f"overlay is disabled because it is meaningless on a "
                f"lattice, and ties will show as a staircase in the "
                f"Q-Q panel.",
            )

        size = figsize or self._LAYOUT_FIGSIZE.get(layout, (16.0, 5.0))
        if hue_name and len(levels) > 1:
            size = (size[0], size[1] * len(levels))

        collected: dict[Any, dict[str, Any]] = {}
        with self._style_scope(style_cfg):
            fig = plt.figure(figsize=size, dpi=self.dpi)
            grids = self._triple_layout(
                fig, layout, len(levels), show_stats, stats_loc
            )
            shared_limits = self._shared_limits(
                work[name], transform, nonpositive, name
            )
            for row, level in enumerate(levels):
                subset = (
                    work if level is None else work[work[hue_name] == level]
                )
                stats = self._draw_triple_row(
                    fig=fig,
                    axes=grids[row],
                    data=subset,
                    name=name,
                    style=style_cfg,
                    layout=layout,
                    level=level,
                    lang=lang,
                    transform=transform,
                    nonpositive=nonpositive,
                    test=test,
                    alpha=alpha,
                    bins=bins,
                    kde=kde and not discrete,
                    discrete=discrete,
                    show_normal_fit=show_normal_fit,
                    conf_band=conf_band,
                    show_stats=show_stats,
                    stats_loc=stats_loc,
                    panel_titles=panel_titles,
                    annot_fmt=annot_fmt,
                    shared_limits=shared_limits,
                    hist_kwargs=hist_kwargs,
                    box_kwargs=box_kwargs,
                    qq_kwargs=qq_kwargs,
                    kwargs=kwargs,
                )
                collected[level] = stats

            head = collected[levels[0]]
            default_sub = _t(
                lang,
                "n_missing",
                n=head["n"],
                missing=head["missing"],
                pct=head["missing_pct"],
            )
            if head.get("transform_note"):
                default_sub = f"{default_sub} - {head['transform_note']}"
            resolved_sub = (
                None
                if stats_loc == "subtitle" and show_stats
                else (subtitle if subtitle is not None else default_sub)
            )
            if stats_loc == "subtitle" and show_stats:
                resolved_sub = self._stats_one_line(head, lang, annot_fmt)
            top = self._place_header(
                fig,
                style_cfg,
                self._auto_title(title, name),
                resolved_sub,
                style_cfg.margins,
            )
            fig.subplots_adjust(
                top=top,
                left=0.08,
                right=0.96,
                bottom=0.12,
                wspace=0.28,
                hspace=0.55,
            )
            self._save(fig, save_as)
            if self.show if show is None else show:
                plt.show()

        if return_stats:
            payload = collected[None] if levels == [None] else collected
            return fig, payload
        return fig

    #: ``distribution_report`` is an alias of :meth:`triple_plot`.
    distribution_report = triple_plot

    def _triple_layout(
        self,
        fig: Figure,
        layout: str,
        n_rows: int,
        show_stats: bool,
        stats_loc: str,
    ) -> list[dict[str, Axes]]:
        """Build the axes for each row of panels."""
        rows: list[dict[str, Axes]] = []
        wants_panel = (
            show_stats and stats_loc == "panel" and (layout == "grid")
        )
        for row in range(n_rows):
            if layout == "row":
                spec = fig.add_gridspec(n_rows, 3, left=0.08, right=0.96)
                axes = {
                    "hist": fig.add_subplot(spec[row, 0]),
                    "box": fig.add_subplot(spec[row, 1]),
                    "qq": fig.add_subplot(spec[row, 2]),
                }
            elif layout == "column":
                spec = fig.add_gridspec(n_rows * 3, 1)
                axes = {
                    "hist": fig.add_subplot(spec[row * 3, 0]),
                    "box": fig.add_subplot(spec[row * 3 + 1, 0]),
                    "qq": fig.add_subplot(spec[row * 3 + 2, 0]),
                }
            elif layout == "stacked":
                spec = fig.add_gridspec(
                    n_rows * 2, 2, height_ratios=[3, 1] * n_rows
                )
                hist_ax = fig.add_subplot(spec[row * 2, 0])
                axes = {
                    "hist": hist_ax,
                    "box": fig.add_subplot(
                        spec[row * 2 + 1, 0], sharex=hist_ax
                    ),
                    "qq": fig.add_subplot(spec[row * 2 : row * 2 + 2, 1]),
                }
            elif layout == "grid":
                spec = fig.add_gridspec(n_rows * 2, 2)
                axes = {
                    "hist": fig.add_subplot(spec[row * 2, 0]),
                    "box": fig.add_subplot(spec[row * 2, 1]),
                    "qq": fig.add_subplot(spec[row * 2 + 1, 0]),
                }
                if wants_panel:
                    text_ax = fig.add_subplot(spec[row * 2 + 1, 1])
                    text_ax.axis("off")
                    axes["stats"] = text_ax
            else:
                raise ValueError(
                    f"layout={layout!r} is not recognised. Use 'row', "
                    f"'stacked', 'grid' or 'column'."
                )
            rows.append(axes)
        return rows

    def _shared_limits(
        self,
        series: pd.Series,
        transform: Any,
        nonpositive: Nonpositive,
        name: str,
    ) -> tuple[float, float] | None:
        """Common x-limits so hue rows stay visually comparable."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            values, _meta = _apply_transform(
                series,
                transform,
                nonpositive,
                column=name,
                lang=self.lang,
            )
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        return float(finite.min()), float(finite.max())

    def _draw_triple_row(
        self,
        *,
        fig: Figure,
        axes: Mapping[str, Axes],
        data: pd.DataFrame,
        name: str,
        style: StyleConfig,
        layout: str,
        level: Any,
        lang: str,
        transform: Any,
        nonpositive: Nonpositive,
        test: str,
        alpha: float,
        bins: Any,
        kde: bool,
        discrete: bool,
        show_normal_fit: bool,
        conf_band: float | None,
        show_stats: bool,
        stats_loc: str,
        panel_titles: Sequence[str] | None,
        annot_fmt: str,
        shared_limits: tuple[float, float] | None,
        hist_kwargs: Mapping[str, Any] | None,
        box_kwargs: Mapping[str, Any] | None,
        qq_kwargs: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Draw one row of three panels and compute its statistics."""
        titles = list(
            panel_titles
            or (
                _t(lang, "panel_hist"),
                _t(lang, "panel_box"),
                _t(lang, "panel_qq"),
            )
        )
        if level is not None:
            titles = [f"{t} - {level}" for t in titles]

        raw = pd.to_numeric(data[name], errors="coerce")
        values, meta = _apply_transform(
            raw, transform, nonpositive, column=name, lang=lang
        )
        self._last_transform = meta
        panel_data = data.assign(**{name: values})
        label = meta.label_for(name)

        self.histplot(
            panel_data,
            x=name,
            ax=axes["hist"],
            bins=bins,
            kde=kde,
            discrete=discrete,
            reference_lines=["mean", "median"],
            panel_title=titles[0],
            xlabel=label,
            legend=True,
            treat_as={name: "numeric"},
            style={"enabled": style.enabled},
            **dict(hist_kwargs or {}),
        )
        if show_normal_fit and not discrete:
            self._overlay_normal_fit(axes["hist"], values, style, lang)
        if shared_limits and layout != "column":
            axes["hist"].set_xlim(*self._pad(shared_limits))

        box_opts: dict[str, Any] = {
            "showmeans": True,
            "showfliers": True,
        }
        n_points = int(np.isfinite(values).sum())
        if n_points <= 500:
            box_opts["overlay"] = "strip"
            box_opts["showfliers"] = False
        box_opts.update(dict(box_kwargs or {}))
        self.boxplot(
            panel_data,
            x=name,
            ax=axes["box"],
            panel_title=titles[1],
            xlabel=label,
            legend=False,
            treat_as={name: "numeric"},
            style={"enabled": style.enabled},
            **box_opts,
        )
        if shared_limits and layout != "column":
            axes["box"].set_xlim(*self._pad(shared_limits))
        self._annotate_whiskers(
            axes["box"], values, box_opts.get("whis", 1.5), style, lang
        )

        self.qqplot(
            panel_data,
            x=name,
            ax=axes["qq"],
            dist="norm",
            fit=True,
            line="q",
            conf_band=conf_band,
            panel_title=titles[2],
            legend=False,
            treat_as={name: "numeric"},
            style={"enabled": style.enabled},
            **dict(qq_kwargs or {}),
            **dict(kwargs),
        )

        stats = self.describe_distribution(values, test, alpha)
        stats["column"] = name
        stats["level"] = level
        stats["transform"] = meta.name
        if meta.lam is not None:
            stats["lambda"] = meta.lam
            stats["transform_note"] = _t(
                lang, "lambda_note", name=meta.label, lam=meta.lam
            )
        if meta.name != "identity":
            raw_stats = self.describe_distribution(
                raw.dropna().to_numpy(float), test, alpha
            )
            stats["raw_p_value"] = raw_stats["p_value"]
            stats["raw_statistic"] = raw_stats["statistic"]
            stats["transform_note"] = _t(
                lang,
                "raw_vs_transformed",
                test=raw_stats["test"],
                raw=_p_compact(raw_stats["p_value"]),
                new=_p_compact(stats["p_value"]),
                name=meta.name,
            )
        if show_stats and stats_loc != "subtitle":
            self._render_stats_block(
                axes.get("stats", axes["qq"]),
                stats,
                style,
                lang,
                annot_fmt,
                dedicated="stats" in axes,
            )
        return stats

    @staticmethod
    def _pad(limits: tuple[float, float]) -> tuple[float, float]:
        """Widen a range by 3% so markers are not clipped."""
        lo, hi = limits
        span = (hi - lo) or 1.0
        return lo - 0.03 * span, hi + 0.03 * span

    def _overlay_normal_fit(
        self,
        ax: Axes,
        values: np.ndarray,
        style: StyleConfig,
        lang: str,
    ) -> None:
        """Draw the fitted normal PDF, scaled to the histogram."""
        from scipy import stats as sps

        finite = values[np.isfinite(values)]
        if finite.size < 3:
            return
        mu = float(finite.mean())
        sigma = float(finite.std(ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            return
        lo, hi = ax.get_xlim()
        grid = np.linspace(lo, hi, 200)
        density = sps.norm.pdf(grid, mu, sigma)
        patches = [p for p in ax.patches if p.get_height() > 0]
        if patches:
            widths = [p.get_width() for p in patches]
            density = density * finite.size * float(np.median(widths))
        ax.plot(
            grid,
            density,
            color=style.subtitle_color,
            linewidth=1.2,
            linestyle="-",
            label=_t(lang, "legend_normal"),
            zorder=style.data_zorder + 2,
        )
        ax.legend(frameon=False, fontsize=style.tick_size * 0.9)

    def _annotate_whiskers(
        self,
        ax: Axes,
        values: np.ndarray,
        whis: Any,
        style: StyleConfig,
        lang: str,
    ) -> None:
        """Note the whisker convention and the outlier count."""
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        if isinstance(whis, (tuple, list)):
            low, high = np.percentile(finite, list(whis))
            text_whis = f"{whis[0]}-{whis[1]} pct"
        else:
            q1, q3 = np.percentile(finite, [25, 75])
            iqr = q3 - q1
            low, high = q1 - whis * iqr, q3 + whis * iqr
            text_whis = f"{whis:g}"
        n_out = int(((finite < low) | (finite > high)).sum())
        ax.annotate(
            _t(lang, "whisker_note", whis=text_whis, n_out=n_out),
            xy=(0.0, -0.32),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=style.annot_size * 0.85,
            color=style.subtitle_color,
        )

    def _stat_rows(
        self, stats: Mapping[str, Any], lang: str, fmt: str
    ) -> list[tuple[str, str]]:
        """Label/value pairs for the statistics block."""

        def render(value: Any) -> str:
            if value is None or (
                isinstance(value, float) and not np.isfinite(value)
            ):
                return "n/a"
            if isinstance(value, (int, np.integer)):
                return f"{int(value):,}"
            return fmt.format(value)

        return [
            (_t(lang, "stat_n"), f"{stats['n']:,}"),
            (
                _t(lang, "stat_missing"),
                f"{stats['missing']:,} ({stats['missing_pct']:.1f}%)",
            ),
            (_t(lang, "stat_mean"), render(stats["mean"])),
            (_t(lang, "stat_median"), render(stats["median"])),
            (_t(lang, "stat_std"), render(stats["std"])),
            (_t(lang, "stat_iqr"), render(stats["iqr"])),
            (_t(lang, "stat_min"), render(stats["min"])),
            (_t(lang, "stat_max"), render(stats["max"])),
            (_t(lang, "stat_skew"), render(stats["skew"])),
            (_t(lang, "stat_kurtosis"), render(stats["kurtosis"])),
        ]

    def _stats_one_line(
        self, stats: Mapping[str, Any], lang: str, fmt: str
    ) -> str:
        """Compress the statistics block into a single subtitle line."""
        pairs = self._stat_rows(stats, lang, fmt)
        return " - ".join(f"{k}: {v}" for k, v in pairs)

    def _render_stats_block(
        self,
        ax: Axes,
        stats: Mapping[str, Any],
        style: StyleConfig,
        lang: str,
        fmt: str,
        dedicated: bool,
    ) -> None:
        """Draw the statistics table and the colour-coded verdict."""
        verdict, reject = self.verdict_text(stats, lang)
        accent = style.negative if reject else style.positive
        note = stats.get("transform_note")

        if not dedicated:
            body = verdict if not note else f"{note}\n{verdict}"
            ax.annotate(
                body,
                xy=(0.03, 0.97),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=style.annot_size * 0.82,
                color=accent,
                linespacing=1.35,
            )
            return

        pairs = self._stat_rows(stats, lang, fmt)
        ax.text(
            0.0,
            1.0,
            _t(lang, "panel_stats"),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=style.panel_title_size,
            fontweight="bold",
            color=style.panel_title_color,
        )
        table = "\n".join(f"{k:<18}{v}" for k, v in pairs)
        ax.text(
            0.0,
            0.90,
            table,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=style.annot_size * 0.95,
            color=style.text_color,
            family="monospace",
            linespacing=1.45,
        )
        body = verdict if not note else f"{note}\n{verdict}"
        ax.text(
            0.02,
            0.30,
            body,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=style.annot_size * 0.88,
            color=accent,
            linespacing=1.4,
        )
        # A thin left rule rather than a filled box, so the panel keeps
        # the flat look while still reading as a verdict. Drawn with
        # plot() in axes coordinates: axvline builds its own transform
        # and refuses one.
        ax.plot(
            [0.0, 0.0],
            [0.0, 0.32],
            color=accent,
            linewidth=2.5,
            transform=ax.transAxes,
            clip_on=False,
        )

    def report_numeric(
        self,
        df: pd.DataFrame | None = None,
        columns: Sequence[str] | None = None,
        *,
        include_low_cardinality: bool = False,
        treat_as: Mapping[str, str] | None = None,
        save_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Figure]:
        """Run :meth:`triple_plot` over every numeric column.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Frame to report on.
        columns : sequence of str, optional
            Restrict to these columns; defaults to every column that
            :meth:`resolve_kind` calls numeric.
        include_low_cardinality : bool, default False
            Include numeric columns that rule 4 reclassified as
            categorical.
        save_dir : str or pathlib.Path, optional
            Write ``<column>.png`` into this directory.

        Returns
        -------
        dict
            Column name to figure.

        Examples
        --------
        >>> import matplotlib, numpy as np, pandas as pd
        >>> matplotlib.use("Agg")
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"v": rng.normal(size=40)})
        >>> list(EDAPlotter(df).report_numeric())
        ['v']
        """
        frame = self._frame(df)
        treat = dict(treat_as or {})
        if columns is None:
            columns = [
                col
                for col in frame.columns
                if self.resolve_kind(frame[col], treat.get(col)) == "numeric"
                or (
                    include_low_cardinality
                    and is_numeric_dtype(frame[col].dtype)
                )
            ]
        figures: dict[str, Figure] = {}
        for column in columns:
            target = (
                Path(save_dir) / f"{column}.png"
                if save_dir is not None
                else None
            )
            try:
                figures[column] = self.triple_plot(
                    frame,
                    x=column,
                    treat_as=treat,
                    save_as=target,
                    **kwargs,
                )
            except ValueError as exc:
                warnings.warn(
                    f"report_numeric skipped {column!r}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
        return figures

    # -- barplot -----------------------------------------------------
    _ESTIMATORS: dict[str, Callable[[Any], float]] = {
        "mean": np.mean,
        "median": np.median,
        "sum": np.sum,
        "count": len,
        "nunique": lambda a: len(np.unique(a)),
        "min": np.min,
        "max": np.max,
    }

    def barplot(
        self,
        df: pd.DataFrame | None = None,
        x: str | pd.Series | None = None,
        y: str | pd.Series | None = None,
        hue: str | pd.Series | None = None,
        *,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | bool | None = None,
        subtitle: str | None = None,
        panel_title: str | bool | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        palette: str | list | dict | None = None,
        legend: bool = True,
        legend_loc: str = "best",
        order: Sequence[Any] | None = None,
        hue_order: Sequence[Any] | None = None,
        top_n: int | None = None,
        log_scale: Any = None,
        log_x: bool = False,
        log_y: bool = False,
        transform: Any = None,
        nonpositive: Nonpositive = "raise",
        facet: str | None = None,
        facet_col_wrap: int = 3,
        sharex: bool = True,
        sharey: bool = True,
        dropna: bool = True,
        treat_as: Mapping[str, str] | None = None,
        orientation: str | None = None,
        estimator: str | Callable[[Any], float] = "mean",
        errorbar: str | Callable | None = "ci",
        ci_level: float = 95,
        stacked: bool = False,
        normalize: bool | str = False,
        annotate: bool = False,
        annot_fmt: str = "{:,.2f}",
        sort: str | None = "value",
        ascending: bool = False,
        baseline: float | str | None = None,
        fill_missing: float | None = None,
        rotate_xticks: int | str | None = "auto",
        xticklabels: Mapping[Any, str] | Sequence[str] | None = None,
        yticklabels: Mapping[Any, str] | Sequence[str] | None = None,
        style: Mapping[str, Any] | str | None = None,
        save_as: str | Path | None = None,
        show: bool | None = None,
        ax_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Axes | Figure:
        """Compare a measure across categorical levels, or count levels.

        Input resolution is deliberately flexible:

        =============== =============== ==========================
        ``x``           ``y``           behaviour
        =============== =============== ==========================
        categorical     numeric         aggregate ``y`` per level
        numeric         categorical     same, drawn horizontally
        categorical     ``None``        count plot
        categorical     categorical     cross-tab, like ``hue``
        numeric         numeric         ``ValueError``
        =============== =============== ==========================

        A numeric column with few distinct values is a valid category
        axis, so a 1-5 satisfaction score works without an override.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Frame to plot.
        x, y : str or pandas.Series, optional
            Category and value variables, in either order.
        hue : str or pandas.Series, optional
            Second categorical split, resolved through
            :meth:`resolve_kind` so a low-cardinality numeric column
            works as classes.
        estimator : str or callable, default "mean"
            ``"mean"``, ``"median"``, ``"sum"``, ``"count"``,
            ``"nunique"``, ``"min"``, ``"max"``, or any callable
            mapping an array to a scalar.
        errorbar : str, callable or None, default "ci"
            ``"ci"`` (bootstrap at ``ci_level``), ``"se"``, ``"sd"``,
            ``"pi"``, a callable, or ``None``. Suppressed automatically
            for ``"count"`` and ``"sum"``, which have no spread.
        stacked : bool, default False
            Stack ``hue`` levels instead of dodging them.
        normalize : bool or {"row", "column"}, default False
            With ``stacked``, draw 100% stacked bars on a percent axis.
        sort : {"value", "alpha", "natural", None}, default "value"
            Level ordering; an explicit ``order`` always wins and may be
            a subset, which filters the data.
        orientation : {"vertical", "horizontal"}, optional
            Horizontal is recommended for long labels or more than
            about eight levels; it also reverses the level order so the
            largest bar sits at the top.
        annotate : bool, default False
            Write each value at the end of its bar.
        baseline : float or str, optional
            Draw a labelled reference line, e.g. the global mean.

        Returns
        -------
        matplotlib.axes.Axes or matplotlib.figure.Figure
            The axes for a single panel, the figure when faceting.

        Raises
        ------
        ValueError
            When both variables are continuous numeric, or when a log
            axis is requested and an aggregated value is not positive.

        Examples
        --------
        >>> import matplotlib, pandas as pd
        >>> matplotlib.use("Agg")
        >>> df = pd.DataFrame({"g": list("aabb"), "v": [1.5, 3.25, 2.75, 4.1]})
        >>> ax = EDAPlotter(df).barplot(x="g", y="v")
        >>> ax.get_ylabel()
        'mean(v)'
        """
        frame, style_cfg, treat, extra_ax = self._resolve_common(
            df,
            style=style,
            figsize=figsize,
            treat_as=treat_as,
            ax=ax,
            facet=facet,
            ax_kwargs=ax_kwargs,
        )
        cat, value, hue_extra, orient = self._resolve_bar_axes(
            frame, x, y, orientation, treat
        )
        cat_name = str(cat.name)
        value_name = str(value.name) if value is not None else None
        hue_spec = hue if hue is not None else hue_extra

        work = frame.copy()
        work[cat_name] = cat.to_numpy()
        if value_name:
            work[value_name] = value.to_numpy()
        hue_series = self._column(work, hue_spec, "hue")
        hue_name = str(hue_series.name) if hue_series is not None else None
        if hue_name:
            work[hue_name] = hue_series.to_numpy()
        work = self._dropna_subset(
            work,
            [cat_name, value_name, hue_name, facet],
            dropna,
            f"barplot({cat_name!r})",
        )
        work[cat_name] = self._limit_levels(work[cat_name], top_n)
        work = self._filter_small_groups(
            work, cat_name, f"barplot({cat_name!r})"
        )
        if hue_name:
            work[hue_name] = self._limit_levels(work[hue_name], None)

        if facet is not None:
            return self._facet(
                work,
                facet,
                lambda panel_ax, sub, level: self.barplot(
                    sub,
                    x=cat_name if orient == "vertical" else value_name,
                    y=value_name if orient == "vertical" else cat_name,
                    hue=hue_name,
                    ax=panel_ax,
                    orientation=orient,
                    estimator=estimator,
                    errorbar=errorbar,
                    stacked=stacked,
                    normalize=normalize,
                    annotate=annotate,
                    sort=sort,
                    legend=legend,
                    treat_as=treat,
                    title=False,
                    panel_title=False,
                    style=style,
                    **kwargs,
                ),
                style=style_cfg,
                col_wrap=facet_col_wrap,
                sharex=sharex,
                sharey=sharey,
                figsize=figsize,
                title=self._auto_title(
                    title,
                    self._bar_title(cat_name, value_name, estimator),
                ),
                subtitle=subtitle,
                save_as=save_as,
                show=show,
            )

        owns_figure = ax is None
        with self._style_scope(style_cfg):
            if owns_figure:
                fig, axes = self._new_figure(style_cfg, figsize)
                ax = axes[0, 0]
            else:
                fig = ax.figure
            self._draw_bar(
                ax=ax,
                data=work,
                cat_name=cat_name,
                value_name=value_name,
                hue_name=hue_name,
                orient=orient,
                style=style_cfg,
                estimator=estimator,
                errorbar=errorbar,
                ci_level=ci_level,
                stacked=stacked,
                normalize=normalize,
                annotate=annotate,
                annot_fmt=annot_fmt,
                sort=sort,
                ascending=ascending,
                order=order,
                hue_order=hue_order,
                palette=palette,
                baseline=baseline,
                fill_missing=fill_missing,
                log_scale=log_scale,
                log_x=log_x,
                log_y=log_y,
                transform=transform,
                nonpositive=nonpositive,
                xlabel=xlabel,
                ylabel=ylabel,
                legend=legend,
                legend_loc=legend_loc,
                rotate_xticks=rotate_xticks,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                kwargs=kwargs,
            )
            self._panel_title(
                ax,
                style_cfg,
                self._auto_title(
                    panel_title,
                    None
                    if owns_figure
                    else self._bar_title(cat_name, value_name, estimator),
                ),
            )
            if extra_ax:
                ax.set(**extra_ax)
            if owns_figure:
                top = self._place_header(
                    fig,
                    style_cfg,
                    self._auto_title(
                        title,
                        self._bar_title(cat_name, value_name, estimator),
                    ),
                    subtitle,
                    style_cfg.margins,
                )
                self._finish(fig, style_cfg, {}, top, save_as, show)
        return ax

    @staticmethod
    def _estimator_name(estimator: Any) -> str:
        """Readable name for an estimator, callable or not."""
        if isinstance(estimator, str):
            return estimator
        return getattr(estimator, "__name__", "f")

    def _bar_title(
        self, cat_name: str, value_name: str | None, estimator: Any
    ) -> str:
        """Default figure title for a bar chart."""
        if value_name is None:
            return f"Distribution of {cat_name}"
        return f"{self._estimator_name(estimator)}({value_name}) by {cat_name}"

    def _resolve_bar_axes(
        self,
        frame: pd.DataFrame,
        x: Any,
        y: Any,
        orientation: str | None,
        treat: Mapping[str, str],
    ) -> tuple[pd.Series, pd.Series | None, Any, str]:
        """Decide which axis carries categories and which the values."""
        xs = self._column(frame, x, "x")
        ys = self._column(frame, y, "y")
        if xs is None and ys is None:
            raise ValueError("barplot needs at least x= or y=.")
        if xs is None:
            xs, ys = ys, None
        kind_x = self.resolve_kind(xs, treat.get(str(xs.name)))
        if ys is None:
            if kind_x != "categorical":
                self._warn_once(
                    f"bar-numeric:{xs.name}",
                    f"x={xs.name!r} is continuous numeric; counting its "
                    f"raw values rarely reads well. Consider histplot, "
                    f"or bin it first.",
                )
            return xs, None, None, orientation or "vertical"

        kind_y = self.resolve_kind(ys, treat.get(str(ys.name)))
        if kind_x == "categorical" and kind_y == "numeric":
            return xs, ys, None, orientation or "vertical"
        if kind_x == "numeric" and kind_y == "categorical":
            if orientation is None:
                self._warn_once(
                    f"bar-orient:{ys.name}",
                    f"y={ys.name!r} is the categorical axis, so the "
                    f"bars are drawn horizontally. Pass "
                    f"orientation='vertical' to override.",
                )
            return ys, xs, None, orientation or "horizontal"
        if kind_x == "categorical" and kind_y == "categorical":
            # Cross-tab: counts of y within x, exactly like hue.
            return xs, None, ys, orientation or "vertical"
        raise ValueError(
            f"x={xs.name!r} and y={ys.name!r} are both continuous "
            f"numeric, so neither can group the other. Use "
            f"scatterplot, histplot, or bin one of them first."
        )

    def _aggregate_bars(
        self,
        data: pd.DataFrame,
        cat_name: str,
        value_name: str | None,
        hue_name: str | None,
        estimator: Any,
        errorbar: Any,
        ci_level: float,
    ) -> pd.DataFrame:
        """Aggregate with groupby; no per-row Python loops."""
        keys = [cat_name] + ([hue_name] if hue_name else [])
        if value_name is None:
            frame = (
                data.groupby(keys, observed=True)
                .size()
                .reset_index(name="value")
            )
            frame["low"] = np.nan
            frame["high"] = np.nan
            return frame

        if isinstance(estimator, str) and estimator not in self._ESTIMATORS:
            raise ValueError(
                f"estimator={estimator!r} is not recognised. Use one of "
                f"{', '.join(sorted(self._ESTIMATORS))}, or a callable."
            )
        func = (
            self._ESTIMATORS[estimator]
            if isinstance(estimator, str)
            else estimator
        )
        grouped = data.groupby(keys, observed=True)[value_name]
        frame = (
            grouped.agg(lambda s: float(func(s.to_numpy(dtype=float))))
            .rename("value")
            .reset_index()
        )

        no_spread = isinstance(estimator, str) and estimator in {
            "count",
            "sum",
        }
        if errorbar is None or no_spread:
            frame["low"] = np.nan
            frame["high"] = np.nan
            return frame

        # One interval per group, not per row: the number of groups is
        # small and a bootstrap is inherently per-group.
        rows: list[tuple[Any, ...]] = []
        for key, series in grouped:
            low, high = self._interval(
                series.to_numpy(dtype=float), errorbar, ci_level, func
            )
            head = key if isinstance(key, tuple) else (key,)
            rows.append((*head, low, high))
        bounds = pd.DataFrame(rows, columns=[*keys, "low", "high"])
        return frame.merge(bounds, on=keys, how="left")

    def _interval(
        self,
        values: np.ndarray,
        errorbar: Any,
        ci_level: float,
        estimator: Callable[[Any], float],
    ) -> tuple[float, float]:
        """Compute one error-bar interval for a single group."""
        values = values[np.isfinite(values)]
        if values.size == 0:
            return (np.nan, np.nan)
        centre = float(estimator(values))
        if callable(errorbar):
            return tuple(errorbar(values))  # type: ignore[return-value]
        if errorbar == "sd":
            spread = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            return centre - spread, centre + spread
        if errorbar == "se":
            spread = (
                float(np.std(values, ddof=1) / np.sqrt(values.size))
                if values.size > 1
                else 0.0
            )
            return centre - spread, centre + spread
        tail = (100 - ci_level) / 2
        if errorbar == "pi":
            return (
                float(np.percentile(values, tail)),
                float(np.percentile(values, 100 - tail)),
            )
        if errorbar != "ci":
            raise ValueError(
                f"errorbar={errorbar!r} is not recognised. Use 'ci', "
                f"'se', 'sd', 'pi', a callable, or None."
            )
        if values.size < 2:
            return (centre, centre)
        rng = np.random.default_rng(self.random_state)
        picks = rng.integers(0, values.size, size=(1000, values.size))
        boots = np.apply_along_axis(estimator, 1, values[picks])
        return (
            float(np.percentile(boots, tail)),
            float(np.percentile(boots, 100 - tail)),
        )

    def _draw_bar(
        self,
        *,
        ax: Axes,
        data: pd.DataFrame,
        cat_name: str,
        value_name: str | None,
        hue_name: str | None,
        orient: str,
        style: StyleConfig,
        estimator: Any,
        errorbar: Any,
        ci_level: float,
        stacked: bool,
        normalize: bool | str,
        annotate: bool,
        annot_fmt: str,
        sort: str | None,
        ascending: bool,
        order: Sequence[Any] | None,
        hue_order: Sequence[Any] | None,
        palette: Any,
        baseline: float | str | None,
        fill_missing: float | None,
        log_scale: Any,
        log_x: bool,
        log_y: bool,
        transform: Any,
        nonpositive: Nonpositive,
        xlabel: str | None,
        ylabel: str | None,
        legend: bool,
        legend_loc: str,
        rotate_xticks: int | str | None,
        xticklabels: Any,
        yticklabels: Any,
        kwargs: Mapping[str, Any],
    ) -> Axes:
        """Aggregate, then draw the bars into ``ax``."""
        value_axis = "y" if orient == "vertical" else "x"
        cat_axis = "x" if orient == "vertical" else "y"
        log_spec = self._normalise_log_scale(
            log_scale, value_axis, log_x, log_y
        )
        self._check_scale_conflict(
            log_spec, transform, value_axis, value_name or cat_name
        )
        if cat_axis in log_spec:
            raise ValueError(
                f"log_scale targets the {cat_axis!r} axis, which holds "
                f"the categories. It only applies to the value axis."
            )

        work = data
        label_suffix = ""
        if transform is not None and value_name:
            values, meta = _apply_transform(
                data[value_name],
                transform,
                nonpositive,
                column=value_name,
                lang=self.lang,
            )
            self._last_transform = meta
            work = data.assign(**{value_name: values})
            label_suffix = meta.label_for(value_name)

        if stacked and isinstance(estimator, str) and estimator == "mean":
            self._warn_once(
                f"bar-stackmean:{cat_name}",
                "stacked=True with estimator='mean' stacks averages, "
                "which rarely means anything. Stacking is intended for "
                "'sum' or 'count'.",
            )
        agg = self._aggregate_bars(
            work,
            cat_name,
            value_name,
            hue_name,
            estimator,
            errorbar,
            ci_level,
        )
        # Rank from the aggregate, not from the row-level column: the
        # two have different lengths, so groupby would not align.
        base_sort = None if sort in {"value", "median"} else sort
        levels = self.ordered_levels(work[cat_name], order, base_sort)
        if sort in {"value", "median"} and order is None:
            ranked = (
                agg.groupby(cat_name, observed=True)["value"]
                .mean()
                .sort_values(ascending=ascending)
            )
            keep = set(levels)
            levels = [lv for lv in ranked.index if lv in keep]
        if orient == "horizontal" and sort in {"value", "median"}:
            levels = list(reversed(levels))
        agg = agg[agg[cat_name].isin(levels)]

        hue_levels = (
            self.ordered_levels(work[hue_name], hue_order) if hue_name else []
        )
        if normalize and hue_name:
            agg = self._normalize_bars(agg, cat_name, normalize)

        mapping = self.color_map(
            hue_name or cat_name,
            hue_levels if hue_name else levels,
            palette,
        )
        if stacked and hue_name:
            self._draw_stacked_bars(
                ax=ax,
                agg=agg,
                cat_name=cat_name,
                hue_name=hue_name,
                levels=levels,
                hue_levels=hue_levels,
                mapping=mapping,
                orient=orient,
                style=style,
                fill_missing=fill_missing,
                annotate=annotate,
                annot_fmt=annot_fmt,
                kwargs=kwargs,
            )
        else:
            self._draw_grouped_bars(
                ax=ax,
                agg=agg,
                cat_name=cat_name,
                hue_name=hue_name,
                levels=levels,
                hue_levels=hue_levels,
                mapping=mapping,
                orient=orient,
                style=style,
                errorbar=errorbar,
                annotate=annotate,
                annot_fmt=annot_fmt,
                kwargs=kwargs,
            )

        if log_spec:
            positive = agg["value"].to_numpy(dtype=float)
            if np.nanmin(positive) <= 0:
                raise ValueError(
                    f"log_scale needs every aggregated value to be > 0, "
                    f"but the smallest is {np.nanmin(positive):g}. Drop "
                    f"the log axis or use transform= instead."
                )
            self._warn_once(
                f"bar-log:{cat_name}",
                "A log-scaled bar chart has a misleading baseline: the "
                "bars no longer start at zero, so bar length stops "
                "encoding magnitude. Consider a dot/lollipop chart, or "
                "transform= instead.",
            )
            self._apply_log_axis(ax, log_spec, style)

        self._apply_grid(ax, style, value_axis)
        self._despine(ax, style)
        default_value_label = (
            _t(self.lang, "axis_count")
            if value_name is None
            else label_suffix
            or f"{self._estimator_name(estimator)}({value_name})"
        )
        if normalize and hue_name:
            default_value_label = "%"
        if orient == "vertical":
            ax.set_xlabel(xlabel if xlabel is not None else cat_name)
            ax.set_ylabel(
                ylabel if ylabel is not None else default_value_label
            )
        else:
            ax.set_ylabel(ylabel if ylabel is not None else cat_name)
            ax.set_xlabel(
                xlabel if xlabel is not None else default_value_label
            )
        if value_axis not in log_spec:
            self._format_value_axis(
                ax, value_axis, style, percent=bool(normalize and hue_name)
            )
        self._relabel_ticks(
            ax,
            xticklabels if cat_axis == "x" else yticklabels,
            levels,
            cat_axis,
        )
        if cat_axis == "x":
            self._rotate_ticks(ax, levels, rotate_xticks)
        if baseline is not None:
            self._draw_baseline(ax, work, value_name, baseline, orient, style)
        self._style_legend(ax, style, hue_name, legend, legend_loc)
        return ax

    @staticmethod
    def _normalize_bars(
        agg: pd.DataFrame, cat_name: str, normalize: bool | str
    ) -> pd.DataFrame:
        """Rescale each group to percentages."""
        key = cat_name if normalize in (True, "column", "row") else cat_name
        totals = agg.groupby(key, observed=True)["value"].transform("sum")
        out = agg.copy()
        out["value"] = 100.0 * out["value"] / totals.replace(0, np.nan)
        out["low"] = np.nan
        out["high"] = np.nan
        return out

    def _draw_grouped_bars(
        self,
        *,
        ax: Axes,
        agg: pd.DataFrame,
        cat_name: str,
        hue_name: str | None,
        levels: Sequence[Any],
        hue_levels: Sequence[Any],
        mapping: Mapping[Any, str],
        orient: str,
        style: StyleConfig,
        errorbar: Any,
        annotate: bool,
        annot_fmt: str,
        kwargs: Mapping[str, Any],
    ) -> None:
        """Dodged bars, one per (level, hue level)."""
        positions = np.arange(len(levels), dtype=float)
        series = hue_levels if hue_name else [None]
        width = 0.8 / max(len(series), 1)
        for index, hue_level in enumerate(series):
            subset = (
                agg if hue_name is None else agg[agg[hue_name] == hue_level]
            )
            lookup = subset.set_index(cat_name)
            values = np.array(
                [float(lookup["value"].get(lv, np.nan)) for lv in levels]
            )
            offset = (
                0.0
                if not hue_name
                else (index - (len(series) - 1) / 2) * width
            )
            colour = mapping.get(
                hue_level if hue_name else levels[0], style.categorical[0]
            )
            colours = (
                [mapping.get(lv, style.categorical[0]) for lv in levels]
                if not hue_name
                else colour
            )
            err = None
            if errorbar is not None and "low" in subset.columns:
                low = np.array(
                    [float(lookup["low"].get(lv, np.nan)) for lv in levels]
                )
                high = np.array(
                    [float(lookup["high"].get(lv, np.nan)) for lv in levels]
                )
                if np.isfinite(low).any():
                    err = np.vstack(
                        [
                            np.nan_to_num(values - low, nan=0.0),
                            np.nan_to_num(high - values, nan=0.0),
                        ]
                    )
            bar_kwargs: dict[str, Any] = {
                "color": colours,
                "edgecolor": "none",
                "zorder": style.data_zorder,
                "label": str(hue_level) if hue_name else None,
            }
            if err is not None:
                bar_kwargs["yerr" if orient == "vertical" else "xerr"] = err
                bar_kwargs["error_kw"] = {
                    "ecolor": style.text_color,
                    "elinewidth": 1.0,
                    "capsize": 2.5,
                    "zorder": style.data_zorder + 1,
                }
            if orient == "vertical":
                ax.bar(
                    positions + offset,
                    values,
                    width=width * 0.92,
                    **bar_kwargs,
                    **dict(kwargs),
                )
            else:
                ax.barh(
                    positions + offset,
                    values,
                    height=width * 0.92,
                    **bar_kwargs,
                    **dict(kwargs),
                )
            if annotate:
                self._annotate_bars(
                    ax,
                    positions + offset,
                    values,
                    orient,
                    style,
                    annot_fmt,
                )
        self._set_category_ticks(ax, positions, levels, orient)

    def _draw_stacked_bars(
        self,
        *,
        ax: Axes,
        agg: pd.DataFrame,
        cat_name: str,
        hue_name: str,
        levels: Sequence[Any],
        hue_levels: Sequence[Any],
        mapping: Mapping[Any, str],
        orient: str,
        style: StyleConfig,
        fill_missing: float | None,
        annotate: bool,
        annot_fmt: str,
        kwargs: Mapping[str, Any],
    ) -> None:
        """Stacked bars; missing combinations stay gaps unless filled."""
        positions = np.arange(len(levels), dtype=float)
        cursor = np.zeros(len(levels), dtype=float)
        for hue_level in hue_levels:
            lookup = agg[agg[hue_name] == hue_level].set_index(cat_name)
            raw = np.array(
                [float(lookup["value"].get(lv, np.nan)) for lv in levels]
            )
            values = np.nan_to_num(raw, nan=float(fill_missing or 0.0))
            colour = mapping.get(hue_level, style.categorical[0])
            if orient == "vertical":
                ax.bar(
                    positions,
                    values,
                    bottom=cursor,
                    width=0.74,
                    color=colour,
                    edgecolor="none",
                    zorder=style.data_zorder,
                    label=str(hue_level),
                    **dict(kwargs),
                )
            else:
                ax.barh(
                    positions,
                    values,
                    left=cursor,
                    height=0.74,
                    color=colour,
                    edgecolor="none",
                    zorder=style.data_zorder,
                    label=str(hue_level),
                    **dict(kwargs),
                )
            if annotate:
                self._annotate_segments(
                    ax,
                    positions,
                    values,
                    cursor,
                    orient,
                    style,
                    annot_fmt,
                )
            cursor = cursor + values
        self._set_category_ticks(ax, positions, levels, orient)

    @staticmethod
    def _set_category_ticks(
        ax: Axes,
        positions: np.ndarray,
        levels: Sequence[Any],
        orient: str,
    ) -> None:
        """Pin category ticks before labelling them."""
        labels = [str(lv) for lv in levels]
        if orient == "vertical":
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(positions)
            ax.set_yticklabels(labels)

    def _annotate_bars(
        self,
        ax: Axes,
        positions: np.ndarray,
        values: np.ndarray,
        orient: str,
        style: StyleConfig,
        annot_fmt: str,
    ) -> None:
        """Write each bar's value just past its end."""
        limit = ax.get_ylim()[1] if orient == "vertical" else ax.get_xlim()[1]
        for pos, value in zip(positions, values):
            if not np.isfinite(value):
                continue
            inside = limit and abs(value) > 0.92 * abs(limit)
            text = self._format_annotation(value, annot_fmt)
            if orient == "vertical":
                ax.annotate(
                    text,
                    xy=(pos, value),
                    xytext=(0, -12 if inside else 3),
                    textcoords="offset points",
                    ha="center",
                    va="top" if inside else "bottom",
                    fontsize=style.annot_size,
                    color="white" if inside else style.text_color,
                )
            else:
                ax.annotate(
                    text,
                    xy=(value, pos),
                    xytext=(-4 if inside else 4, 0),
                    textcoords="offset points",
                    ha="right" if inside else "left",
                    va="center",
                    fontsize=style.annot_size,
                    color="white" if inside else style.text_color,
                )

    def _annotate_segments(
        self,
        ax: Axes,
        positions: np.ndarray,
        values: np.ndarray,
        base: np.ndarray,
        orient: str,
        style: StyleConfig,
        annot_fmt: str,
    ) -> None:
        """Label stacked segments only where the text fits."""
        span = (
            np.ptp(ax.get_ylim())
            if orient == "vertical"
            else np.ptp(ax.get_xlim())
        )
        for pos, value, start in zip(positions, values, base):
            if not np.isfinite(value) or value <= 0:
                continue
            if span and value < 0.06 * span:
                continue
            centre = start + value / 2
            xy = (pos, centre) if orient == "vertical" else (centre, pos)
            ax.annotate(
                self._format_annotation(value, annot_fmt),
                xy=xy,
                ha="center",
                va="center",
                fontsize=style.annot_size * 0.9,
                color="white",
            )

    @staticmethod
    def _format_annotation(value: float, annot_fmt: str) -> str:
        """Accept both ``"{:,.2f}"`` and ``".2f"`` annotation formats."""
        if "{" in annot_fmt:
            return annot_fmt.format(value)
        return format(value, annot_fmt)

    def _draw_baseline(
        self,
        ax: Axes,
        data: pd.DataFrame,
        value_name: str | None,
        baseline: float | str,
        orient: str,
        style: StyleConfig,
    ) -> None:
        """Draw a labelled reference line across the value axis."""
        if isinstance(baseline, str):
            if value_name is None:
                raise ValueError(
                    f"baseline={baseline!r} needs a numeric value "
                    f"column; this is a count plot."
                )
            level = float(
                self._ESTIMATORS[baseline](
                    data[value_name].dropna().to_numpy(dtype=float)
                )
            )
            label = f"{baseline}({value_name}) = {level:,.4g}"
        else:
            level = float(baseline)
            label = f"baseline = {level:,.4g}"
        drawer = ax.axhline if orient == "vertical" else ax.axvline
        drawer(
            level,
            color=style.subtitle_color,
            linestyle="--",
            linewidth=1.3,
            zorder=style.data_zorder + 2,
            label=label,
        )

    # -- curveplot ---------------------------------------------------
    def curveplot(
        self,
        df: pd.DataFrame | None = None,
        x: str | pd.Series | tuple[float, float] | None = None,
        y: str | pd.Series | Sequence[str] | Callable | None = None,
        hue: str | pd.Series | None = None,
        *,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | bool | None = None,
        subtitle: str | None = None,
        panel_title: str | bool | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        palette: str | list | dict | None = None,
        legend: bool = True,
        legend_loc: str = "best",
        hue_order: Sequence[Any] | None = None,
        top_n: int | None = None,
        log_scale: Any = None,
        log_x: bool = False,
        log_y: bool = False,
        transform: Any = None,
        nonpositive: Nonpositive = "raise",
        facet: str | None = None,
        facet_col_wrap: int = 3,
        sharex: bool = True,
        sharey: bool = True,
        dropna: bool = True,
        treat_as: Mapping[str, str] | None = None,
        kind: str = "line",
        agg: str | Callable | None = "mean",
        errorband: str | None = "ci",
        ci_level: float = 95,
        smooth: str | None = None,
        smooth_window: Any = None,
        show_raw: bool = True,
        markers: bool = False,
        dashes: bool = False,
        fill: bool = False,
        baseline: float = 0.0,
        secondary_y: str | None = None,
        annotate_last: bool = False,
        sort_x: bool = True,
        complementary: bool = False,
        bw_adjust: float = 1.0,
        n_points: int = 200,
        vlines: Any = None,
        hlines: Any = None,
        style: Mapping[str, Any] | str | None = None,
        save_as: str | Path | None = None,
        show: bool | None = None,
        ax_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Axes | Figure:
        """Show how a value evolves along an ordered axis.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Frame to plot.
        x : str, pandas.Series or tuple, optional
            Ordered axis. For ``kind="function"`` pass the domain as
            ``(lo, hi)``.
        y : str, sequence of str or callable, optional
            Value column, several columns (one curve each, wide format),
            or the function to evaluate.
        kind : {"line", "ecdf", "kde", "cumulative", "function", \
"step"}, default "line"
            Curve type. ``"step"`` is correct for state changes and
            counts.
        agg : str, callable or None, default "mean"
            How repeated ``x`` values collapse. ``None`` draws every
            raw trajectory.
        errorband : {"ci", "se", "sd", "pi", None}, default "ci"
            Band drawn around an aggregated curve.
        smooth : {"rolling", "ewm", "lowess", "savgol", None}, optional
            Smoothing, always computed **per hue group** so it never
            runs across a group boundary.
        annotate_last : bool, default False
            Label each series at its right end instead of using a
            legend; usually more readable for time series.
        secondary_y : str, optional
            Column drawn on a twin axis with a merged legend.

        Returns
        -------
        matplotlib.axes.Axes or matplotlib.figure.Figure
            The axes for a single panel, the figure when faceting.

        Examples
        --------
        >>> import matplotlib, pandas as pd
        >>> matplotlib.use("Agg")
        >>> df = pd.DataFrame({"t": range(10),
        ...                    "v": [float(i) for i in range(10)]})
        >>> ax = EDAPlotter(df).curveplot(x="t", y="v")
        >>> ax.get_xlabel()
        't'
        """
        frame, style_cfg, treat, extra_ax = self._resolve_common(
            df,
            style=style,
            figsize=figsize,
            treat_as=treat_as,
            ax=ax,
            facet=facet,
            ax_kwargs=ax_kwargs,
        )
        if kind == "function":
            return self._curve_function(
                x=x,
                y=y,
                style_cfg=style_cfg,
                ax=ax,
                figsize=figsize,
                n_points=n_points,
                title=title,
                subtitle=subtitle,
                panel_title=panel_title,
                xlabel=xlabel,
                ylabel=ylabel,
                hue_order=hue_order,
                legend=legend,
                legend_loc=legend_loc,
                save_as=save_as,
                show=show,
                kwargs=kwargs,
            )

        if isinstance(y, (list, tuple)) and not isinstance(y, str):
            id_vars = [c for c in [x, hue, facet] if isinstance(c, str)]
            frame = frame.melt(
                id_vars=id_vars,
                value_vars=list(y),
                var_name="series",
                value_name="value",
            )
            y, hue = "value", hue or "series"

        xs = self._column(frame, x, "x")
        ys = self._column(frame, y, "y")
        if xs is None and kind in {"ecdf", "kde"} and ys is not None:
            xs, ys = ys, None
        if xs is None:
            raise ValueError("curveplot needs x=<column>.")
        x_name = str(xs.name)
        y_name = str(ys.name) if ys is not None else None
        if kind in {"line", "step", "cumulative"} and y_name is None:
            raise ValueError(
                f"kind={kind!r} needs a y= value column; only 'ecdf' "
                f"and 'kde' work from x alone."
            )

        x_kind = self.resolve_kind(xs, treat.get(x_name))
        if x_kind == "categorical" and kind in {"line", "step"}:
            ordered = self._is_ordinal(xs) or (
                isinstance(xs.dtype, pd.CategoricalDtype) and xs.dtype.ordered
            )
            self._warn_once(
                f"curve-cat:{x_name}",
                f"x={x_name!r} resolves to categorical and a line "
                f"implies continuity between points."
                + (
                    " It is ordinal, so the request is honoured."
                    if ordered
                    else " Consider barplot instead."
                ),
            )

        work = frame.copy()
        work[x_name] = xs.to_numpy()
        if y_name:
            work[y_name] = ys.to_numpy()
        hue_series = self._column(work, hue, "hue")
        hue_name = str(hue_series.name) if hue_series is not None else None
        if hue_name:
            work[hue_name] = hue_series.to_numpy()
        work = self._dropna_subset(
            work,
            [x_name, y_name, hue_name, facet],
            dropna,
            f"curveplot({x_name!r})",
        )
        if hue_name:
            work[hue_name] = self._limit_levels(work[hue_name], top_n)
            if work[hue_name].nunique() > 6:
                self._warn_once(
                    f"curve-hue:{hue_name}",
                    f"hue={hue_name!r} has "
                    f"{work[hue_name].nunique()} levels; overlaid "
                    f"curves get hard to follow. Consider "
                    f"facet={hue_name!r}.",
                )

        if facet is not None:
            return self._facet(
                work,
                facet,
                lambda panel_ax, sub, level: self.curveplot(
                    sub,
                    x=x_name,
                    y=y_name,
                    hue=hue_name,
                    ax=panel_ax,
                    kind=kind,
                    agg=agg,
                    errorband=errorband,
                    smooth=smooth,
                    smooth_window=smooth_window,
                    markers=markers,
                    dashes=dashes,
                    fill=fill,
                    legend=legend,
                    treat_as=treat,
                    title=False,
                    panel_title=False,
                    style=style,
                    **kwargs,
                ),
                style=style_cfg,
                col_wrap=facet_col_wrap,
                sharex=sharex,
                sharey=sharey,
                figsize=figsize,
                title=self._auto_title(
                    title, self._curve_title(kind, x_name, y_name)
                ),
                subtitle=subtitle,
                save_as=save_as,
                show=show,
            )

        owns_figure = ax is None
        with self._style_scope(style_cfg):
            if owns_figure:
                fig, axes = self._new_figure(style_cfg, figsize)
                ax = axes[0, 0]
            else:
                fig = ax.figure
            self._draw_curve(
                ax=ax,
                data=work,
                x_name=x_name,
                y_name=y_name,
                hue_name=hue_name,
                style=style_cfg,
                kind=kind,
                agg=agg,
                errorband=errorband,
                ci_level=ci_level,
                smooth=smooth,
                smooth_window=smooth_window,
                show_raw=show_raw,
                markers=markers,
                dashes=dashes,
                fill=fill,
                baseline=baseline,
                annotate_last=annotate_last,
                sort_x=sort_x,
                complementary=complementary,
                bw_adjust=bw_adjust,
                hue_order=hue_order,
                palette=palette,
                log_scale=log_scale,
                log_x=log_x,
                log_y=log_y,
                transform=transform,
                nonpositive=nonpositive,
                secondary_y=secondary_y,
                vlines=vlines,
                hlines=hlines,
                xlabel=xlabel,
                ylabel=ylabel,
                legend=legend,
                legend_loc=legend_loc,
                kwargs=kwargs,
            )
            self._panel_title(
                ax,
                style_cfg,
                self._auto_title(
                    panel_title,
                    None
                    if owns_figure
                    else self._curve_title(kind, x_name, y_name),
                ),
            )
            if extra_ax:
                ax.set(**extra_ax)
            if owns_figure:
                top = self._place_header(
                    fig,
                    style_cfg,
                    self._auto_title(
                        title, self._curve_title(kind, x_name, y_name)
                    ),
                    subtitle,
                    style_cfg.margins,
                )
                self._finish(fig, style_cfg, {}, top, save_as, show)
        return ax

    @staticmethod
    def _curve_title(kind: str, x_name: str, y_name: str | None) -> str:
        """Default figure title for a curve."""
        if kind == "ecdf":
            return f"ECDF of {x_name}"
        if kind == "kde":
            return f"Density of {x_name}"
        return f"{y_name} over {x_name}"

    def _curve_function(
        self,
        *,
        x: Any,
        y: Any,
        style_cfg: StyleConfig,
        ax: Axes | None,
        figsize: tuple[float, float] | None,
        n_points: int,
        title: Any,
        subtitle: str | None,
        panel_title: Any,
        xlabel: str | None,
        ylabel: str | None,
        hue_order: Sequence[Any] | None,
        legend: bool,
        legend_loc: str,
        save_as: str | Path | None,
        show: bool | None,
        kwargs: Mapping[str, Any],
    ) -> Axes:
        """Plot one or more callables over a numeric domain."""
        if not (isinstance(x, (tuple, list)) and len(x) == 2):
            raise ValueError("kind='function' needs x=(lo, hi) as the domain.")
        funcs = y if isinstance(y, (list, tuple)) else [y]
        if not all(callable(f) for f in funcs):
            raise ValueError(
                "kind='function' needs y= to be a callable or a list "
                "of callables."
            )
        labels = list(hue_order or [])
        grid = np.linspace(float(x[0]), float(x[1]), n_points)
        owns_figure = ax is None
        with self._style_scope(style_cfg):
            if owns_figure:
                fig, axes = self._new_figure(style_cfg, figsize)
                ax = axes[0, 0]
            else:
                fig = ax.figure
            names = [
                labels[i]
                if i < len(labels)
                else getattr(funcs[i], "__name__", f"f{i}")
                for i in range(len(funcs))
            ]
            mapping = self.color_map("__function__", names)
            for name, func in zip(names, funcs):
                ax.plot(
                    grid,
                    np.asarray(func(grid), dtype=float),
                    color=mapping[name],
                    linewidth=style_cfg.line_width,
                    label=str(name),
                    zorder=style_cfg.data_zorder,
                    **dict(kwargs),
                )
            self._apply_grid(ax, style_cfg, "y")
            self._despine(ax, style_cfg)
            ax.set_xlabel(xlabel or "x")
            ax.set_ylabel(ylabel or "f(x)")
            self._style_legend(
                ax, style_cfg, None, legend and len(funcs) > 1, legend_loc
            )
            self._panel_title(
                ax, style_cfg, self._auto_title(panel_title, None)
            )
            if owns_figure:
                top = self._place_header(
                    fig,
                    style_cfg,
                    self._auto_title(title, "Function"),
                    subtitle,
                    style_cfg.margins,
                )
                self._finish(fig, style_cfg, {}, top, save_as, show)
        return ax

    @staticmethod
    def _window_size(window: Any, n: int) -> int:
        """Resolve a smoothing window to a usable integer.

        Accepts ``None`` (a twentieth of the series), a fraction in
        ``(0, 1)`` — which is what a LOWESS ``frac`` becomes when it
        falls back to a rolling mean — or an explicit count.
        """
        if window is None:
            size = max(3, n // 20)
        elif 0 < float(window) < 1:
            size = max(3, int(n * float(window)))
        else:
            size = max(3, int(window))
        return max(1, min(size, n or 1))

    def _smooth_series(
        self,
        values: pd.Series,
        method: str,
        window: Any,
        x_values: np.ndarray | None = None,
    ) -> pd.Series:
        """Smooth one group's series; never across group boundaries."""
        if method == "rolling":
            size = self._window_size(window, len(values))
            return values.rolling(size, center=True, min_periods=1).mean()
        if method == "ewm":
            span = self._window_size(window, len(values))
            return values.ewm(span=span, adjust=False).mean()
        if method == "savgol":
            from scipy.signal import savgol_filter

            size = int(window or max(5, len(values) // 10))
            size = min(size if size % 2 else size + 1, len(values))
            if size < 3:
                return values
            order = min(3, size - 1)
            return pd.Series(
                savgol_filter(values.to_numpy(float), size, order),
                index=values.index,
            )
        if method == "lowess":
            try:
                from statsmodels.nonparametric.smoothers_lowess import (
                    lowess,
                )
            except ImportError:
                self._warn_once(
                    "lowess-missing",
                    "smooth='lowess' needs statsmodels, which is not "
                    "installed; falling back to a centred rolling mean. "
                    "Install statsmodels for LOWESS.",
                )
                return self._smooth_series(values, "rolling", window)
            frac = float(window) if window else 0.3
            fitted = lowess(
                values.to_numpy(float),
                np.asarray(x_values, dtype=float),
                frac=frac,
                return_sorted=False,
            )
            return pd.Series(fitted, index=values.index)
        raise ValueError(
            f"smooth={method!r} is not recognised. Use 'rolling', "
            f"'ewm', 'lowess', 'savgol' or None."
        )

    def _draw_curve(
        self,
        *,
        ax: Axes,
        data: pd.DataFrame,
        x_name: str,
        y_name: str | None,
        hue_name: str | None,
        style: StyleConfig,
        kind: str,
        agg: Any,
        errorband: str | None,
        ci_level: float,
        smooth: str | None,
        smooth_window: Any,
        show_raw: bool,
        markers: bool,
        dashes: bool,
        fill: bool,
        baseline: float,
        annotate_last: bool,
        sort_x: bool,
        complementary: bool,
        bw_adjust: float,
        hue_order: Sequence[Any] | None,
        palette: Any,
        log_scale: Any,
        log_x: bool,
        log_y: bool,
        transform: Any,
        nonpositive: Nonpositive,
        secondary_y: str | None,
        vlines: Any,
        hlines: Any,
        xlabel: str | None,
        ylabel: str | None,
        legend: bool,
        legend_loc: str,
        kwargs: Mapping[str, Any],
    ) -> Axes:
        """Draw curves into ``ax``."""
        log_spec = self._normalise_log_scale(log_scale, "y", log_x, log_y)
        self._check_scale_conflict(log_spec, transform, "y", y_name or x_name)
        label = y_name or x_name
        work = data
        if transform is not None and y_name:
            values, meta = _apply_transform(
                data[y_name],
                transform,
                nonpositive,
                column=y_name,
                lang=self.lang,
            )
            self._last_transform = meta
            work = data.assign(**{y_name: values})
            label = meta.label_for(y_name)

        levels = (
            self.ordered_levels(work[hue_name], hue_order)
            if hue_name
            else [None]
        )
        mapping = (
            self.color_map(hue_name or x_name, levels, palette)
            if hue_name
            else {None: style.categorical[0]}
        )
        styles = ["-", "--", "-.", ":"]
        marks = ["o", "s", "^", "D", "v", "P"]

        for index, level in enumerate(levels):
            subset = work if level is None else work[work[hue_name] == level]
            colour = mapping[level]
            line_kwargs: dict[str, Any] = {
                "color": colour,
                "linewidth": style.line_width,
                "zorder": style.data_zorder,
                "label": None if level is None else str(level),
            }
            if dashes:
                line_kwargs["linestyle"] = styles[index % len(styles)]
            if markers:
                line_kwargs["marker"] = marks[index % len(marks)]
                line_kwargs["markersize"] = 4
            self._draw_one_curve(
                ax=ax,
                subset=subset,
                x_name=x_name,
                y_name=y_name,
                style=style,
                kind=kind,
                agg=agg,
                errorband=errorband,
                ci_level=ci_level,
                smooth=smooth,
                smooth_window=smooth_window,
                show_raw=show_raw,
                fill=fill,
                baseline=baseline,
                sort_x=sort_x,
                complementary=complementary,
                bw_adjust=bw_adjust,
                colour=colour,
                line_kwargs=line_kwargs,
                annotate_last=annotate_last,
                level=level,
                kwargs=kwargs,
            )

        if secondary_y is not None:
            twin = ax.twinx()
            twin.plot(
                work[x_name],
                pd.to_numeric(work[secondary_y], errors="coerce"),
                color=style.categorical[3],
                linewidth=style.line_width,
                label=secondary_y,
            )
            twin.set_ylabel(secondary_y, color=style.categorical[3])
            twin.tick_params(axis="y", colors=style.categorical[3])
            sns.despine(ax=twin, left=True, bottom=True, right=True)
        self._draw_event_lines(ax, vlines, hlines, style)

        if log_spec:
            self._apply_log_axis(
                ax,
                log_spec,
                style,
                {"y": work[y_name].to_numpy(float)} if y_name else None,
                {"y": y_name} if y_name else None,
                nonpositive,
            )
        grid_axis = "y"
        if is_datetime64_any_dtype(work[x_name].dtype):
            grid_axis = "both"
            ax.figure.autofmt_xdate()
        self._apply_grid(ax, style, grid_axis)
        self._despine(ax, style)
        ax.set_xlabel(xlabel if xlabel is not None else x_name)
        default_y = {
            "ecdf": "Proportion",
            "kde": "Density",
        }.get(kind, label)
        ax.set_ylabel(ylabel if ylabel is not None else default_y)
        if "y" not in log_spec:
            self._format_value_axis(ax, "y", style)
        self._style_legend(
            ax,
            style,
            hue_name,
            legend and not annotate_last,
            legend_loc,
        )
        return ax

    def _draw_one_curve(
        self,
        *,
        ax: Axes,
        subset: pd.DataFrame,
        x_name: str,
        y_name: str | None,
        style: StyleConfig,
        kind: str,
        agg: Any,
        errorband: str | None,
        ci_level: float,
        smooth: str | None,
        smooth_window: Any,
        show_raw: bool,
        fill: bool,
        baseline: float,
        sort_x: bool,
        complementary: bool,
        bw_adjust: float,
        colour: str,
        line_kwargs: Mapping[str, Any],
        annotate_last: bool,
        level: Any,
        kwargs: Mapping[str, Any],
    ) -> None:
        """Draw a single series, with its band and smoothing."""
        opts = dict(line_kwargs)
        if kind == "ecdf":
            values = np.sort(
                pd.to_numeric(subset[x_name], errors="coerce")
                .dropna()
                .to_numpy(float)
            )
            proportion = np.arange(1, values.size + 1) / values.size
            if complementary:
                proportion = 1.0 - proportion
            ax.step(values, proportion, where="post", **opts, **dict(kwargs))
            return
        if kind == "kde":
            sns.kdeplot(
                x=pd.to_numeric(subset[x_name], errors="coerce"),
                ax=ax,
                color=colour,
                bw_adjust=bw_adjust,
                linewidth=style.line_width,
                label=opts.get("label"),
                zorder=style.data_zorder,
                **dict(kwargs),
            )
            return

        assert y_name is not None
        frame = subset[[x_name, y_name]].copy()
        if sort_x:
            frame = frame.sort_values(x_name)
        low = high = None
        duplicated = bool(frame[x_name].duplicated().any())
        if agg is not None and duplicated:
            grouped = frame.groupby(x_name, observed=True)[y_name]
            func = agg if callable(agg) else agg
            curve = grouped.agg(func)
            if errorband:
                estimator = (
                    np.mean
                    if not callable(agg) and agg in {"mean", None}
                    else (np.median if agg == "median" else np.mean)
                )
                bounds = [
                    self._interval(
                        group.to_numpy(dtype=float),
                        errorband,
                        ci_level,
                        estimator,
                    )
                    for _key, group in grouped
                ]
                low = np.array([b[0] for b in bounds])
                high = np.array([b[1] for b in bounds])
            xs = curve.index.to_numpy()
            ys = curve.to_numpy(dtype=float)
        elif agg is None and duplicated:
            self._warn_once(
                f"curve-raw:{y_name}",
                "agg=None draws every raw trajectory; with many units "
                "this becomes a hairball. Set agg='mean' to collapse "
                "them.",
            )
            xs = frame[x_name].to_numpy()
            ys = frame[y_name].to_numpy(dtype=float)
        else:
            xs = frame[x_name].to_numpy()
            ys = frame[y_name].to_numpy(dtype=float)

        if kind == "cumulative":
            ys = np.nancumsum(ys)
        plotted = pd.Series(ys)
        if smooth:
            raw_ys = ys
            plotted = self._smooth_series(
                pd.Series(ys),
                smooth,
                smooth_window,
                pd.to_numeric(pd.Series(xs), errors="coerce").to_numpy(),
            )
            if show_raw:
                ax.plot(
                    xs,
                    raw_ys,
                    color=colour,
                    linewidth=1.0,
                    alpha=0.28,
                    zorder=style.data_zorder - 1,
                )
        drawer = ax.step if kind == "step" else ax.plot
        step_kw = {"where": "post"} if kind == "step" else {}
        drawer(xs, plotted.to_numpy(), **opts, **step_kw, **dict(kwargs))
        if low is not None and high is not None:
            ax.fill_between(
                xs,
                low,
                high,
                color=colour,
                alpha=0.15,
                linewidth=0,
                zorder=style.data_zorder - 1,
            )
        if fill:
            ax.fill_between(
                xs,
                baseline,
                plotted.to_numpy(),
                color=colour,
                alpha=0.25,
                linewidth=0,
                zorder=style.data_zorder - 1,
            )
        if annotate_last and len(xs):
            ax.annotate(
                str(level) if level is not None else (y_name or ""),
                xy=(xs[-1], plotted.to_numpy()[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=style.annot_size,
                color=colour,
            )

    def _draw_event_lines(
        self, ax: Axes, vlines: Any, hlines: Any, style: StyleConfig
    ) -> None:
        """Draw labelled event markers."""
        for spec, drawer in ((vlines, ax.axvline), (hlines, ax.axhline)):
            if spec is None:
                continue
            items = (
                spec.items()
                if isinstance(spec, Mapping)
                else [(None, v) for v in np.atleast_1d(spec)]
            )
            for label, value in items:
                drawer(
                    value,
                    color=style.subtitle_color,
                    linestyle=":",
                    linewidth=1.2,
                    zorder=style.data_zorder + 2,
                    label=str(label) if label is not None else None,
                )

    # -- scatterplot -------------------------------------------------
    def scatterplot(
        self,
        df: pd.DataFrame | None = None,
        x: str | pd.Series | None = None,
        y: str | pd.Series | None = None,
        hue: str | pd.Series | None = None,
        *,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | bool | None = None,
        subtitle: str | None = None,
        panel_title: str | bool | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        palette: str | list | dict | None = None,
        legend: bool = True,
        legend_loc: str = "best",
        hue_order: Sequence[Any] | None = None,
        top_n: int | None = None,
        log_scale: Any = None,
        log_x: bool = False,
        log_y: bool = False,
        transform: Any = None,
        nonpositive: Nonpositive = "raise",
        facet: str | None = None,
        facet_col_wrap: int = 3,
        sharex: bool = True,
        sharey: bool = True,
        dropna: bool = True,
        treat_as: Mapping[str, str] | None = None,
        size: str | None = None,
        size_range: tuple[float, float] = (20, 200),
        size_norm: str = "linear",
        style_by: str | None = None,
        alpha: float | None = None,
        trend: str | None = None,
        trend_kwargs: Mapping[str, Any] | None = None,
        trend_per_group: bool = True,
        trend_ci: float | None = 95,
        trend_show_eq: bool = False,
        degree: int = 2,
        annotate_corr: str | Sequence[str] | None = None,
        jitter: float | None = None,
        sample: int | float | None = None,
        marginals: bool = False,
        identity_line: bool = False,
        label_points: str | None = None,
        kind: str = "points",
        cmap: Any = None,
        style: Mapping[str, Any] | str | None = None,
        save_as: str | Path | None = None,
        show: bool | None = None,
        ax_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Axes | Figure:
        """Show the relationship between two numeric variables.

        The ``hue`` channel branches on kind: a categorical ``hue`` gets
        a discrete palette and a legend, while a high-cardinality
        numeric ``hue`` gets a continuous colormap and a **colorbar**.

        ``log_scale`` and ``transform`` differ in a way that matters
        most here. With ``log_scale`` the trend line is fitted in **raw**
        space and drawn on a log axis, so a power law looks curved. With
        ``transform`` it is fitted in **transformed** space, so a power
        law becomes a straight line. Both are legitimate;
        ``annotate_corr`` states which space the number came from.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Frame to plot.
        x, y : str or pandas.Series
            Numeric variables, both required.
        size : str, optional
            Numeric column driving bubble area.
        style_by : str, optional
            Categorical column driving marker shape.
        trend : {"linear", "poly", "lowess", "theilsen", None}, optional
            Trend line, fitted per ``hue`` group by default.
        annotate_corr : str or sequence, optional
            ``"pearson"``, ``"spearman"`` and/or ``"kendall"``.
        jitter : float, optional
            Jitter width; auto-enabled for low-cardinality axes.
        sample : int or float, optional
            Draw a reproducible subset before plotting.
        marginals : bool, default False
            Build a joint layout with marginal distributions. Returns
            the ``Figure`` rather than an ``Axes``, and is incompatible
            with a supplied ``ax``.
        identity_line : bool, default False
            Draw ``y = x`` and force equal limits and aspect.

        Returns
        -------
        matplotlib.axes.Axes or matplotlib.figure.Figure
            Axes normally; Figure when faceting or with ``marginals``.

        Examples
        --------
        >>> import matplotlib, pandas as pd
        >>> matplotlib.use("Agg")
        >>> df = pd.DataFrame(
        ...     {"a": [1.5, 2.5, 3.5, 4.5], "b": [2.1, 3.9, 6.2, 7.8]}
        ... )
        >>> ax = EDAPlotter(df).scatterplot(x="a", y="b")
        >>> ax.get_xlabel()
        'a'
        """
        frame, style_cfg, treat, extra_ax = self._resolve_common(
            df,
            style=style,
            figsize=figsize,
            treat_as=treat_as,
            ax=ax,
            facet=facet,
            ax_kwargs=ax_kwargs,
        )
        if marginals and ax is not None:
            raise ValueError(
                "marginals=True builds its own joint layout, so it "
                "cannot draw into a supplied ax."
            )
        xs = self._column(frame, x, "x")
        ys = self._column(frame, y, "y")
        if xs is None or ys is None:
            raise ValueError("scatterplot needs both x= and y=.")
        x_name, y_name = str(xs.name), str(ys.name)
        kind_x = self.resolve_kind(xs, treat.get(x_name))
        kind_y = self.resolve_kind(ys, treat.get(y_name))
        if kind_x == "categorical" and kind_y == "categorical":
            raise ValueError(
                f"Both x={x_name!r} and y={y_name!r} are categorical, "
                f"so there is no relationship to draw. Use a count "
                f"heatmap or barplot for a cross-tab."
            )
        if "categorical" in (kind_x, kind_y):
            self._warn_once(
                f"scatter-cat:{x_name}:{y_name}",
                f"One of {x_name!r} / {y_name!r} is categorical; "
                f"boxplot or barplot would read better. Drawing a "
                f"jittered strip-style scatter instead.",
            )
            jitter = jitter if jitter is not None else 0.2

        work = frame.copy()
        work[x_name] = pd.to_numeric(xs, errors="coerce").to_numpy()
        work[y_name] = pd.to_numeric(ys, errors="coerce").to_numpy()
        hue_series = self._column(work, hue, "hue")
        hue_name = str(hue_series.name) if hue_series is not None else None
        if hue_name:
            work[hue_name] = hue_series.to_numpy()
        used = [x_name, y_name, hue_name, size, style_by, facet]
        work = self._dropna_subset(
            work, used, dropna, f"scatterplot({x_name!r}, {y_name!r})"
        )

        if facet is not None:
            return self._facet(
                work,
                facet,
                lambda panel_ax, sub, level: self.scatterplot(
                    sub,
                    x=x_name,
                    y=y_name,
                    hue=hue_name,
                    ax=panel_ax,
                    size=size,
                    style_by=style_by,
                    trend=trend,
                    annotate_corr=annotate_corr,
                    jitter=jitter,
                    alpha=alpha,
                    legend=legend,
                    treat_as=treat,
                    title=False,
                    panel_title=False,
                    style=style,
                    **kwargs,
                ),
                style=style_cfg,
                col_wrap=facet_col_wrap,
                sharex=sharex,
                sharey=sharey,
                figsize=figsize,
                title=self._auto_title(title, f"{y_name} vs {x_name}"),
                subtitle=subtitle,
                min_rows=2,
                save_as=save_as,
                show=show,
            )

        if marginals:
            return self._scatter_joint(
                work=work,
                x_name=x_name,
                y_name=y_name,
                hue_name=hue_name,
                style_cfg=style_cfg,
                figsize=figsize,
                title=title,
                subtitle=subtitle,
                palette=palette,
                hue_order=hue_order,
                alpha=alpha,
                save_as=save_as,
                show=show,
            )

        owns_figure = ax is None
        with self._style_scope(style_cfg):
            if owns_figure:
                fig, axes = self._new_figure(style_cfg, figsize)
                ax = axes[0, 0]
            else:
                fig = ax.figure
            self._draw_scatter(
                ax=ax,
                data=work,
                x_name=x_name,
                y_name=y_name,
                hue_name=hue_name,
                style=style_cfg,
                treat=treat,
                size=size,
                size_range=size_range,
                size_norm=size_norm,
                style_by=style_by,
                alpha=alpha,
                trend=trend,
                trend_kwargs=trend_kwargs,
                trend_per_group=trend_per_group,
                trend_ci=trend_ci,
                trend_show_eq=trend_show_eq,
                degree=degree,
                annotate_corr=annotate_corr,
                jitter=jitter,
                sample=sample,
                identity_line=identity_line,
                label_points=label_points,
                kind=kind,
                cmap=cmap,
                palette=palette,
                hue_order=hue_order,
                top_n=top_n,
                log_scale=log_scale,
                log_x=log_x,
                log_y=log_y,
                transform=transform,
                nonpositive=nonpositive,
                xlabel=xlabel,
                ylabel=ylabel,
                legend=legend,
                legend_loc=legend_loc,
                kwargs=kwargs,
            )
            self._panel_title(
                ax,
                style_cfg,
                self._auto_title(
                    panel_title,
                    None if owns_figure else f"{y_name} vs {x_name}",
                ),
            )
            if extra_ax:
                ax.set(**extra_ax)
            if owns_figure:
                top = self._place_header(
                    fig,
                    style_cfg,
                    self._auto_title(title, f"{y_name} vs {x_name}"),
                    subtitle,
                    style_cfg.margins,
                )
                self._finish(fig, style_cfg, {}, top, save_as, show)
        return ax

    def _jitter_width(self, values: np.ndarray, requested: float) -> float:
        """Jitter scaled to the minimum spacing between distinct values."""
        unique = np.unique(values[np.isfinite(values)])
        if unique.size < 2:
            return 0.0
        return float(requested * np.min(np.diff(unique)))

    def _draw_scatter(
        self,
        *,
        ax: Axes,
        data: pd.DataFrame,
        x_name: str,
        y_name: str,
        hue_name: str | None,
        style: StyleConfig,
        treat: Mapping[str, str],
        size: str | None,
        size_range: tuple[float, float],
        size_norm: str,
        style_by: str | None,
        alpha: float | None,
        trend: str | None,
        trend_kwargs: Mapping[str, Any] | None,
        trend_per_group: bool,
        trend_ci: float | None,
        trend_show_eq: bool,
        degree: int,
        annotate_corr: Any,
        jitter: float | None,
        sample: int | float | None,
        identity_line: bool,
        label_points: str | None,
        kind: str,
        cmap: Any,
        palette: Any,
        hue_order: Sequence[Any] | None,
        top_n: int | None,
        log_scale: Any,
        log_x: bool,
        log_y: bool,
        transform: Any,
        nonpositive: Nonpositive,
        xlabel: str | None,
        ylabel: str | None,
        legend: bool,
        legend_loc: str,
        kwargs: Mapping[str, Any],
    ) -> Axes:
        """Draw the scatter, its encodings and any trend line."""
        log_spec = self._normalise_log_scale(log_scale, "y", log_x, log_y)
        for axis, column in (("x", x_name), ("y", y_name)):
            self._check_scale_conflict(log_spec, transform, axis, column)

        work = data
        labels = {"x": x_name, "y": y_name}
        if transform is not None:
            spec = (
                transform
                if isinstance(transform, Mapping)
                else {"x": transform, "y": transform}
            )
            for axis, column in (("x", x_name), ("y", y_name)):
                if spec.get(axis) is None:
                    continue
                values, meta = _apply_transform(
                    work[column],
                    spec[axis],
                    nonpositive,
                    column=column,
                    lang=self.lang,
                )
                work = work.assign(**{column: values})
                labels[axis] = meta.label_for(column)
                self._last_transform = meta

        if sample is not None:
            n = (
                int(sample)
                if sample >= 1
                else max(1, int(len(work) * float(sample)))
            )
            if n < len(work):
                work = work.sample(n, random_state=self.random_state)
                self._warn_once(
                    f"scatter-sample:{x_name}",
                    f"Plotting a random sample of {n:,} of "
                    f"{len(data):,} rows (random_state="
                    f"{self.random_state}).",
                )
        elif len(work) > 50_000:
            self._warn_once(
                f"scatter-dense:{x_name}",
                f"n = {len(work):,} points will overplot badly. Pass "
                f"sample=..., or kind='hexbin'/'kde' for a density "
                f"view.",
            )

        xv = work[x_name].to_numpy(dtype=float)
        yv = work[y_name].to_numpy(dtype=float)
        for axis, values, name in (
            ("x", xv, x_name),
            ("y", yv, y_name),
        ):
            if (
                jitter is None
                and np.unique(values[np.isfinite(values)]).size
                <= self.cat_max_cardinality
            ):
                jitter = 0.4
                self._warn_once(
                    f"scatter-jitter:{name}",
                    f"{name!r} has few distinct values on the {axis} "
                    f"axis, so points would stack. Jitter is enabled "
                    f"automatically; pass jitter=0 to disable.",
                )
        rng = np.random.default_rng(self.random_state)
        if jitter:
            xv = xv + rng.uniform(-1, 1, xv.size) * self._jitter_width(
                xv, jitter
            )
            yv = yv + rng.uniform(-1, 1, yv.size) * self._jitter_width(
                yv, jitter
            )

        if alpha is None:
            alpha = float(min(0.8, max(0.1, 1000.0 / max(len(work), 1))))

        if kind in {"hexbin", "kde"}:
            if kind == "hexbin":
                ax.hexbin(
                    xv,
                    yv,
                    gridsize=45,
                    cmap=cmap or style.sequential_cmap(),
                    mincnt=1,
                    zorder=style.data_zorder,
                )
            else:
                sns.kdeplot(
                    x=xv,
                    y=yv,
                    ax=ax,
                    fill=True,
                    cmap=cmap or style.sequential_cmap(),
                    zorder=style.data_zorder,
                )
        else:
            self._scatter_points(
                ax=ax,
                work=work,
                xv=xv,
                yv=yv,
                x_name=x_name,
                y_name=y_name,
                hue_name=hue_name,
                style=style,
                treat=treat,
                size=size,
                size_range=size_range,
                size_norm=size_norm,
                style_by=style_by,
                alpha=alpha,
                palette=palette,
                hue_order=hue_order,
                top_n=top_n,
                cmap=cmap,
                legend=legend,
                kwargs=kwargs,
            )

        if trend:
            self._draw_trend(
                ax=ax,
                work=work.assign(**{x_name: xv, y_name: yv}),
                x_name=x_name,
                y_name=y_name,
                hue_name=hue_name if trend_per_group else None,
                style=style,
                trend=trend,
                trend_kwargs=trend_kwargs,
                trend_ci=trend_ci,
                trend_show_eq=trend_show_eq,
                degree=degree,
                palette=palette,
                hue_order=hue_order,
            )
        if identity_line:
            lo = float(np.nanmin([xv.min(), yv.min()]))
            hi = float(np.nanmax([xv.max(), yv.max()]))
            ax.plot(
                [lo, hi],
                [lo, hi],
                color=style.subtitle_color,
                linestyle="--",
                linewidth=1.2,
                zorder=style.data_zorder - 1,
                label="y = x",
            )
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
        if label_points:
            self._label_points(ax, work, xv, yv, label_points, style)
        if annotate_corr:
            self._annotate_corr(
                ax=ax,
                work=work,
                x_name=x_name,
                y_name=y_name,
                hue_name=hue_name,
                methods=annotate_corr,
                style=style,
                space=("transformed" if transform is not None else "raw"),
            )
        if log_spec:
            self._apply_log_axis(
                ax,
                log_spec,
                style,
                {"x": xv, "y": yv},
                {"x": x_name, "y": y_name},
                nonpositive,
            )
        self._apply_grid(ax, style, "both")
        self._despine(ax, style)
        ax.set_xlabel(xlabel if xlabel is not None else labels["x"])
        ax.set_ylabel(ylabel if ylabel is not None else labels["y"])
        self._style_legend(ax, style, hue_name, legend, legend_loc)
        return ax

    def _scatter_points(
        self,
        *,
        ax: Axes,
        work: pd.DataFrame,
        xv: np.ndarray,
        yv: np.ndarray,
        x_name: str,
        y_name: str,
        hue_name: str | None,
        style: StyleConfig,
        treat: Mapping[str, str],
        size: str | None,
        size_range: tuple[float, float],
        size_norm: str,
        style_by: str | None,
        alpha: float,
        palette: Any,
        hue_order: Sequence[Any] | None,
        top_n: int | None,
        cmap: Any,
        legend: bool,
        kwargs: Mapping[str, Any],
    ) -> None:
        """Plot the markers, honouring hue, size and style channels."""
        sizes = 28.0
        if size is not None:
            if not self.is_numeric(work[size], treat.get(size)):
                self._warn_once(
                    f"scatter-size:{size}",
                    f"size={size!r} is categorical; marker area encodes "
                    f"magnitude, so the channel is ignored. Use "
                    f"style_by={size!r} for shape instead.",
                )
            else:
                raw = work[size].to_numpy(dtype=float)
                sizes = self._scale_sizes(raw, size_range, size_norm)

        markers = ["o", "s", "^", "D", "v", "P", "X"]
        style_levels = (
            self.ordered_levels(work[style_by]) if style_by else [None]
        )

        hue_kind = (
            self.resolve_kind(work[hue_name], treat.get(hue_name))
            if hue_name
            else None
        )
        if hue_name and hue_kind == "numeric":
            # Continuous hue gets a colormap and a colorbar, not a
            # categorical legend.
            mappable = ax.scatter(
                xv,
                yv,
                c=work[hue_name].to_numpy(dtype=float),
                cmap=cmap or style.sequential_cmap(),
                s=sizes,
                alpha=alpha,
                linewidth=style.marker_edge_width,
                edgecolor="white",
                zorder=style.data_zorder,
                **dict(kwargs),
            )
            bar = ax.figure.colorbar(mappable, ax=ax, pad=0.02)
            bar.set_label(hue_name, fontsize=style.label_size)
            bar.outline.set_visible(False)
            return

        levels = (
            self.ordered_levels(
                self._limit_levels(work[hue_name], top_n), hue_order
            )
            if hue_name
            else [None]
        )
        mapping = (
            self.color_map(hue_name, levels, palette)
            if hue_name
            else {None: style.categorical[0]}
        )
        for level in levels:
            for s_index, s_level in enumerate(style_levels):
                mask = np.ones(len(work), dtype=bool)
                if hue_name:
                    mask &= (work[hue_name] == level).to_numpy()
                if style_by:
                    mask &= (work[style_by] == s_level).to_numpy()
                if not mask.any():
                    continue
                label_bits = [
                    str(b) for b in (level, s_level) if b is not None
                ]
                ax.scatter(
                    xv[mask],
                    yv[mask],
                    s=sizes[mask] if isinstance(sizes, np.ndarray) else sizes,
                    color=mapping[level],
                    marker=markers[s_index % len(markers)],
                    alpha=alpha,
                    linewidth=style.marker_edge_width,
                    edgecolor="white",
                    zorder=style.data_zorder,
                    label=" / ".join(label_bits) or None,
                    **dict(kwargs),
                )
        if size is not None and isinstance(sizes, np.ndarray) and legend:
            self._add_size_legend(
                ax,
                work[size].to_numpy(dtype=float),
                size_range,
                size_norm,
                size,
                style,
            )

    @staticmethod
    def _scale_sizes(
        raw: np.ndarray,
        size_range: tuple[float, float],
        size_norm: str,
    ) -> np.ndarray:
        """Map a numeric column onto a marker-area range."""
        values = np.asarray(raw, dtype=float)
        if size_norm == "log":
            values = np.log10(np.clip(values, 1e-12, None))
        elif size_norm == "sqrt":
            values = np.sqrt(np.clip(values, 0, None))
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.full(values.shape, size_range[0])
        lo, hi = float(finite.min()), float(finite.max())
        span = (hi - lo) or 1.0
        unit = (values - lo) / span
        return size_range[0] + unit * (size_range[1] - size_range[0])

    def _add_size_legend(
        self,
        ax: Axes,
        raw: np.ndarray,
        size_range: tuple[float, float],
        size_norm: str,
        name: str,
        style: StyleConfig,
    ) -> None:
        """Add a three-value size key beside the main legend."""
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            return
        picks = np.percentile(finite, [10, 50, 90])
        areas = self._scale_sizes(picks, size_range, size_norm)
        handles = [
            plt.scatter(
                [],
                [],
                s=area,
                color=style.subtitle_color,
                edgecolor="white",
                linewidth=style.marker_edge_width,
                label=f"{value:,.3g}",
            )
            for value, area in zip(picks, areas)
        ]
        extra = ax.legend(
            handles=handles,
            title=name,
            frameon=False,
            loc="lower right",
            fontsize=style.tick_size * 0.9,
        )
        extra.get_title().set_fontsize(style.tick_size)
        ax.add_artist(extra)

    def _draw_trend(
        self,
        *,
        ax: Axes,
        work: pd.DataFrame,
        x_name: str,
        y_name: str,
        hue_name: str | None,
        style: StyleConfig,
        trend: str,
        trend_kwargs: Mapping[str, Any] | None,
        trend_ci: float | None,
        trend_show_eq: bool,
        degree: int,
        palette: Any,
        hue_order: Sequence[Any] | None,
    ) -> None:
        """Fit and draw a trend line, per hue group by default."""
        levels = (
            self.ordered_levels(work[hue_name], hue_order)
            if hue_name
            else [None]
        )
        mapping = (
            self.color_map(hue_name, levels, palette)
            if hue_name
            else {None: style.text_color}
        )
        for level in levels:
            subset = work if level is None else work[work[hue_name] == level]
            xv = subset[x_name].to_numpy(dtype=float)
            yv = subset[y_name].to_numpy(dtype=float)
            ok = np.isfinite(xv) & np.isfinite(yv)
            xv, yv = xv[ok], yv[ok]
            if xv.size < 3:
                continue
            grid = np.linspace(xv.min(), xv.max(), 100)
            label = None
            if trend == "linear":
                slope, intercept = np.polyfit(xv, yv, 1)
                fitted = intercept + slope * grid
                if trend_show_eq:
                    r = float(np.corrcoef(xv, yv)[0, 1])
                    label = (
                        f"y = {slope:,.4g}x + {intercept:,.4g} "
                        f"(R2 = {r * r:.3f})"
                    )
            elif trend == "poly":
                coefs = np.polyfit(xv, yv, degree)
                fitted = np.polyval(coefs, grid)
                if trend_show_eq:
                    label = f"poly degree {degree}"
            elif trend == "theilsen":
                from scipy import stats as sps

                slope, intercept, _lo, _hi = sps.theilslopes(yv, xv)
                fitted = intercept + slope * grid
                if trend_show_eq:
                    label = f"Theil-Sen slope = {slope:,.4g}"
            elif trend == "lowess":
                order = np.argsort(xv)
                smoothed = self._smooth_series(
                    pd.Series(yv[order]),
                    "lowess",
                    (trend_kwargs or {}).get("frac", 0.3),
                    xv[order],
                )
                grid = xv[order]
                fitted = smoothed.to_numpy()
            else:
                raise ValueError(
                    f"trend={trend!r} is not recognised. Use 'linear', "
                    f"'poly', 'lowess', 'theilsen' or None."
                )
            ax.plot(
                grid,
                fitted,
                color=mapping[level],
                linewidth=1.6,
                zorder=style.data_zorder + 1,
                label=label,
            )
            if trend_ci and trend == "linear" and xv.size > 3:
                self._draw_trend_band(
                    ax, xv, yv, grid, trend_ci, mapping[level], style
                )

    def _draw_trend_band(
        self,
        ax: Axes,
        xv: np.ndarray,
        yv: np.ndarray,
        grid: np.ndarray,
        level: float,
        colour: str,
        style: StyleConfig,
    ) -> None:
        """Confidence band around an OLS fit."""
        from scipy import stats as sps

        n = xv.size
        slope, intercept = np.polyfit(xv, yv, 1)
        resid = yv - (intercept + slope * xv)
        dof = max(n - 2, 1)
        sigma = float(np.sqrt(np.sum(resid**2) / dof))
        mean_x = float(xv.mean())
        sxx = float(np.sum((xv - mean_x) ** 2)) or 1.0
        se = sigma * np.sqrt(1.0 / n + (grid - mean_x) ** 2 / sxx)
        crit = float(sps.t.ppf(0.5 + level / 200.0, dof))
        centre = intercept + slope * grid
        ax.fill_between(
            grid,
            centre - crit * se,
            centre + crit * se,
            color=colour,
            alpha=0.14,
            linewidth=0,
            zorder=style.data_zorder - 1,
        )

    def _label_points(
        self,
        ax: Axes,
        work: pd.DataFrame,
        xv: np.ndarray,
        yv: np.ndarray,
        label_points: str,
        style: StyleConfig,
    ) -> None:
        """Annotate individual points, capped at 30 labels."""
        if label_points not in work.columns:
            raise ValueError(f"label_points={label_points!r} is not a column.")
        index = np.arange(len(work))
        if len(work) > 30:
            slope, intercept = np.polyfit(
                xv[np.isfinite(xv) & np.isfinite(yv)],
                yv[np.isfinite(xv) & np.isfinite(yv)],
                1,
            )
            distance = np.abs(yv - (intercept + slope * xv))
            index = np.argsort(distance)[-30:]
            self._warn_once(
                f"scatter-labels:{label_points}",
                f"label_points would draw {len(work):,} labels; only "
                f"the 30 furthest from the fit are shown.",
            )
        texts = work[label_points].to_numpy()
        for i in index:
            ax.annotate(
                str(texts[i]),
                xy=(xv[i], yv[i]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=style.annot_size * 0.8,
                color=style.subtitle_color,
            )

    def _annotate_corr(
        self,
        *,
        ax: Axes,
        work: pd.DataFrame,
        x_name: str,
        y_name: str,
        hue_name: str | None,
        methods: Any,
        style: StyleConfig,
        space: str,
    ) -> None:
        """Render correlation coefficients, naming the space used."""
        from scipy import stats as sps

        wanted = [methods] if isinstance(methods, str) else list(methods)
        funcs = {
            "pearson": sps.pearsonr,
            "spearman": sps.spearmanr,
            "kendall": sps.kendalltau,
        }
        groups = (
            [
                (lv, work[work[hue_name] == lv])
                for lv in self.ordered_levels(work[hue_name])
            ]
            if hue_name
            else [(None, work)]
        )
        lines: list[str] = []
        for level, subset in groups:
            xv = subset[x_name].to_numpy(dtype=float)
            yv = subset[y_name].to_numpy(dtype=float)
            ok = np.isfinite(xv) & np.isfinite(yv)
            if ok.sum() < 3:
                continue
            for method in wanted:
                if method not in funcs:
                    raise ValueError(
                        f"annotate_corr={method!r} is not recognised. "
                        f"Use 'pearson', 'spearman' or 'kendall'."
                    )
                stat = funcs[method](xv[ok], yv[ok])
                prefix = f"{level}: " if level is not None else ""
                lines.append(
                    f"{prefix}{method} r = {stat[0]:.3f}, "
                    f"{_format_p(float(stat[1]))}"
                )
        lines.append(f"computed on {space} data")
        ax.annotate(
            "\n".join(lines),
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=style.annot_size * 0.9,
            color=style.text_color,
        )

    def _scatter_joint(
        self,
        *,
        work: pd.DataFrame,
        x_name: str,
        y_name: str,
        hue_name: str | None,
        style_cfg: StyleConfig,
        figsize: tuple[float, float] | None,
        title: Any,
        subtitle: str | None,
        palette: Any,
        hue_order: Sequence[Any] | None,
        alpha: float | None,
        save_as: str | Path | None,
        show: bool | None,
    ) -> Figure:
        """Joint layout with marginal distributions on both axes."""
        with self._style_scope(style_cfg):
            fig = plt.figure(figsize=figsize or (8.0, 8.0), dpi=self.dpi)
            spec = fig.add_gridspec(
                2,
                2,
                width_ratios=[4, 1],
                height_ratios=[1, 4],
                wspace=0.06,
                hspace=0.06,
            )
            main = fig.add_subplot(spec[1, 0])
            top_ax = fig.add_subplot(spec[0, 0], sharex=main)
            right_ax = fig.add_subplot(spec[1, 1], sharey=main)
            self._draw_scatter(
                ax=main,
                data=work,
                x_name=x_name,
                y_name=y_name,
                hue_name=hue_name,
                style=style_cfg,
                treat={},
                size=None,
                size_range=(20, 200),
                size_norm="linear",
                style_by=None,
                alpha=alpha,
                trend=None,
                trend_kwargs=None,
                trend_per_group=True,
                trend_ci=None,
                trend_show_eq=False,
                degree=2,
                annotate_corr=None,
                jitter=None,
                sample=None,
                identity_line=False,
                label_points=None,
                kind="points",
                cmap=None,
                palette=palette,
                hue_order=hue_order,
                top_n=None,
                log_scale=None,
                log_x=False,
                log_y=False,
                transform=None,
                nonpositive="raise",
                xlabel=None,
                ylabel=None,
                legend=True,
                legend_loc="best",
                kwargs={},
            )
            for axis, column, orient in (
                (top_ax, x_name, "x"),
                (right_ax, y_name, "y"),
            ):
                sns.histplot(
                    data=work,
                    **{orient: column},
                    hue=hue_name,
                    palette=(
                        self.color_map(
                            hue_name,
                            self.ordered_levels(work[hue_name]),
                            palette,
                        )
                        if hue_name
                        else None
                    ),
                    color=None if hue_name else style_cfg.categorical[0],
                    ax=axis,
                    bins=30,
                    element="step",
                    fill=True,
                    alpha=0.35,
                    legend=False,
                )
                axis.set_xlabel("")
                axis.set_ylabel("")
                axis.tick_params(labelbottom=False, labelleft=False, length=0)
                sns.despine(ax=axis, left=True, bottom=True)
            top = self._place_header(
                fig,
                style_cfg,
                self._auto_title(title, f"{y_name} vs {x_name}"),
                subtitle,
                style_cfg.margins,
            )
            fig.subplots_adjust(top=top, left=0.12, right=0.95)
            self._save(fig, save_as)
            if self.show if show is None else show:
                plt.show()
        return fig

    # -- corr_heatmap ------------------------------------------------
    def corr_heatmap(
        self,
        df: pd.DataFrame | None = None,
        *,
        columns: Sequence[str] | None = None,
        hue: str | None = None,
        method: str | Callable = "pearson",
        mask: str | np.ndarray | None = "upper",
        annot: bool = True,
        annot_fmt: str = ".2f",
        annot_n: bool = False,
        cmap: Any = None,
        vmin: float = -1,
        vmax: float = 1,
        center: float = 0,
        cluster: bool = False,
        linkage_method: str = "average",
        show_dendrogram: bool = False,
        order: Sequence[str] | None = None,
        threshold: float | None = None,
        min_periods: int | None = None,
        target: str | None = None,
        diff_vs: Any = None,
        drop_constant: bool = True,
        include_low_cardinality: bool = False,
        cbar: bool = True,
        square: bool = True,
        top_n: int | None = None,
        transform: Any = None,
        nonpositive: Nonpositive = "raise",
        log_scale: Any = None,
        treat_as: Mapping[str, str] | None = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | bool | None = None,
        subtitle: str | None = None,
        panel_title: str | bool | None = None,
        style: Mapping[str, Any] | str | None = None,
        save_as: str | Path | None = None,
        show: bool | None = None,
        **kwargs: Any,
    ) -> Axes | Figure:
        """Show the pairwise association structure of numeric columns.

        Column selection is where the shared resolver pays off: with
        ``columns=None`` every column that :meth:`resolve_kind` calls
        numeric is used, and numeric-but-low-cardinality columns are
        **excluded by default**, because a Pearson correlation over a
        1-5 code is usually misleading. The exclusions are always named
        in a warning.

        The colormap never autoscales: ``vmin=-1``, ``vmax=1`` and
        ``center=0`` are fixed, because autoscaling makes weak
        correlations look strong.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Frame to correlate.
        columns : sequence of str, optional
            Restrict to these columns, raising if any is non-numeric
            unless ``treat_as`` says otherwise.
        hue : str, optional
            Compute one matrix per level and lay them out side by side
            with a shared scale.
        method : str or callable, default "pearson"
            ``"pearson"`` (linear), ``"spearman"`` (monotonic,
            rank-based, the better choice for skewed or ordinal data),
            ``"kendall"``, or a callable.
        mask : {"upper", "lower", None} or ndarray, default "upper"
            Hide the redundant triangle.
        threshold : float, optional
            Blank cells whose absolute correlation is below this.
        cluster : bool, default False
            Reorder by hierarchical clustering on ``1 - |corr|``.
            Conflicts with ``order``.
        target : str, optional
            Skip the matrix and draw a sorted, signed bar chart of each
            feature's correlation with this column, reusing
            :meth:`barplot`.
        diff_vs : optional
            With ``hue``, plot each group as a difference from this
            reference level.
        include_low_cardinality : bool, default False
            Pull numeric-low-cardinality columns back in.

        Returns
        -------
        matplotlib.axes.Axes or matplotlib.figure.Figure
            Axes for one matrix, Figure when ``hue`` creates a grid.

        Raises
        ------
        ValueError
            When fewer than two usable columns remain, when ``cluster``
            and ``order`` are both given, or when ``log_scale`` is used
            — a correlation matrix has no continuous axes.

        Examples
        --------
        >>> import matplotlib, pandas as pd
        >>> matplotlib.use("Agg")
        >>> df = pd.DataFrame(
        ...     {
        ...         "a": [1.5, 2.5, 3.5, 4.5],
        ...         "b": [2.1, 3.9, 6.2, 7.8],
        ...         "c": [9.5, 7.1, 5.2, 1.3],
        ...     }
        ... )
        >>> ax = EDAPlotter(df).corr_heatmap()
        >>> ax.get_title(loc="left")
        ''
        """
        if log_scale is not None and log_scale is not False:
            raise ValueError(
                "log_scale does not apply to a correlation matrix: its "
                "axes are column names, not continuous values. Use "
                "transform= to change the columns before correlating."
            )
        if cluster and order is not None:
            raise ValueError(
                "cluster=True and order= both set the row order; pass "
                "only one."
            )
        frame, style_cfg, treat, _ = self._resolve_common(
            df, style=style, figsize=figsize, treat_as=treat_as
        )
        picked = self._select_corr_columns(
            frame, columns, treat, include_low_cardinality, drop_constant
        )
        work = self._transform_corr_columns(
            frame, picked, transform, nonpositive, method
        )

        if target is not None:
            return self._corr_target(
                work=work,
                picked=picked,
                target=target,
                hue=hue,
                method=method,
                min_periods=min_periods,
                top_n=top_n,
                style_cfg=style_cfg,
                figsize=figsize,
                title=title,
                subtitle=subtitle,
                save_as=save_as,
                show=show,
                kwargs=kwargs,
            )

        if hue is not None:
            return self._corr_by_group(
                work=work,
                picked=picked,
                hue=hue,
                method=method,
                min_periods=min_periods,
                diff_vs=diff_vs,
                mask=mask,
                annot=annot,
                annot_fmt=annot_fmt,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                threshold=threshold,
                square=square,
                style_cfg=style_cfg,
                figsize=figsize,
                title=title,
                subtitle=subtitle,
                save_as=save_as,
                show=show,
                kwargs=kwargs,
            )

        matrix = self._corr_matrix(work, picked, method, min_periods)
        counts = self._pairwise_counts(work, picked)
        self.last_corr_counts = counts
        if cluster:
            picked = self._cluster_order(matrix, linkage_method)
        elif order is not None:
            picked = [c for c in order if c in matrix.columns]
        matrix = matrix.loc[picked, picked]
        counts = counts.loc[picked, picked]

        owns_figure = ax is None
        size = figsize or self._corr_figsize(len(picked), show_dendrogram)
        with self._style_scope(style_cfg):
            if owns_figure:
                fig, axes = self._new_figure(style_cfg, size)
                ax = axes[0, 0]
            else:
                fig = ax.figure
            self._draw_heatmap(
                ax=ax,
                matrix=matrix,
                counts=counts,
                style=style_cfg,
                mask=mask,
                annot=annot,
                annot_fmt=annot_fmt,
                annot_n=annot_n,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                threshold=threshold,
                cbar=cbar,
                square=square,
                kwargs=kwargs,
            )
            self._panel_title(
                ax, style_cfg, self._auto_title(panel_title, None)
            )
            if owns_figure:
                top = self._place_header(
                    fig,
                    style_cfg,
                    self._auto_title(
                        title,
                        f"{self._method_name(method)} correlation",
                    ),
                    subtitle,
                    style_cfg.margins,
                )
                fig.subplots_adjust(
                    top=top, left=0.22, right=0.98, bottom=0.22
                )
                self._save(fig, save_as)
                if self.show if show is None else show:
                    plt.show()
        return ax

    @staticmethod
    def _method_name(method: Any) -> str:
        """Readable name for a correlation method."""
        return (
            method
            if isinstance(method, str)
            else getattr(method, "__name__", "custom")
        )

    @staticmethod
    def _corr_figsize(n: int, dendrogram: bool) -> tuple[float, float]:
        """Scale the figure with the size of the matrix."""
        side = float(np.clip(1.0 + 0.62 * n, 5.0, 18.0))
        return (side + (1.6 if dendrogram else 0.0), side)

    def _select_corr_columns(
        self,
        frame: pd.DataFrame,
        columns: Sequence[str] | None,
        treat: Mapping[str, str],
        include_low_cardinality: bool,
        drop_constant: bool,
    ) -> list[str]:
        """Pick the columns to correlate, explaining every exclusion."""
        if columns is not None:
            bad = [
                c
                for c in columns
                if not is_numeric_dtype(frame[c].dtype)
                and treat.get(c) != "numeric"
            ]
            if bad:
                raise ValueError(
                    f"columns {bad} are not numeric. Drop them, or pass "
                    f"treat_as={{{bad[0]!r}: 'numeric'}}."
                )
            picked = list(columns)
        else:
            picked = []
            excluded: list[str] = []
            for column in frame.columns:
                if not is_numeric_dtype(frame[column].dtype):
                    continue
                kind = self.resolve_kind(frame[column], treat.get(column))
                if kind == "numeric" or include_low_cardinality:
                    picked.append(column)
                else:
                    excluded.append(column)
            if excluded:
                warnings.warn(
                    f"corr_heatmap excluded {len(excluded)} "
                    f"numeric-but-low-cardinality column(s) "
                    f"({', '.join(excluded[:6])}): a correlation over a "
                    f"small integer code is usually misleading. Pass "
                    f"include_low_cardinality=True to keep them.",
                    UserWarning,
                    stacklevel=3,
                )
        if drop_constant:
            constant = [
                c
                for c in picked
                if frame[c].nunique(dropna=True) <= 1
                or not np.isfinite(
                    pd.to_numeric(frame[c], errors="coerce").std()
                )
                or float(pd.to_numeric(frame[c], errors="coerce").std()) == 0.0
            ]
            if constant:
                warnings.warn(
                    f"corr_heatmap dropped {len(constant)} "
                    f"zero-variance column(s) ({', '.join(constant)}); "
                    f"they produce NaN correlations.",
                    UserWarning,
                    stacklevel=3,
                )
                picked = [c for c in picked if c not in constant]
        if len(picked) < 2:
            raise ValueError(
                f"corr_heatmap needs at least 2 usable numeric columns, "
                f"found {len(picked)}."
            )
        return picked

    def _transform_corr_columns(
        self,
        frame: pd.DataFrame,
        picked: Sequence[str],
        transform: Any,
        nonpositive: Nonpositive,
        method: Any,
    ) -> pd.DataFrame:
        """Apply per-column transforms before correlating."""
        if transform is None:
            return frame
        if isinstance(method, str) and method in {"spearman", "kendall"}:
            self._warn_once(
                f"corr-monotone:{method}",
                f"method={method!r} is rank-based and therefore already "
                f"invariant under any monotone transform, so transform= "
                f"is a no-op here. It only changes a Pearson result.",
            )
        spec = (
            {c: transform for c in picked}
            if not isinstance(transform, Mapping)
            else dict(transform)
        )
        if spec.get("all") is not None:
            spec = {c: spec["all"] for c in picked}
        work = frame.copy()
        for column, how in spec.items():
            if column not in picked or how is None:
                continue
            values, meta = _apply_transform(
                work[column],
                how,
                nonpositive,
                column=column,
                lang=self.lang,
            )
            work[column] = values
            self._last_transform = meta
        return work

    def _corr_matrix(
        self,
        frame: pd.DataFrame,
        picked: Sequence[str],
        method: Any,
        min_periods: int | None,
    ) -> pd.DataFrame:
        """Correlation matrix over the selected columns."""
        numeric = frame[list(picked)].apply(pd.to_numeric, errors="coerce")
        # pandas accepts min_periods=None only for Pearson; the rank
        # methods require an int, so normalise it here.
        return numeric.corr(
            method=method,
            min_periods=1 if min_periods is None else int(min_periods),
        )

    @staticmethod
    def _pairwise_counts(
        frame: pd.DataFrame, picked: Sequence[str]
    ) -> pd.DataFrame:
        """Pairwise complete observation counts, matching the matrix."""
        present = frame[list(picked)].notna().astype(float)
        return pd.DataFrame(
            present.T.to_numpy() @ present.to_numpy(),
            index=list(picked),
            columns=list(picked),
        ).astype(int)

    @staticmethod
    def _cluster_order(matrix: pd.DataFrame, linkage_method: str) -> list[str]:
        """Order columns by hierarchical clustering on ``1 - |corr|``."""
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        distance = 1.0 - matrix.abs().fillna(0.0).to_numpy()
        np.fill_diagonal(distance, 0.0)
        distance = (distance + distance.T) / 2.0
        tree = linkage(squareform(distance, checks=False), linkage_method)
        leaves = dendrogram(tree, no_plot=True)["leaves"]
        return [matrix.columns[i] for i in leaves]

    def _draw_heatmap(
        self,
        *,
        ax: Axes,
        matrix: pd.DataFrame,
        counts: pd.DataFrame | None,
        style: StyleConfig,
        mask: Any,
        annot: bool,
        annot_fmt: str,
        annot_n: bool,
        cmap: Any,
        vmin: float,
        vmax: float,
        center: float,
        threshold: float | None,
        cbar: bool,
        square: bool,
        kwargs: Mapping[str, Any],
    ) -> Axes:
        """Render one correlation matrix."""
        n = matrix.shape[0]
        values = matrix.copy()
        if threshold is not None:
            values = values.where(values.abs() >= threshold)
        hidden = self._corr_mask(mask, n)
        if annot and n > 25:
            self._warn_once(
                "corr-annot",
                f"The matrix is {n}x{n}; per-cell annotations would be "
                f"unreadable, so they are disabled.",
            )
            annot = False
        font = float(np.clip(11.0 - 0.28 * n, 5.5, 10.0))
        labels = None
        if annot and annot_n and counts is not None:
            labels = np.array(
                [
                    [
                        ""
                        if not np.isfinite(values.iat[i, j])
                        else (
                            f"{values.iat[i, j]:{annot_fmt}}\n"
                            f"n={counts.iat[i, j]:,}"
                        )
                        for j in range(n)
                    ]
                    for i in range(n)
                ]
            )
        sns.heatmap(
            values,
            mask=hidden,
            cmap=cmap if cmap is not None else style.diverging_cmap(),
            vmin=vmin,
            vmax=vmax,
            center=center,
            annot=labels if labels is not None else annot,
            fmt="" if labels is not None else annot_fmt,
            annot_kws={"fontsize": font},
            square=square,
            linewidths=0.6,
            linecolor="white",
            cbar=cbar,
            cbar_kws={"shrink": 0.7, "pad": 0.02},
            ax=ax,
            **dict(kwargs),
        )
        # A correlation matrix keeps its frame: the cells are the plot,
        # so there is no value axis for a grid to serve.
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(style.grid_color)
        ax.grid(False)
        ax.tick_params(length=0, labelsize=min(style.tick_size, font + 1))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        return ax

    @staticmethod
    def _corr_mask(mask: Any, n: int) -> np.ndarray | None:
        """Build the triangle mask."""
        if mask is None:
            return None
        if isinstance(mask, np.ndarray):
            return mask
        if mask == "upper":
            return np.triu(np.ones((n, n), dtype=bool))
        if mask == "lower":
            return np.tril(np.ones((n, n), dtype=bool))
        raise ValueError(
            f"mask={mask!r} is not recognised. Use 'upper', 'lower', "
            f"None, or a boolean array."
        )

    def _corr_by_group(
        self,
        *,
        work: pd.DataFrame,
        picked: Sequence[str],
        hue: str,
        method: Any,
        min_periods: int | None,
        diff_vs: Any,
        mask: Any,
        annot: bool,
        annot_fmt: str,
        cmap: Any,
        vmin: float,
        vmax: float,
        center: float,
        threshold: float | None,
        square: bool,
        style_cfg: StyleConfig,
        figsize: tuple[float, float] | None,
        title: Any,
        subtitle: str | None,
        save_as: str | Path | None,
        show: bool | None,
        kwargs: Mapping[str, Any],
    ) -> Figure:
        """One comparable matrix per level of ``hue``."""
        floor = max(self.min_group_size, 30)
        counts = work[hue].value_counts()
        levels = [
            lv
            for lv in self.ordered_levels(work[hue])
            if int(counts.get(lv, 0)) >= floor
        ]
        skipped = [
            lv for lv in self.ordered_levels(work[hue]) if lv not in levels
        ]
        if skipped:
            warnings.warn(
                f"corr_heatmap skipped {len(skipped)} group(s) of "
                f"{hue!r} with fewer than {floor} rows "
                f"({', '.join(map(str, skipped[:5]))}).",
                UserWarning,
                stacklevel=3,
            )
        if not levels:
            raise ValueError(f"No level of {hue!r} has at least {floor} rows.")
        matrices = {
            lv: self._corr_matrix(
                work[work[hue] == lv], picked, method, min_periods
            )
            for lv in levels
        }
        if diff_vs is not None:
            if diff_vs not in matrices:
                raise ValueError(
                    f"diff_vs={diff_vs!r} is not one of the retained "
                    f"levels: {', '.join(map(str, levels))}."
                )
            reference = matrices[diff_vs]
            matrices = {
                lv: matrices[lv] - reference for lv in levels if lv != diff_vs
            }
            levels = [lv for lv in levels if lv != diff_vs]
            vmin, vmax = -1.0, 1.0

        ncols = min(3, len(levels))
        nrows = int(np.ceil(len(levels) / ncols))
        side = self._corr_figsize(len(picked), False)[0]
        with self._style_scope(style_cfg):
            fig, axes = self._new_figure(
                style_cfg,
                figsize or (side * ncols * 0.85, side * nrows * 0.9),
                nrows,
                ncols,
            )
            flat = axes.ravel()
            for index, level in enumerate(levels):
                self._draw_heatmap(
                    ax=flat[index],
                    matrix=matrices[level],
                    counts=None,
                    style=style_cfg,
                    mask=mask,
                    annot=annot,
                    annot_fmt=annot_fmt,
                    annot_n=False,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    center=center,
                    threshold=threshold,
                    cbar=index == len(levels) - 1,
                    square=square,
                    kwargs=kwargs,
                )
                self._panel_title(flat[index], style_cfg, str(level))
            for spare in flat[len(levels) :]:
                spare.set_visible(False)
            default = f"{self._method_name(method)} correlation by {hue}"
            if diff_vs is not None:
                default = f"{default} - difference from {diff_vs}"
            top = self._place_header(
                fig,
                style_cfg,
                self._auto_title(title, default),
                subtitle,
                style_cfg.margins,
            )
            fig.subplots_adjust(
                top=top, left=0.12, right=0.97, bottom=0.16, wspace=0.35
            )
            self._save(fig, save_as)
            if self.show if show is None else show:
                plt.show()
        return fig

    def _corr_target(
        self,
        *,
        work: pd.DataFrame,
        picked: Sequence[str],
        target: str,
        hue: str | None,
        method: Any,
        min_periods: int | None,
        top_n: int | None,
        style_cfg: StyleConfig,
        figsize: tuple[float, float] | None,
        title: Any,
        subtitle: str | None,
        save_as: str | Path | None,
        show: bool | None,
        kwargs: Mapping[str, Any],
    ) -> Axes | Figure:
        """Signed bar chart of each feature's correlation with target.

        Delegates the drawing to :meth:`barplot` rather than
        reimplementing bars.
        """
        if target not in work.columns:
            raise ValueError(
                f"target={target!r} is not a column of the frame."
            )
        columns = [c for c in picked if c != target]
        if not columns:
            raise ValueError(
                f"target={target!r} leaves no other numeric column to "
                f"correlate against."
            )

        def series_for(subset: pd.DataFrame) -> pd.Series:
            matrix = self._corr_matrix(
                subset, [*columns, target], method, min_periods
            )
            return matrix[target].drop(labels=[target])

        if hue is None:
            values = series_for(work)
            frame = values.rename("corr").reset_index()
            frame.columns = ["feature", "corr"]
        else:
            rows = []
            for level in self.ordered_levels(work[hue]):
                part = series_for(work[work[hue] == level])
                rows.append(
                    pd.DataFrame(
                        {
                            "feature": part.index,
                            "corr": part.to_numpy(),
                            hue: level,
                        }
                    )
                )
            frame = pd.concat(rows, ignore_index=True)

        ranking = (
            frame.groupby("feature", observed=True)["corr"]
            .apply(lambda s: float(np.nanmax(np.abs(s))))
            .sort_values(ascending=False)
        )
        if top_n:
            ranking = ranking.head(top_n)
        frame = frame[frame["feature"].isin(ranking.index)]
        levels = list(ranking.index)

        plotter = EDAPlotter(
            frame,
            palette=self.palette,
            save_dir=self.save_dir,
            show=False,
            random_state=self.random_state,
            lang=self.lang,
            style_overrides={},
        )
        plotter.style = style_cfg
        if hue is None:
            plotter.set_color_map(
                "feature",
                {
                    name: (
                        style_cfg.positive
                        if frame.loc[frame["feature"] == name, "corr"].iloc[0]
                        >= 0
                        else style_cfg.negative
                    )
                    for name in levels
                },
            )
        return plotter.barplot(
            x="corr",
            y="feature",
            hue=hue,
            orientation="horizontal",
            estimator="mean",
            errorbar=None,
            order=list(reversed(levels)),
            sort=None,
            annotate=True,
            annot_fmt="{:+.2f}",
            baseline=0.0,
            figsize=figsize or (8.0, max(3.5, 0.42 * len(levels) + 2.0)),
            title=self._auto_title(
                title,
                f"{self._method_name(method)} correlation with {target}",
            ),
            subtitle=subtitle,
            xlabel="correlation",
            ylabel="",
            legend=hue is not None,
            save_as=save_as,
            show=show,
            **dict(kwargs),
        )

    # -- dispatcher and overview -------------------------------------
    _KINDS = (
        "barplot",
        "histplot",
        "boxplot",
        "qqplot",
        "triple_plot",
        "distribution_report",
        "curveplot",
        "scatterplot",
        "corr_heatmap",
        "report_numeric",
        "summary_grid",
    )

    def plot(self, kind: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch to any plotting method by name.

        Lets the whole utility be driven from a config dict or a loop.

        Parameters
        ----------
        kind : str
            Method name, e.g. ``"histplot"`` or ``"triple_plot"``.
        *args, **kwargs
            Forwarded verbatim to that method.

        Returns
        -------
        Any
            Whatever the target method returns.

        Raises
        ------
        ValueError
            When ``kind`` is not a known method.

        Examples
        --------
        >>> import matplotlib, pandas as pd
        >>> matplotlib.use("Agg")
        >>> df = pd.DataFrame({"v": [1.5, 2.25, 3.75, 4.1]})
        >>> ax = EDAPlotter(df).plot("histplot", x="v")
        >>> ax.get_xlabel()
        'v'
        """
        if kind not in self._KINDS:
            close = get_close_matches(kind, self._KINDS, n=3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ValueError(
                f"plot(kind={kind!r}) is not a known plot. Available: "
                f"{', '.join(self._KINDS)}.{hint}"
            )
        return getattr(self, kind)(*args, **kwargs)

    def summary_grid(
        self,
        df: pd.DataFrame | None = None,
        target: str | None = None,
        *,
        columns: Sequence[str] | None = None,
        col_wrap: int = 3,
        max_columns: int = 12,
        top_n: int = 10,
        treat_as: Mapping[str, str] | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | bool | None = None,
        subtitle: str | None = None,
        style: Mapping[str, Any] | str | None = None,
        save_as: str | Path | None = None,
        show: bool | None = None,
    ) -> Figure:
        """One sensible panel per column, laid out in a grid.

        Picks the plot from the resolved kind: a histogram for numeric
        columns, a bar chart for categorical ones, and a boxplot against
        ``target`` when one is given. This is where the shared kind
        resolution pays for itself.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Frame to summarise.
        target : str, optional
            Numeric column to plot each other column against.
        columns : sequence of str, optional
            Restrict to these columns.
        col_wrap : int, default 3
            Panels per row.
        max_columns : int, default 12
            Cap on the number of panels.
        top_n : int, default 10
            Level cap for categorical panels.

        Returns
        -------
        matplotlib.figure.Figure
            The grid.

        Examples
        --------
        >>> import matplotlib, pandas as pd
        >>> matplotlib.use("Agg")
        >>> df = pd.DataFrame({"v": [1.5, 2.25, 3.75, 4.1], "g": list("aabb")})
        >>> fig = EDAPlotter(df).summary_grid()
        >>> len(fig.axes)
        2
        """
        frame, style_cfg, treat, _ = self._resolve_common(
            df, style=style, figsize=figsize, treat_as=treat_as
        )
        picked = list(columns or frame.columns)
        if target is not None:
            picked = [c for c in picked if c != target]
        usable: list[tuple[str, Kind]] = []
        for column in picked:
            kind = self.resolve_kind(frame[column], treat.get(column))
            if kind == "categorical" and frame[column].nunique() > 60:
                continue
            usable.append((column, kind))
        usable = usable[:max_columns]
        if not usable:
            raise ValueError("summary_grid found no plottable columns.")

        ncols = min(col_wrap, len(usable))
        nrows = int(np.ceil(len(usable) / ncols))
        with self._style_scope(style_cfg):
            fig, axes = self._new_figure(
                style_cfg,
                figsize or (5.4 * ncols, 3.8 * nrows),
                nrows,
                ncols,
            )
            flat = axes.ravel()
            for index, (column, kind) in enumerate(usable):
                panel = flat[index]
                try:
                    self._summary_panel(
                        panel,
                        frame,
                        column,
                        kind,
                        target,
                        top_n,
                        treat,
                        style_cfg,
                    )
                except (ValueError, TypeError) as exc:
                    panel.axis("off")
                    panel.text(
                        0.5,
                        0.5,
                        f"{column}\n({exc})"[:140],
                        ha="center",
                        va="center",
                        fontsize=style_cfg.annot_size * 0.8,
                        color=style_cfg.subtitle_color,
                    )
            for spare in flat[len(usable) :]:
                spare.set_visible(False)
            default = (
                "Column overview"
                if target is None
                else f"Column overview against {target}"
            )
            top = self._place_header(
                fig,
                style_cfg,
                self._auto_title(title, default),
                subtitle,
                style_cfg.margins,
            )
            fig.subplots_adjust(
                top=top,
                left=0.08,
                right=0.97,
                bottom=0.10,
                wspace=0.3,
                hspace=0.75,
            )
            self._save(fig, save_as)
            if self.show if show is None else show:
                plt.show()
        return fig

    def _summary_panel(
        self,
        ax: Axes,
        frame: pd.DataFrame,
        column: str,
        kind: Kind,
        target: str | None,
        top_n: int,
        treat: Mapping[str, str],
        style_cfg: StyleConfig,
    ) -> None:
        """Draw the panel that suits one column."""
        scoped = {"enabled": style_cfg.enabled}
        if target is not None and kind == "categorical":
            self.boxplot(
                frame,
                x=column,
                y=target,
                ax=ax,
                top_n=top_n,
                panel_title=column,
                legend=False,
                treat_as=treat,
                style=scoped,
            )
        elif target is not None and kind == "numeric":
            self.scatterplot(
                frame,
                x=column,
                y=target,
                ax=ax,
                panel_title=column,
                legend=False,
                treat_as=treat,
                style=scoped,
            )
        elif kind == "categorical":
            self.barplot(
                frame,
                x=column,
                ax=ax,
                top_n=top_n,
                panel_title=column,
                legend=False,
                treat_as=treat,
                style=scoped,
            )
        else:
            self.histplot(
                frame,
                x=column,
                ax=ax,
                panel_title=column,
                legend=False,
                treat_as=treat,
                style=scoped,
            )


def _demo_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame covering every kind the resolver must handle.

    Parameters
    ----------
    n : int, default 400
        Number of rows.
    seed : int, default 0
        Seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Continuous, skewed, high-cardinality string, low-cardinality
        integer code, boolean, datetime and ~10% null columns.

    Examples
    --------
    >>> sorted(_demo_frame(20).columns)[:3]
    ['continuous', 'counts', 'flag']
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "continuous": rng.normal(50, 12, n),
            "skewed": rng.lognormal(3.0, 1.1, n),
            "counts": rng.poisson(3, n),
            "rating": rng.integers(1, 6, n),
            "flag": rng.random(n) > 0.55,
            "group": rng.choice(list("abcd"), n),
            "label": [f"id-{i:04d}" for i in range(n)],
            "when": pd.date_range("2024-01-01", periods=n, freq="D"),
        }
    )
    holes = rng.choice(n, size=max(1, n // 10), replace=False)
    frame.loc[holes, "with_nulls"] = np.nan
    frame["with_nulls"] = rng.normal(10, 2, n)
    frame.loc[holes, "with_nulls"] = np.nan
    return frame


def _smoke(out_dir: Path) -> int:
    """Call every implemented method and report how many calls ran."""
    frame = _demo_frame()
    plotter = EDAPlotter(frame, save_dir=out_dir, show=False)
    calls: list[tuple[str, Callable[[], Any]]] = [
        ("histplot", lambda: plotter.histplot(x="continuous")),
        ("histplot/cat", lambda: plotter.histplot(x="group")),
        ("histplot/bool", lambda: plotter.histplot(x="flag")),
        ("histplot/code", lambda: plotter.histplot(x="rating")),
        ("histplot/nulls", lambda: plotter.histplot(x="with_nulls")),
        (
            "histplot/hue",
            lambda: plotter.histplot(
                x="continuous", hue="group", element="step"
            ),
        ),
        (
            "histplot/facet",
            lambda: plotter.histplot(x="continuous", facet="group"),
        ),
        (
            "histplot/log",
            lambda: plotter.histplot(x="skewed", log_scale=True),
        ),
        (
            "histplot/transform",
            lambda: plotter.histplot(x="skewed", transform="log"),
        ),
        ("boxplot", lambda: plotter.boxplot(x="group", y="continuous")),
        (
            "boxplot/flip",
            lambda: plotter.boxplot(x="continuous", y="group"),
        ),
        (
            "boxplot/code",
            lambda: plotter.boxplot(x="rating", y="continuous"),
        ),
        ("boxplot/single", lambda: plotter.boxplot(y="continuous")),
        (
            "boxplot/wide",
            lambda: plotter.boxplot(y=["continuous", "with_nulls"]),
        ),
        (
            "boxplot/hue",
            lambda: plotter.boxplot(x="group", y="continuous", hue="flag"),
        ),
        (
            "boxplot/facet",
            lambda: plotter.boxplot(x="rating", y="continuous", facet="group"),
        ),
        (
            "boxplot/violin",
            lambda: plotter.boxplot(
                x="group", y="continuous", violin=True, overlay="strip"
            ),
        ),
        ("qqplot", lambda: plotter.qqplot(x="continuous")),
        (
            "qqplot/band",
            lambda: plotter.qqplot(
                x="continuous", conf_band=0.95, annotate_test="shapiro"
            ),
        ),
        (
            "qqplot/hue",
            lambda: plotter.qqplot(x="continuous", hue="flag"),
        ),
        (
            "qqplot/facet",
            lambda: plotter.qqplot(x="continuous", facet="group"),
        ),
        (
            "qqplot/lognorm",
            lambda: plotter.qqplot(x="skewed", dist="lognorm"),
        ),
        ("triple_plot", lambda: plotter.triple_plot(x="continuous")),
        (
            "triple_plot/stacked",
            lambda: plotter.triple_plot(x="skewed", layout="stacked"),
        ),
        (
            "triple_plot/grid",
            lambda: plotter.triple_plot(
                x="continuous", layout="grid", save_as="triple_grid.png"
            ),
        ),
        (
            "triple_plot/column",
            lambda: plotter.triple_plot(x="counts", layout="column"),
        ),
        (
            "triple_plot/hue",
            lambda: plotter.triple_plot(x="continuous", hue="flag"),
        ),
        (
            "triple_plot/transform",
            lambda: plotter.triple_plot(x="skewed", transform="log"),
        ),
        (
            "triple_plot/es",
            lambda: plotter.triple_plot(x="continuous", lang="es"),
        ),
        (
            "report_numeric",
            lambda: plotter.report_numeric(
                columns=["continuous", "skewed"], save_dir=out_dir
            ),
        ),
        ("barplot", lambda: plotter.barplot(x="group", y="continuous")),
        ("barplot/count", lambda: plotter.barplot(x="group")),
        (
            "barplot/flip",
            lambda: plotter.barplot(x="continuous", y="group"),
        ),
        (
            "barplot/crosstab",
            lambda: plotter.barplot(x="group", y="flag"),
        ),
        (
            "barplot/hue",
            lambda: plotter.barplot(x="group", y="continuous", hue="flag"),
        ),
        (
            "barplot/stacked",
            lambda: plotter.barplot(
                x="group", hue="flag", stacked=True, estimator="count"
            ),
        ),
        (
            "barplot/pct",
            lambda: plotter.barplot(
                x="group",
                hue="flag",
                stacked=True,
                normalize=True,
                estimator="count",
            ),
        ),
        (
            "barplot/annot",
            lambda: plotter.barplot(
                x="group", y="continuous", annotate=True, baseline="mean"
            ),
        ),
        (
            "barplot/facet",
            lambda: plotter.barplot(x="rating", y="continuous", facet="group"),
        ),
        ("curveplot", lambda: plotter.curveplot(x="when", y="skewed")),
        (
            "curveplot/hue",
            lambda: plotter.curveplot(
                x="when", y="skewed", hue="group", dashes=True
            ),
        ),
        (
            "curveplot/ecdf",
            lambda: plotter.curveplot(x="continuous", kind="ecdf"),
        ),
        (
            "curveplot/kde",
            lambda: plotter.curveplot(x="continuous", kind="kde"),
        ),
        (
            "curveplot/step",
            lambda: plotter.curveplot(x="when", y="counts", kind="step"),
        ),
        (
            "curveplot/function",
            lambda: plotter.curveplot(
                x=(0.0, 6.28),
                y=[np.sin, np.cos],
                kind="function",
                hue_order=["sin", "cos"],
            ),
        ),
        (
            "curveplot/smooth",
            lambda: plotter.curveplot(
                x="when", y="skewed", smooth="rolling", smooth_window=7
            ),
        ),
        (
            "curveplot/facet",
            lambda: plotter.curveplot(x="when", y="skewed", facet="group"),
        ),
        (
            "scatterplot",
            lambda: plotter.scatterplot(x="continuous", y="skewed"),
        ),
        (
            "scatterplot/hue-cat",
            lambda: plotter.scatterplot(
                x="continuous", y="skewed", hue="group"
            ),
        ),
        (
            "scatterplot/hue-num",
            lambda: plotter.scatterplot(
                x="continuous", y="skewed", hue="with_nulls"
            ),
        ),
        (
            "scatterplot/size+style",
            lambda: plotter.scatterplot(
                x="continuous",
                y="skewed",
                size="counts",
                style_by="flag",
            ),
        ),
        (
            "scatterplot/trend",
            lambda: plotter.scatterplot(
                x="continuous",
                y="skewed",
                trend="linear",
                trend_show_eq=True,
                annotate_corr="pearson",
            ),
        ),
        (
            "scatterplot/marginals",
            lambda: plotter.scatterplot(
                x="continuous", y="skewed", marginals=True
            ),
        ),
        (
            "scatterplot/facet",
            lambda: plotter.scatterplot(
                x="continuous", y="skewed", facet="group"
            ),
        ),
        ("corr_heatmap", lambda: plotter.corr_heatmap()),
        (
            "corr_heatmap/spearman",
            lambda: plotter.corr_heatmap(method="spearman", cluster=True),
        ),
        (
            "corr_heatmap/target",
            lambda: plotter.corr_heatmap(target="skewed"),
        ),
        (
            "corr_heatmap/hue",
            lambda: plotter.corr_heatmap(hue="flag"),
        ),
        (
            "plot/dispatch",
            lambda: plotter.plot("histplot", x="continuous"),
        ),
        ("summary_grid", lambda: plotter.summary_grid()),
        (
            "summary_grid/target",
            lambda: plotter.summary_grid(target="continuous"),
        ),
    ]
    failures = 0
    for label, call in calls:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                call()
            print(f"  ok    {label}")
        except Exception as exc:  # noqa: BLE001 - smoke test
            failures += 1
            print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
        finally:
            plt.close("all")
    print(f"\n{len(calls) - failures}/{len(calls)} calls succeeded.")
    return failures


if __name__ == "__main__":
    import sys
    import tempfile

    mpl.use("Agg")
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(1 if _smoke(Path(tmp)) else 0)
