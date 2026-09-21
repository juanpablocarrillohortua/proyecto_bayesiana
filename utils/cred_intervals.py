"""Intervalos de credibilidad para una distribución de ``scipy.stats``.

Cuatro funciones públicas, una por método:

=========================  ===========================================
``colas_iguales_exacto``   ``ppf`` en las dos colas
``hpdi_exacto``            mínimo ancho por ``scipy.optimize.minimize``
``colas_iguales_mcmc``     percentiles sobre una muestra simulada
``hpdi_mcmc``              ventana deslizante sobre la muestra ordenada
=========================  ===========================================

Cada una recibe la distribución y el nivel y calcula su intervalo por su
cuenta. :func:`credible_intervals` solo las llama y arma la tabla.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --- STYLE PALETTE (same tokens as triple_plot.py) ---
C_DARK = "#2c3e50"
C_MID = "#34495e"
C_SOFT = "#7f8c8d"
C_GRID = "#eeeeee"
C_FILL = "#d6eaf8"

#: One colour per cell of the grid, so a row in the table and its bar on
#: the figure are the same colour.
ROW_COLORS = {
    ("Exacto", "Colas Iguales"): "#2b5c8f",
    ("Exacto", "HPDI"): "#2e8b57",
    ("MCMC", "Colas Iguales"): "#e06d53",
    ("MCMC", "HPDI"): "#b07aa1",
}

#: Tail mass left outside the density curve drawn by ``plot=True``.
PLOT_TAIL = 0.001

__all__ = [
    "colas_iguales_exacto",
    "colas_iguales_mcmc",
    "credible_intervals",
    "hpdi_exacto",
    "hpdi_mcmc",
]


# ----------------------------------------------------------------------
# Los cuatro métodos
# ----------------------------------------------------------------------
def colas_iguales_exacto(
    dist: Any, nivel: float = 0.95
) -> tuple[float, float]:
    """Intervalo de colas iguales, exacto.

    Generaliza ``stats.gamma.ppf(0.025, a=alpha, scale=1 / beta)`` y su
    gemelo en 0.975.
    """
    cola = (1.0 - nivel) / 2.0
    return float(dist.ppf(cola)), float(dist.ppf(1.0 - cola))


def hpdi_exacto(dist: Any, nivel: float = 0.95) -> tuple[float, float]:
    """Intervalo de máxima densidad, exacto.

    Minimiza el ancho sobre el límite inferior: el superior se coloca en
    ``ppf(cdf(lb) + nivel)``, así que la cobertura ya es correcta por
    construcción y solo queda el ancho. Generaliza el bloque
    ``minimize(interval_width, ...)`` del script; ``ppf(0)`` es el 0
    literal que el script usa como piso del soporte de la Gamma.
    """

    def ancho(lb: Any) -> float:
        low = float(np.ravel(lb)[0])
        return float(dist.ppf(dist.cdf(low) + nivel)) - low

    res = minimize(
        ancho,
        x0=[float(dist.ppf((1.0 - nivel) / 2.0))],
        bounds=[(float(dist.ppf(0.0)), float(dist.ppf(1.0 - nivel)))],
    )
    low = float(res.x[0])
    return low, float(dist.ppf(dist.cdf(low) + nivel))


def colas_iguales_mcmc(
    dist: Any,
    nivel: float = 0.95,
    n_sim: int = 100_000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Intervalo de colas iguales sobre una muestra simulada.

    Generaliza ``np.random.gamma(shape, scale, size)`` seguido de
    ``np.percentile(muestras, 2.5)`` y ``97.5``.
    """
    rng = np.random.default_rng(random_state)
    muestras = dist.rvs(n_sim, random_state=rng)
    cola = 100.0 * (1.0 - nivel) / 2.0
    return (
        float(np.percentile(muestras, cola)),
        float(np.percentile(muestras, 100.0 - cola)),
    )


def hpdi_mcmc(
    dist: Any,
    nivel: float = 0.95,
    n_sim: int = 100_000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Intervalo de máxima densidad sobre una muestra simulada.

    Ordena las muestras y desliza una ventana que contiene la masa
    pedida, quedándose con la posición más angosta. Generaliza
    ``calc_hpdi_mcmc`` del script.
    """
    rng = np.random.default_rng(random_state)
    muestras = np.sort(dist.rvs(n_sim, random_state=rng))
    n = muestras.size
    k = int(np.floor(nivel * n))
    anchos = muestras[k:] - muestras[: n - k]
    i = int(np.argmin(anchos))
    return float(muestras[i]), float(muestras[i + k])


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
        color = ROW_COLORS[row["key"]]
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
        [r["label"] for r in rows] if first else [],
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
    title: str | None = None
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
            f"Intervalos de credibilidad al {level:.0%}" if title is None else title,
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
# Tabla
# ----------------------------------------------------------------------
def credible_intervals(
    dist: Any,
    nivel: float = 0.95,
    *,
    n_sim: int = 100_000,
    random_state: int = 42,
    plot: bool = False,
    figsize: tuple[float, float] = (11.0, 6.0),
    show: bool = True,
    title: str | None = None
) -> pd.DataFrame:
    """Llama a los cuatro métodos y arma la tabla.

    Devuelve una fila por celda de la grilla ``fuente`` x ``tipo``, con
    las columnas ``fuente``, ``tipo``, ``nivel``, ``limite_inferior``,
    ``limite_superior`` y ``ancho``. Con ``plot=True`` dibuja la densidad
    con los intervalos debajo y deja la figura en ``df.attrs["fig"]``.
    """
    limites = {
        ("Exacto", "Colas Iguales"): colas_iguales_exacto(dist, nivel),
        ("Exacto", "HPDI"): hpdi_exacto(dist, nivel),
        ("MCMC", "Colas Iguales"): colas_iguales_mcmc(
            dist, nivel, n_sim, random_state
        ),
        ("MCMC", "HPDI"): hpdi_mcmc(dist, nivel, n_sim, random_state),
    }

    tabla = pd.DataFrame(
        {
            "fuente": [fuente for fuente, _ in limites],
            "tipo": [tipo for _, tipo in limites],
            "nivel": float(nivel),
            "limite_inferior": [low for low, _ in limites.values()],
            "limite_superior": [high for _, high in limites.values()],
        }
    )
    tabla["ancho"] = tabla["limite_superior"] - tabla["limite_inferior"]

    if plot:
        rows = [
            {
                "key": key,
                "label": f"{key[0]} {key[1]}",
                "low": low,
                "high": high,
            }
            for key, (low, high) in limites.items()
        ]
        tabla.attrs["fig"] = _build_figure(
            [(None, dist, rows)], nivel, figsize, show, title = title
        )

    return tabla
