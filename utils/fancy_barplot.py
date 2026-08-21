from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

INK, TXT, GRID = "#1F2E3D", "#4A4A4A", "#E2E8F0"
PALETTE = [
    "#5BA7DE",
    "#E28585",
    "#7FC4A0",
    "#F0B26B",
    "#9B8FC7",
    "#96A5A5",
    "#6FC3D0",
    "#D98CB3",
]
YLAB = {False: "Conteo", True: "Porcentaje"}

Labels = Literal["none", "count", "pct", "both"]


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()


def _colors(n: int, palette=None):
    if isinstance(palette, str):
        return sns.color_palette(palette, n)
    pal = palette or PALETTE
    return (
        list(pal[:n]) if n <= len(pal) else sns.husl_palette(n, s=0.55, l=0.68)
    )


def _bar_labels(ax, container, counts, pcts, labels: Labels, fontsize=11):
    """Anota una serie de barras con conteos y/o porcentajes."""
    if labels == "none":
        return
    tpl = {
        "count": "{c:,.0f}",
        "pct": "{p:.1f}%",
        "both": "{c:,.0f}\n({p:.1f}%)",
    }[labels]
    txt = [
        "" if c == 0 else tpl.format(c=c, p=p) for c, p in zip(counts, pcts)
    ]
    ax.bar_label(
        container,
        labels=txt,
        padding=4,
        fontsize=fontsize,
        fontweight="bold",
        color=INK,
    )


def _style(ax, title, xlabel, ylabel, pct, rot, headroom):
    """Aplica el estilo visual compartido por todos los gráficos."""
    ax.set_facecolor("white")
    ax.set_title(
        title, loc="left", pad=20, fontsize=18, fontweight="bold", color=INK
    )
    ax.set_xlabel(xlabel, fontsize=12, color=TXT, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=12, color=TXT, labelpad=12)
    ax.yaxis.set_major_formatter(
        mtick.PercentFormatter(decimals=0)
        if pct
        else mtick.FuncFormatter(
            lambda v, _: f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}"
        )
    )
    ax.tick_params(axis="x", rotation=rot, labelsize=10, colors=TXT, length=0)
    ax.tick_params(axis="y", labelsize=10, colors=TXT, length=0)
    if rot:
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color=GRID, linewidth=1.0)
    ax.xaxis.grid(False)
    ax.margins(y=headroom)
    sns.despine(ax=ax, left=True, bottom=True)
    plt.tight_layout()


def fancy_bars(
    df: pd.DataFrame,
    x: str,
    pct: bool = False,
    labels: Labels = "none",
    title: str | None = None,
    order: list | None = None,
    palette=None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (9, 7),
    rot: float = 0,
    top: int | None = None,
) -> plt.Axes:
    """Distribución de una variable categórica.

    pct     : False -> conteos | True -> porcentajes
    labels  : "none" | "count" | "pct" | "both"
    order   : orden de las categorías (default: de mayor a menor)
    palette : nombre de paleta seaborn o lista de colores
    top     : mostrar solo las N categorías más frecuentes
    """
    if labels not in {"none", "count", "pct", "both"}:
        raise ValueError("labels debe ser: 'none', 'count', 'pct' o 'both'")

    vc = df[x].value_counts(dropna=False)
    if order is not None:
        vc = vc.reindex(order).fillna(0)
    if top:
        vc = vc.head(top)

    cats = vc.index.astype(str)
    pcts = vc / vc.sum() * 100

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    colors = _colors(len(vc), palette)
    sns.barplot(
        x=cats,
        y=(pcts if pct else vc).values,
        color=colors[0],
        width=0.62,
        ax=ax,
    )
    for patch, color in zip(ax.containers[0], colors):
        patch.set_facecolor(color)
    _bar_labels(ax, ax.containers[0], vc.values, pcts.values, labels)
    _style(
        ax,
        title or f"Distribución de {_pretty(x)}",
        _pretty(x),
        YLAB[pct],
        pct,
        rot,
        0.14 if labels != "none" else 0.05,
    )
    return ax


def plot_cat_relation(
    df: pd.DataFrame,
    x: str,
    hue: str,
    pct: bool = False,
    norm: Literal["x", "hue", "all"] = "x",
    labels: Labels = "none",
    title: str | None = None,
    order: list | None = None,
    hue_order: list | None = None,
    palette=None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
    rot: float = 0,
    top: int | None = None,
) -> plt.Axes:
    """Relación entre dos variables categóricas (barras agrupadas).

    pct  : False -> conteos | True -> porcentajes
    norm : base del porcentaje -> "x" (cada grupo de x suma 100%),
           "hue" (cada categoría de hue suma 100%) o "all" (sobre el total)
    Resto de argumentos: igual que plot_dist.
    """
    if labels not in {"none", "count", "pct", "both"}:
        raise ValueError("labels debe ser: 'none', 'count', 'pct' o 'both'")

    ct = pd.crosstab(df[x], df[hue])
    order = (
        list(order)
        if order is not None
        else list(ct.sum(1).sort_values(ascending=False).index)
    )
    hue_order = (
        list(hue_order)
        if hue_order is not None
        else list(ct.sum(0).sort_values(ascending=False).index)
    )
    ct = ct.reindex(
        index=order[:top] if top else order, columns=hue_order
    ).fillna(0)

    denom = {
        "x": ct.sum(1).to_numpy()[:, None],
        "hue": ct.sum(0).to_numpy()[None, :],
        "all": ct.to_numpy().sum(),
    }[norm]
    pcts = ct / denom * 100

    long = (
        (pcts if pct else ct)
        .reset_index()
        .melt(id_vars=x, var_name=hue, value_name="_v")
    )
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=long,
        x=x,
        y="_v",
        hue=hue,
        order=list(ct.index),
        hue_order=hue_order,
        palette=_colors(len(hue_order), palette),
        width=0.7,
        ax=ax,
    )

    for cont, col in zip(ax.containers, hue_order):
        _bar_labels(
            ax, cont, ct[col].values, pcts[col].values, labels, fontsize=9
        )

    _style(
        ax,
        title or f"Distribución de {_pretty(hue)} por {_pretty(x)}",
        _pretty(x),
        YLAB[pct],
        pct,
        rot,
        0.14 if labels != "none" else 0.05,
    )

    legend = ax.legend(
        title=_pretty(hue),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        fontsize=10,
    )
    legend.get_title().set(fontweight="bold", color=INK)
    return ax
