"""
Hub de pruebas de hipótesis para dependencia entre una variable numérica
y una lista de variables categóricas.

Ruta de decisión:
    2 grupos  + normales     -> t de Student (o Welch si no hay
                                homocedasticidad)
    2 grupos  + no normales  -> U de Mann-Whitney
    >2 grupos + normales     -> ANOVA (F) + post-hoc Tukey HSD
    >2 grupos + no normales  -> Kruskal-Wallis + post-hoc Mann-Whitney con
                                Bonferroni
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def _p_normalidad(x):
    """Shapiro-Wilk si n < 50, D'Agostino-Pearson si n >= 50."""
    if len(x) < 3 or np.ptp(x) == 0:
        return 0.0
    try:
        return (
            stats.shapiro(x).pvalue
            if len(x) < 50
            else stats.normaltest(x).pvalue
        )
    except Exception:
        return 0.0


def _posthoc_tukey(grupos, niveles, alpha):
    res = stats.tukey_hsd(*grupos)
    pares = [
        f"{niveles[i]} vs {niveles[j]} (p={res.pvalue[i, j]:.4f})"
        for i, j in combinations(range(len(grupos)), 2)
        if res.pvalue[i, j] < alpha
    ]
    return "; ".join(pares) if pares else "sin pares significativos"


def _posthoc_dunn_bonf(grupos, niveles, alpha):
    pares_idx = list(combinations(range(len(grupos)), 2))
    m = len(pares_idx)
    pares = []
    for i, j in pares_idx:
        p = min(stats.mannwhitneyu(grupos[i], grupos[j]).pvalue * m, 1.0)
        if p < alpha:
            pares.append(f"{niveles[i]} vs {niveles[j]} (p={p:.4f})")
    return "; ".join(pares) if pares else "sin pares significativos"


def hub_pruebas_num_cat(
    df: pd.DataFrame,
    var_num: str,
    vars_cat: list,
    alpha: float = 0.05,
    n_min: int = 3,
):
    """
    Parámetros
    ----------
    df       : DataFrame
    var_num  : str, nombre de la variable numérica
    vars_cat : str o lista de str, variables categóricas a evaluar
    alpha    : nivel de significancia (default 0.05)
    n_min    : tamaño mínimo por grupo para incluirlo (default 3)

    Retorna
    -------
    DataFrame con una fila por combinación numérica-categórica.
    """
    if isinstance(vars_cat, str):
        vars_cat = [vars_cat]

    filas = []
    for cat in vars_cat:
        sub = df[[var_num, cat]].dropna()
        datos = [
            (str(nivel), g[var_num].astype(float).values)
            for nivel, g in sub.groupby(cat, observed=True)
        ]
        datos = [(n, g) for n, g in datos if len(g) >= n_min]
        niveles, grupos = ([d[0] for d in datos], [d[1] for d in datos])
        k = len(grupos)

        base = {"var_numerica": var_num, "var_categorica": cat, "n_grupos": k}

        if k < 2:
            filas.append(
                {
                    **base,
                    "prueba": "no aplica",
                    "estadistico": np.nan,
                    "p_valor": np.nan,
                    "normalidad": np.nan,
                    "homocedasticidad": np.nan,
                    "decision": "no evaluada",
                    "conclusion": f"menos de 2 grupos con n>={n_min}",
                    "post_hoc": "",
                }
            )
            continue

        normal = all(_p_normalidad(g) > alpha for g in grupos)
        p_levene = stats.levene(*grupos, center="median").pvalue
        homo = p_levene > alpha
        post_hoc = ""

        if k == 2:
            if normal:
                est, p = stats.ttest_ind(grupos[0], grupos[1], equal_var=homo)
                prueba = "t de Student" if homo else "t de Welch"
            else:
                est, p = stats.mannwhitneyu(grupos[0], grupos[1])
                prueba = "U de Mann-Whitney"
        else:
            if normal:
                est, p = stats.f_oneway(*grupos)
                prueba = "ANOVA (F)"
                if p < alpha:
                    post_hoc = _posthoc_tukey(grupos, niveles, alpha)
            else:
                est, p = stats.kruskal(*grupos)
                prueba = "Kruskal-Wallis (H)"
                if p < alpha:
                    post_hoc = _posthoc_dunn_bonf(grupos, niveles, alpha)

        rechaza = p < alpha
        filas.append(
            {
                **base,
                "prueba": prueba,
                "estadistico": round(float(est), 4),
                "p_valor": round(float(p), 6),
                "normalidad": "sí" if normal else "no",
                "homocedasticidad": "sí" if homo else "no",
                "decision": "Se rechaza H0" if rechaza else "No se rechaza H0",
                "conclusion": (
                    "Dependencia: la numérica difiere entre categorías"
                    if rechaza
                    else "Independencia: no hay evidencia de diferencia"
                ),
                "post_hoc": post_hoc,
            }
        )

    return pd.DataFrame(filas).sort_values("p_valor").reset_index(drop=True)
