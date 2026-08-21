# ruff: noqa: N999
"""Diagnóstico de modelos OLS (statsmodels) encapsulado en una clase."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from IPython.display import display
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import durbin_watson, jarque_bera


class DiagnosticoOLS:
    """Agrupa el diagnóstico de un modelo OLS ya ajustado.

    Parámetros
    ----------
    modelo : statsmodels.regression.linear_model.RegressionResultsWrapper
        Resultado de `sm.OLS(...).fit()`.
    """

    def __init__(self, modelo):
        self.modelo = modelo
        self.coeficientes = None
        self.resultados_pruebas = None
        self.df_influencia = None

    # ------------------------------------------------------------------
    # Coeficientes
    # ------------------------------------------------------------------
    def tabla_coeficientes(self, alpha=0.05, hc3=False, mostrar=True):
        """Tabla de coeficientes.

        Si `hc3=True` agrega, junto a los resultados clásicos, los errores
        estándar, t, p-valores e intervalos robustos a heterocedasticidad
        (HC3).
        """
        ci = self.modelo.conf_int(alpha=alpha)
        self.coeficientes = pd.DataFrame(
            {
                "coeficiente": self.modelo.params,
                "error_estandar": self.modelo.bse,
                "t": self.modelo.tvalues,
                "p_valor": self.modelo.pvalues,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            }
        )

        if hc3:
            indice = self.modelo.params.index
            robusto = self.modelo.get_robustcov_results(cov_type="HC3")
            ci_hc3 = pd.DataFrame(robusto.conf_int(alpha=alpha), index=indice)

            self.coeficientes["error_estandar_hc3"] = pd.Series(
                robusto.bse, index=indice
            )
            self.coeficientes["t_hc3"] = pd.Series(
                robusto.tvalues, index=indice
            )
            self.coeficientes["p_valor_hc3"] = pd.Series(
                robusto.pvalues, index=indice
            )
            self.coeficientes["ci_lower_hc3"] = ci_hc3[0]
            self.coeficientes["ci_upper_hc3"] = ci_hc3[1]

        if mostrar:
            with pd.option_context("display.float_format", "{:.6f}".format):
                display(self.coeficientes)

        return self.coeficientes

    def coeficientes_significativos(self, alpha=0.05, hc3=False, mostrar=True):
        """Filtra por p-valor clásico o, si `hc3=True`, por el p-valor HC3.

        En ambos casos la tabla mostrada conserva las columnas robustas y no
        robustas; lo único que cambia es la columna usada para filtrar.
        """
        columna = "p_valor_hc3" if hc3 else "p_valor"

        if self.coeficientes is None or columna not in self.coeficientes:
            self.tabla_coeficientes(alpha=alpha, hc3=hc3, mostrar=False)

        significativos = self.coeficientes[self.coeficientes[columna] < alpha]

        if mostrar:
            with pd.option_context("display.float_format", "{:.6f}".format):
                display(significativos)

        return significativos

    # ------------------------------------------------------------------
    # Bondad de ajuste
    # ------------------------------------------------------------------
    def bondad_ajuste(self):
        print(f"R**2: {self.modelo.rsquared}")
        print(f"R**2 ajustado: {self.modelo.rsquared_adj}")
        return self.modelo.rsquared, self.modelo.rsquared_adj

    def prueba_f_global(self):
        print(f"F global: {self.modelo.fvalue:.2f}")
        print(f"p_valor prueba F: {self.modelo.f_pvalue:.4e}")
        return self.modelo.fvalue, self.modelo.f_pvalue

    def error_estandar_residual(self):
        rse = np.sqrt(self.modelo.mse_resid)

        print(f"Error Estándar Residual (RSE): {rse:,.2f}")
        return rse

    # ------------------------------------------------------------------
    # Gráficos de residuales
    # ------------------------------------------------------------------
    def grafico_residuales_vs_ajustados(self):
        plt.figure(figsize=(8, 5))
        plt.scatter(self.modelo.fittedvalues, self.modelo.resid, alpha=0.65)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Valores ajustados")
        plt.ylabel("Residuales")
        plt.title("Residuales frente a valores ajustados")
        plt.show()

    def grafico_qq(self):
        sm.qqplot(self.modelo.resid, line="45", fit=True)
        plt.title("Gráfico Q-Q de los residuales")
        plt.show()

    # ------------------------------------------------------------------
    # Pruebas de hipótesis sobre los supuestos
    # ------------------------------------------------------------------
    def pruebas_hipotesis(self, mostrar=True):
        jb_stat, jb_p, asimetria, curtosis = jarque_bera(self.modelo.resid)
        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(
            self.modelo.resid, self.modelo.model.exog
        )

        reset = linear_reset(self.modelo, power=2, use_f=True)
        dw = durbin_watson(self.modelo.resid)

        self.resultados_pruebas = pd.DataFrame(
            {
                "prueba": [
                    "Jarque-Bera",
                    "Breusch-Pagan",
                    "Ramsey RESET",
                    "Durbin-Watson",
                ],
                "estadistico": [
                    jb_stat,
                    bp_lm,
                    float(reset.fvalue),
                    dw,
                ],
                "p_valor": [
                    jb_p,
                    bp_lm_p,
                    float(reset.pvalue),
                    np.nan,
                ],
                "H0": [
                    "normalidad",
                    "homocedasticidad",
                    "forma funcional adecuada",
                    "sin autocorrelación de orden 1, lectura heurística",
                ],
            }
        )

        if mostrar:
            display(self.resultados_pruebas.round(5))

        return self.resultados_pruebas

    @staticmethod
    def decision(nombre, p, texto_rechazo, texto_no_rechazo, alpha=0.05):
        if p < alpha:
            return f"{nombre}: se rechaza H0 al {100 * alpha:.0f} %. {texto_rechazo}"  # noqa: E501
        return f"{nombre}: no se rechaza H0 al {100 * alpha:.0f} %. {texto_no_rechazo}"  # noqa: E501

    def decisiones(self, alpha=0.05, mostrar=True):
        """Aplica `decision` sobre la tabla que produce `pruebas_hipotesis`."""
        if self.resultados_pruebas is None:
            self.pruebas_hipotesis(mostrar=False)

        textos = {
            "Jarque-Bera": (
                "los residuales no son normales.",
                "no hay evidencia contra la normalidad de los residuales.",
            ),
            "Breusch-Pagan": (
                "hay evidencia de heterocedasticidad.",
                "no hay evidencia contra la homocedasticidad.",
            ),
            "Ramsey RESET": (
                "hay evidencia de mala especificación de la forma funcional.",
                "no hay evidencia contra la forma funcional propuesta.",
            ),
        }

        lineas = []
        for _, fila in self.resultados_pruebas.iterrows():
            nombre, p = fila["prueba"], fila["p_valor"]

            if pd.isna(p):
                lineas.append(
                    f"{nombre}: estadístico = {fila['estadistico']:.4f}."
                    " Sin p-valor, lectura heurística (≈2 sugiere ausencia de"
                    " autocorrelación de orden 1)."
                )
                continue

            texto_rechazo, texto_no_rechazo = textos[nombre]
            lineas.append(
                self.decision(
                    nombre, p, texto_rechazo, texto_no_rechazo, alpha
                )
            )

        if mostrar:
            for linea in lineas:
                print(linea)

        return lineas

    # ------------------------------------------------------------------
    # Influencia
    # ------------------------------------------------------------------
    def metricas_influencia(self):
        # 1. Obtener objeto de influencia del modelo OLS
        influence = self.modelo.get_influence()

        # 2. Extraer dataframe con métricas
        self.df_influencia = pd.DataFrame(
            {
                "resid_studentized_internal": influence.resid_studentized_internal,  # noqa: E501
                "hat_diag": influence.hat_matrix_diag,
                "cook_d": influence.cooks_distance[
                    0
                ],  # cooks_distance retorna una tupla (distancia, p-valor)
            }
        )

        # 3. Definir Umbrales de Criterio
        n = len(self.modelo.model.endog)  # Número de observaciones
        k = len(
            self.modelo.params
        )  # Número de parámetros (incluyendo intercepto)

        # Umbrales estándar
        criterio_student = 2.0  # |Residuo Estudiantizado| > 2 (outliers en Y)
        criterio_leverage = 2 * (k / n)  # Leverage > 2k/n (outliers en X)
        criterio_cook = 4 / n  # Distancia de Cook > 4/n (puntos influyentes)

        # 4. Filtrar observaciones atípicas e influyentes
        outliers_y = self.df_influencia[
            np.abs(self.df_influencia["resid_studentized_internal"])
            > criterio_student
        ]
        outliers_x = self.df_influencia[
            self.df_influencia["hat_diag"] > criterio_leverage
        ]
        influyentes_cook = self.df_influencia[
            self.df_influencia["cook_d"] > criterio_cook
        ]

        print(
            f"Observaciones con residuos atípicos (|student| > 2): {len(outliers_y)}"  # noqa: E501
        )
        print(
            f"Observaciones con alto leverage (hat > {criterio_leverage:.3f}):"
            f" {len(outliers_x)}"
        )
        print(
            f"Observaciones influyentes (Cook D > {criterio_cook:.3f}):"
            f" {len(influyentes_cook)}"
        )

        return self.df_influencia, outliers_y, outliers_x, influyentes_cook

    def diagnosticar_influencia_ols(self, df):
        """DataFrame con métricas de influencia + gráfico interactivo.

        Retorna:
            diag          : df original + apalancamiento,
                            residual_estudentizado, cook, revisar
            mask_cook     : cook > 4/n
            mask_revisar  : cook, apalancamiento o |residual| fuera de umbral
        """
        influencia = self.modelo.get_influence()

        n, p = int(self.modelo.nobs), int(self.modelo.df_model + 1)
        umbral_h, umbral_cook = 2 * p / n, 4 / n

        diag = df.copy()
        diag["apalancamiento"] = influencia.hat_matrix_diag
        diag["residual_estudentizado"] = influencia.resid_studentized_internal
        diag["cook"] = influencia.cooks_distance[0]

        mask_cook = diag["cook"] > umbral_cook
        mask_revisar = (
            mask_cook
            | (diag["apalancamiento"] > umbral_h)
            | (diag["residual_estudentizado"].abs() > 2)
        )
        diag["revisar"] = mask_revisar

        etiqueta = diag.index.name or "índice"

        fig = px.scatter(
            diag,
            x="apalancamiento",
            y="residual_estudentizado",
            size=diag["cook"]
            + diag["cook"].max() * 0.05,  # evita puntos invisibles
            color="revisar",
            hover_name=[f"{etiqueta}: {i}" for i in diag.index],
            hover_data={
                "apalancamiento": ":.4f",
                "residual_estudentizado": ":.3f",
                "cook": ":.4f",
            },
            labels={
                "apalancamiento": "Apalancamiento",
                "residual_estudentizado": "Residual Estudentizado",
                "cook": "Distancia de Cook",
            },
            title="Diagnóstico de Influencia",
            opacity=0.7,
        )
        fig.add_vline(
            x=umbral_h,
            line_dash="dash",
            line_color="red",
            annotation_text=f"h = {umbral_h:.3f}",
        )
        fig.add_hline(y=2, line_dash="dash", line_color="gray")
        fig.add_hline(y=-2, line_dash="dash", line_color="gray")
        fig.show()

        return diag, mask_cook, mask_revisar

    # ------------------------------------------------------------------
    # Corrida completa
    # ------------------------------------------------------------------
    def reporte_completo(self, df=None, alpha=0.05, hc3=False):
        self.tabla_coeficientes(alpha=alpha, hc3=hc3)
        self.coeficientes_significativos(alpha=alpha, hc3=hc3)
        self.bondad_ajuste()
        self.prueba_f_global()
        self.error_estandar_residual()
        self.grafico_residuales_vs_ajustados()
        self.grafico_qq()
        self.pruebas_hipotesis()
        self.decisiones(alpha=alpha)
        self.metricas_influencia()

        if df is not None:
            return self.diagnosticar_influencia_ols(df)
