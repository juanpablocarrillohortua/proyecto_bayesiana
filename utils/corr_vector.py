import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_target_correlation(
    df,
    columns,
    target_col,
    title=None,
    subtitle=None,
    figsize=(6, 10),
    dpi=150,
):
    """Genera un heatmap de correlación de Pearson estilo editorial contra una
    sola variable.

    Parámetros:
    -----------
    df : pd.DataFrame
        El DataFrame con los datos.
    columns : list
        Lista de nombres de columnas numéricas a considerar en el análisis.
    target_col : str
        Nombre de la variable objetivo contra la cual se calculará la
        correlación.
    title : str, opcional
        Título principal del gráfico.
    subtitle : str, opcional
        Subtítulo descriptivo.
    figsize : tuple, por defecto (6, 10)
        Dimensiones de la figura (ancho, alto).
    dpi : int, por defecto 150
        Resolución de la imagen.
    """
    # 1. Validar que target_col esté en las columnas especificadas
    if target_col not in columns:
        columns = list(columns) + [target_col]

    # Filtrar solo las columnas seleccionadas y numéricas
    df_sub = df[columns].select_dtypes(include=[np.number])

    if target_col not in df_sub.columns:
        raise ValueError(
            f"La columna objetivo '{target_col}' no es numérica "
            "o no está en el DataFrame."
        )

    # 2. Calcular la matriz de correlación de Pearson
    corr_matrix = df_sub.corr(method="pearson")

    corr_target = (
        corr_matrix[[target_col]]
        .drop(index=target_col)
        .sort_values(by=target_col, ascending=False)
    )

    # 3. Configurar estilo y figura
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    custom_cmap = sns.diverging_palette(15, 210, s=75, l=60, as_cmap=True)

    # 4. Dibujar el Heatmap con anotaciones
    sns.heatmap(
        corr_target,
        annot=True,  # Muestra los valores numéricos
        fmt=".2f",  # Formato con 2 decimales
        cmap=custom_cmap,
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=1.5,  # Separación blanca entre celdas
        linecolor="white",
        cbar_kws={
            "shrink": 0.7,
            "label": "Coeficiente de Pearson",
        },
        ax=ax,
        annot_kws={"size": 10, "weight": "bold"},
    )

    # 5. Títulos alineados a la izquierda
    title_text = title if title else f"Correlación con '{target_col}'"
    subtitle_text = (
        subtitle
        if subtitle
        else "Correlaciones de Pearson entre variables seleccionadas"
    )

    # Posicionamiento del título/subtítulo sobre los ejes
    plt.text(
        x=-0.5,
        y=-0.8,
        s=title_text,
        fontsize=15,
        fontweight="bold",
        color="#2C3E50",
        ha="left",
    )
    plt.text(
        x=-0.5,
        y=-0.3,
        s=subtitle_text,
        fontsize=9,
        color="#7F8C8D",
        ha="left",
    )

    # Ajustes finales del eje
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.xticks([])  # Ocultar etiquetas del eje X
    plt.yticks(
        rotation=0, fontsize=10, color="#333333"
    )  # Nombres de variables horizontales

    plt.tight_layout()
    return fig, ax
