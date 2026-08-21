import matplotlib.pyplot as plt
import seaborn as sns


def plot_faceted_histograms(
    df,
    num_col: str,
    cat_col: str,
    palette: str = "mako",
    kde: bool = True,
    col_wrap: int = 3,
    title: str = None,
):
    """
    Genera histogramas individuales en una cuadrícula (subplots) para una
    variable numérica.

    segmentada por cada categoría de una variable cualitativa.
    """
    # 1. Configurar tema limpio
    sns.set_theme(style="white", font="sans-serif")

    # 2. Crear la cuadrícula FacetGrid por la variable categórica
    g = sns.FacetGrid(
        df,
        col=cat_col,
        hue=cat_col,
        palette=palette,
        col_wrap=col_wrap,
        height=3.5,
        aspect=1.2,
        sharey=False,
    )

    # 3. Mapear el histograma en cada recuadro
    g.map_dataframe(
        sns.histplot,
        x=num_col,
        kde=kde,
        element="bars",
        alpha=0.6,
        linewidth=0,
    )

    # 4. Estilizado moderno para cada subgráfico
    g.despine(top=True, right=True)

    # Ajustar títulos de cada recuadro
    g.set_titles(col_template="{col_name}", size=11, weight="bold")

    # Ajustar nombres de ejes
    formatted_num_title = num_col.replace("_", " ").title()
    g.set_axis_labels(formatted_num_title, "Frecuencia", clear_inner=False)
    # Formatear estética de títulos y cuadrícula en cada ax
    for ax in g.axes.flat:
        ax.tick_params(labelbottom=True)
        ax.set_title(
            ax.get_title(), color="#1e293b", pad=10
        )  # Título gris oscuro
        ax.xaxis.label.set_color("#475569")
        ax.yaxis.label.set_color("#475569")
        ax.yaxis.grid(
            True, linestyle="--", alpha=0.3, color="#94a3b8"
        )  # Lógica de grid suave
        ax.set_axisbelow(True)

    # 5. Título principal del Dashboard / Facet
    formatted_cat_title = cat_col.replace("_", " ").title()
    main_title = (
        title
        if title
        else f"Distribución de {formatted_num_title} por {formatted_cat_title}"
    )

    plt.subplots_adjust(top=0.85)  # Espacio para el título principal
    g.fig.suptitle(
        main_title,
        fontsize=14,
        fontweight="bold",
        color="#1e293b",
        x=0.05,
        ha="left",
    )

    plt.show()
