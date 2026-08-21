import folium
from folium.plugins import MarkerCluster


def create_map(df, latitude_col="latitude", longitude_col="longitude"):
    """
    Create a folium map with a marker cluster for the given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing latitude and longitude columns.
    latitude_col (str): Name of the latitude column in the DataFrame.
    longitude_col (str): Name of the longitude column in the DataFrame.

    Returns:
    folium.Map: A folium map object with clustered markers.
    """
    clean_df = df.dropna(subset=[latitude_col, longitude_col])
    if clean_df.empty:
        raise ValueError(
            """The DataFrame is empty after dropping rows with NaN values in
            latitude and longitude columns."""
        )

    # create the map
    mapa = folium.Map(
        location=[
            clean_df[latitude_col].mean(),
            clean_df[longitude_col].mean(),
        ],
        zoom_start=11,
        tiles="OpenStreetMap",
    )

    # create a marker cluster
    marker_cluster = MarkerCluster().add_to(mapa)

    # add the points to the cluster instead of directly to the map
    for _, row in clean_df.iterrows():
        folium.Marker(
            location=[row[latitude_col], row[longitude_col]],
            popup=f"Lat: {row[latitude_col]}, Lon: {row[longitude_col]}",
        ).add_to(marker_cluster)

    return mapa
