# Helper functions
import numpy as np
import geopandas as gpd
import shapely
import plotly.express as px
from shapely.geometry import Polygon, box
from shapely import wkt
import folium
import streamlit as st
import pandas as pd

def create_grid(gdf=None, bounds=None, n_cells=10, overlap=False, crs="EPSG:4326"):
    """Create square grid that covers a geodataframe area
    or a fixed boundary with x-y coords
    returns: a GeoDataFrame of grid polygons
    see https://james-brennan.github.io/posts/fast_gridding_geopandas/
    """

    if bounds != None:
        xmin, ymin, xmax, ymax= bounds
    else:
        xmin, ymin, xmax, ymax= gdf.total_bounds

    # get cell size
    cell_size = (xmax-xmin)/n_cells
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax+cell_size, cell_size ):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            x1 = x0-cell_size
            y1 = y0+cell_size
            poly = shapely.geometry.box(x0, y0, x1, y1)
            #print (gdf.overlay(poly, how='intersection'))
            grid_cells.append( poly )

    cells = gpd.GeoDataFrame(grid_cells, columns=['geometry'],
                                     crs=crs)
    if overlap == True:
        cols = ['grid_id','geometry','grid_area']
        cells = cells.sjoin(gdf, how='inner').drop_duplicates('geometry')
    return cells

def create_hex_grid(gdf=None, bounds=None, n_cells=10, overlap=False, crs="EPSG:4326"):
    """Hexagonal grid over geometry.
    See https://sabrinadchan.github.io/data-blog/building-a-hexagonal-cartogram.html
    """

    from shapely.geometry import Polygon
    import geopandas as gpd
    if bounds != None:
        xmin, ymin, xmax, ymax= bounds
    else:
        xmin, ymin, xmax, ymax= gdf.total_bounds

    unit = (xmax-xmin)/n_cells
    a = np.sin(np.pi / 3)
    cols = np.arange(np.floor(xmin), np.ceil(xmax), 3 * unit)
    rows = np.arange(np.floor(ymin) / a, np.ceil(ymax) / a, unit)

    #print (len(cols))
    hexagons = []
    for x in cols:
      for i, y in enumerate(rows):
        if (i % 2 == 0):
          x0 = x
        else:
          x0 = x + 1.5 * unit

        hexagons.append(Polygon([
          (x0, y * a),
          (x0 + unit, y * a),
          (x0 + (1.5 * unit), (y + unit) * a),
          (x0 + unit, (y + (2 * unit)) * a),
          (x0, (y + (2 * unit)) * a),
          (x0 - (0.5 * unit), (y + unit) * a),
        ]))

    grid = gpd.GeoDataFrame({'geometry': hexagons},crs=crs)
    grid["grid_area"] = grid.area
    grid = grid.reset_index().rename(columns={"index": "grid_id"})
    if overlap == True:
        cols = ['grid_id','geometry','grid_area']
        grid = grid.sjoin(gdf, how='inner').drop_duplicates('geometry')
    return grid

def visualize_poi_by_wkt(draw_geometry_wkt, engine):
    # Define POI layers and colors
    df_property_data = None

    poi_layers = {
        'school': ('school_indonesia_', 'purple'),
        'hospital': ('hospital_indonesia_', 'blue'),
        'cemetery': ('cemetery_indonesia_', 'black'),
        'convenience store': ('convenience_store_indonesia_', 'orange'),
        'cafe/restaurant': ('cafe_restaurant_indonesia_', 'purple'),
        'bus stop': ('bus_stop_indonesia_', 'pink'),
        # 'road': ('road_indonesia_', 'brown'),
        'train station': ('train_indonesia_', 'yellow'),
        'government institution': ('government_institution_or_services_', 'gray'), 
        'property_data' : ('property_data_with_geometry', 'green'), 
        'genangan_banjir' : ('genangan_banjir_2020', 'blue'),
        'sutet' : ('sutet_indonesia_', 'red'),
    }

    poi_counts = []
   

    # Draw selected WKT on map
    drawn_geom = wkt.loads(draw_geometry_wkt) 
    center = [drawn_geom.centroid.y, drawn_geom.centroid.x]  # [lat, lon]
    m = folium.Map(location=center, zoom_start=14)
    drawn_gdf = gpd.GeoDataFrame(geometry=[drawn_geom], crs="EPSG:4326")
    drawn_gdf.explore(m=m, color='black', style_kwds={'fillOpacity': 0.05, 'weight': 2}, name="Selected Area")
    st.write(drawn_gdf.to_crs(drawn_gdf.estimate_utm_crs()).area)

    for label, (table_name, color) in poi_layers.items():
        sql = f"""
            SELECT * FROM {table_name}
            WHERE ST_Intersects(geometry, ST_GeomFromText('{draw_geometry_wkt}', 4326))
        """
        gdf = gpd.read_postgis(sql, engine, geom_col='geometry')
        count = len(gdf)
        poi_counts.append({"POI": label, "Count": count})

        if not gdf.empty:
            if(table_name == 'genangan_banjir_2020' or 'sutet_indonesia_'):
                gdf = gdf.clip(drawn_gdf)
                gdf.explore(m=m, color=color, name=label, marker_kwds={'radius': 4, 'fillOpacity': 0.6})
            if(table_name == 'property_data_with_geometry'):
                df_property_data = gdf 
                
            gdf.explore(
                m=m,
                color=color,
                name=label,
                marker_kwds={'radius': 4, 'fillOpacity': 0.6}
            ) 
           

    folium.LayerControl().add_to(m)

    poi_df = pd.DataFrame(poi_counts)
    bar_fig = px.bar(
        poi_df.sort_values('Count', ascending=False),
        x='POI', y='Count', color='POI',
        title="POI Count in Selected Area",
        text='Count'
    ) 

# try:
    land_price_hist = px.histogram(
        x=df_property_data['kemungkinan_transaksi_tanahm2'],
        nbins=10,
        title="Land Price Distribution in Selected Area (IDR/m²)"
    ) 

    building_price_hist = px.histogram( 
        x=df_property_data['kemungkinan_transaksi_bangunanm2'],
        nbins=10,
        title="Building Price Distribution in Selected Area (IDR/m²)",
    ) 


    # Group by year and calculate the median price
    median_price_per_year = (
        df_property_data.groupby('tahun')['kemungkinan_transaksi_tanahm2']
        .median()
        .reset_index()
        .sort_values('tahun')
    )

    # Create the line chart
    yearly_price_development = px.line(
        median_price_per_year,
        x='tahun',
        y='kemungkinan_transaksi_tanahm2',
        title="Median Estimated Land Price (sqm) per Year",
        labels={'tahun': 'Year', 'kemungkinan_transaksi_tanahm2': 'Median Land Price (IDR/m²)'},
        markers=True  # optional: adds markers on data points
    )



    # Create the bar chart
    kondisi_wilayah_unique = df_property_data['kondisi_wilayah_sekitar'].unique() 
    value_ = [] 
    for i in kondisi_wilayah_unique:
        value_.append(df_property_data[df_property_data['kondisi_wilayah_sekitar'] == i].shape[0])

    surrounding_environment = px.pie(
        names=kondisi_wilayah_unique,
        values=value_,
        title="Surrounding Environment in Selected Area",
    )

   
    return m, bar_fig, land_price_hist, building_price_hist, yearly_price_development, surrounding_environment


