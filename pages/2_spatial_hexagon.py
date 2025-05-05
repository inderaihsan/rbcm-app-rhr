import streamlit as st
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
import folium
import numpy as np
from streamlit_folium import st_folium, folium_static

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

import mapclassify
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import json


filtered_data = None
if 'generate_button' not in st.session_state:
    st.session_state['generate_button'] = False
if 'map_generated' not in st.session_state:
    st.session_state['map_generated'] = False
if 'map' not in st.session_state:
    st.session_state['map'] = None

def apply_tin_interpolation_v2(gdf, value_column):
    """Interpolate missing values using TIN with barycentric coordinate interpolation.
    
    Args:
        gdf: GeoDataFrame containing polygons with some missing values
        value_column: Name of the column with values to interpolate
        
    Returns:
        GeoDataFrame with interpolated values
    """
    # Split into known and unknown points
    known = gdf.dropna(subset=[value_column])
    unknown = gdf[gdf[value_column].isna()]
    
    # Check if there's anything to interpolate
    if len(unknown) == 0:
        return gdf
    
    # Need at least 3 known points for Delaunay triangulation
    if len(known) < 3:
        raise ValueError("At least 3 known points are required for TIN interpolation")
    
    # Get centroids and values of known polygons
    known_points = np.array([(geom.x, geom.y) for geom in known.geometry.centroid])
    known_values = known[value_column].values
    
    # Create Delaunay triangulation
    tri = Delaunay(known_points)
    
    # Get centroids of unknown polygons
    unknown_points = np.array([(geom.x, geom.y) for geom in unknown.geometry.centroid])
    
    # Perform interpolation for each unknown point
    interpolated_values = []
    for point in unknown_points:
        # Find the simplex (triangle) containing this point
        simplex_idx = tri.find_simplex(point)
        
        if simplex_idx != -1:  # If point is inside the triangulation
            # Get vertices of the triangle
            triangle_vertices = tri.simplices[simplex_idx]
            
            # Get coordinates of triangle vertices
            triangle_points = known_points[triangle_vertices]
            
            # Calculate barycentric coordinates
            b = np.zeros(3)
            for i in range(3):
                # Create vectors
                v0 = triangle_points[(i+1) % 3] - triangle_points[i]
                v1 = triangle_points[(i+2) % 3] - triangle_points[i]
                v2 = point - triangle_points[i]
                
                # Calculate areas using cross product
                area_total = np.abs(np.cross(v0, v1))
                area_sub = np.abs(np.cross(v0, v2))
                
                if area_total != 0:
                    b[i] = area_sub / area_total
                else:
                    b[i] = 1/3  # Equal weights if degenerate triangle
            
            # Normalize barycentric coordinates
            b = b / np.sum(b)
            
            # Interpolate using barycentric coordinates
            interpolated_value = np.sum(known_values[triangle_vertices] * b)
        else:
            # Point is outside the triangulation, use nearest neighbor
            distances = np.sqrt(np.sum((known_points - point)**2, axis=1))
            nearest_idx = np.argmin(distances)
            interpolated_value = known_values[nearest_idx]
        
        interpolated_values.append(interpolated_value)
    
    # Apply the interpolated values back to the original GeoDataFrame
    gdf.loc[gdf[value_column].isna(), value_column] = interpolated_values
    
    return gdf

# def apply_tin_interpolation(gdf, value_column):
#     """Interpolate missing values using TIN (LinearNDInterpolator).
    
#     Args:
#         gdf: GeoDataFrame containing the data
#         value_column: Name of the column with values to interpolate
        
#     Returns:
#         GeoDataFrame with interpolated values
#     """
#     # Check if there are any missing values
#     if gdf[value_column].isna().sum() == 0:
#         return gdf  # No missing values, return original
    
#     known = gdf.dropna(subset=[value_column])
#     unknown = gdf[gdf[value_column].isna()]

#     # Need at least 3 known points to create a TIN
#     if len(known) < 3:
#         raise ValueError("At least 3 known points are required for TIN interpolation")

#     # Get centroids of polygons as coordinates
#     known_coords = np.array(list(known.geometry.centroid.apply(lambda geom: (geom.x, geom.y))))
#     known_values = known[value_column].values

#     interpolator = LinearNDInterpolator(known_coords, known_values)

#     # Interpolate for unknown values
#     unknown_coords = np.array(list(unknown.geometry.centroid.apply(lambda geom: (geom.x, geom.y))))
#     interpolated_values = interpolator(unknown_coords)

#     # Apply the interpolated values back to the original GeoDataFrame
#     gdf.loc[gdf[value_column].isna(), value_column] = interpolated_values

#     return gdf


def create_hex_grid(gdf=None, bounds=None, n_cells=10, overlap=False, crs="EPSG:4326", buffer_size=0.05):
    """
    Create a hexagonal grid over the provided geometry with full edge coverage.

    Parameters:
    - gdf: GeoDataFrame with input geometry (preferred).
    - bounds: Optional bounds tuple (xmin, ymin, xmax, ymax).
    - n_cells: Approximate number of hexes horizontally.
    - overlap: If True, keep only hexes that intersect the geometry.
    - crs: Coordinate Reference System (default EPSG:4326).
    - buffer_size: Buffer applied to geometry to ensure coverage.

    Returns:
    - GeoDataFrame of hexagonal polygons covering the area.
    """
    if gdf is not None:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(buffer_size)

    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
    else:
        xmin, ymin, xmax, ymax = gdf.total_bounds

    # Hexagon width and height
    width = (xmax - xmin) / n_cells
    height = np.sqrt(3) * width / 2

    # Adjust bounds to ensure complete coverage
    xmin -= width
    xmax += width
    ymin -= height
    ymax += height

    # Create hex grid
    hexagons = []
    col = 0
    x = xmin
    while x < xmax + width:
        y = ymin if col % 2 == 0 else ymin + height
        while y < ymax + height:
            hexagon = Polygon([
                (x, y),
                (x + width / 2, y + height),
                (x + 1.5 * width, y + height),
                (x + 2.0 * width, y),
                (x + 1.5 * width, y - height),
                (x + width / 2, y - height),
            ])
            hexagons.append(hexagon)
            y += 2 * height
        x += 1.5 * width
        col += 1

    grid = gpd.GeoDataFrame({'geometry': hexagons}, crs=crs)
    grid["grid_area"] = grid.area
    grid = grid.reset_index().rename(columns={"index": "grid_id"})

    if overlap:
        grid = gpd.sjoin(grid, gdf, how='inner', predicate='intersects').drop_duplicates('geometry')

    grid = grid.clip(gdf)  # Clip hexagons to the boundary
    return grid


# Streamlit page configuration
st.set_page_config(page_title="📊 Spatial Grid App", layout="wide")
st.title("📌 Spatial Grid Generator")
st.markdown("Upload your **point** and **polygon** GeoJSON files, choose a grid type, and visualize spatial aggregation.")

@st.cache_resource
def upload_geojson(file):
    """Load GeoJSON file and return a GeoDataFrame."""
    if file is not None:
        gdf = gpd.read_file(file)
        return gdf
    return None

# File upload
point_file = st.sidebar.file_uploader("🟢 Upload Point GeoJSON (with price attribute)", type=["geojson", "json"]) 
poly_file = st.sidebar.file_uploader("🟣 Upload Polygon GeoJSON (study boundary)", type=["geojson", "json"])

if poly_file:
    polygon = upload_geojson(poly_file)
    uniform_crs = polygon.estimate_utm_crs()
    polygon.to_crs(uniform_crs, inplace=True)
    index_col = [items for items in polygon.columns if "index" in items] 
    polygon.drop(index_col, axis=1, inplace=True)
    st.success("✅ Polygon file uploaded and loaded successfully!")

    # Hex grid resolution selector
    # n_cells = st.sidebar.slider(label="Grid Resolution", min_value=10, max_value=50, value=50, step=5)

    if point_file:
        raw_points = upload_geojson(point_file)
        raw_points.to_crs(uniform_crs, inplace=True)
        points = raw_points.copy()

        # Year filtering asdoindwaoksnd owiansodjnaoimnsmdoinwawoisndwoiansodikn
        if "tahun" in raw_points.columns:
            try:
                raw_points["tahun"] = pd.to_numeric(raw_points["tahun"], errors='coerce').astype('Int64')
            except Exception as e:
                st.warning(f"Gagal mengonversi kolom tahun: {e}")

            valid_tahun = raw_points["tahun"].dropna()
            if not valid_tahun.empty:
                unique_years = sorted(valid_tahun.unique())

                year_filter_type = st.sidebar.radio(
                    "Filter berdasarkan tahun",
                    options=["Semua Tahun", "Pilih Tahun"]
                )

                if year_filter_type == "Pilih Tahun":
                    selected_years = st.sidebar.multiselect(
                        "Pilih tahun",
                        options=unique_years,
                        default=unique_years
                    )
                    points = points[points["tahun"].isin(selected_years)]

        # Filter by Area Condition if the column exists
        if 'kondisi_wilayah_sekitar' in points.columns:
            conditions = points['kondisi_wilayah_sekitar'].dropna().unique()
            
            condition_filter_type = st.sidebar.radio(
                "Filter by Area Condition",
                options=["All Conditions", "Select Conditions"]
            )
            
            if condition_filter_type == "Select Conditions":
                selected_conditions = st.sidebar.multiselect(
                    "Select Area Conditions",
                    options=conditions,
                    default=conditions
                )
                points = points[points['kondisi_wilayah_sekitar'].isin(selected_conditions)]

        # Filter by Land Area if the column exists
        if 'luas_tanah' in points.columns:
            try:
                points['luas_tanah_numeric'] = pd.to_numeric(points['luas_tanah'], errors='coerce')
                min_area = points['luas_tanah_numeric'].min()
                max_area = points['luas_tanah_numeric'].max()

                land_area_filter_type = st.sidebar.radio(
                    "Filter by Land Area",
                    options=["All Land Areas", "Range Filter", "Category Filter"]
                )

                if land_area_filter_type == "Range Filter":
                    land_area_min_max = st.sidebar.slider(
                        "Land Area Range (m²)",
                        min_value=float(min_area),
                        max_value=float(max_area),
                        value=(float(min_area), float(max_area)),
                        step=10.0
                    )
                    points = points[
                        points['luas_tanah_numeric'].between(land_area_min_max[0], land_area_min_max[1])
                    ]

                elif land_area_filter_type == "Category Filter":
                    land_area_category = st.sidebar.multiselect(
                        "Land Area Categories",
                        options=["< 1,000 m²", "1,000 - 10,000 m²", "> 10,000 m²"],
                        default=["< 1,000 m²", "1,000 - 10,000 m²", "> 10,000 m²"]
                    )
                    def categorize_area(val):
                        if pd.isna(val):
                            return None
                        if val < 1000:
                            return "< 1,000 m²"
                        elif val <= 10000:
                            return "1,000 - 10,000 m²"
                        else:
                            return "> 10,000 m²"

                    points['area_category'] = points['luas_tanah_numeric'].apply(categorize_area)
                    points = points[points['area_category'].isin(land_area_category)]

            except Exception as e:
                st.sidebar.warning(f"⚠️ Tidak dapat memproses 'luas_tanah': {e}")

        st.sidebar.write(f"📍 {len(points)} titik digunakan setelah filter.")
        st.success("✅ Data telah difilter sesuai kriteria.")
        
        index_col = [items for items in points.columns if "index" in items] 
        points.drop(index_col, axis=1, inplace=True)
        filtered_data = points.copy()

        # Dropdown to select attribute
        numeric_cols = points.select_dtypes(include=['number']).columns
        attribute_column = st.sidebar.selectbox("Select Attribute for Analysis", options=numeric_cols)


        st.subheader("📊 Hexagonal Grid Visualization")

        # Grid slider
        n_cells = st.sidebar.slider("Select Grid Density", min_value=50, max_value=500, step=5, value=50)
        hexagrid = create_hex_grid(gdf=polygon, n_cells=n_cells, overlap=True, crs=uniform_crs)

        hexa_col = [items for items in hexagrid.columns if "index" in items] 
        hexagrid.drop(hexa_col, axis=1, inplace=True) 

        # Use filtered_data if it exists and has rows, otherwise fallback to raw points
        if 'filtered_data' in locals() and not filtered_data.empty:
            points_used = filtered_data
        else:
            points_used = points

        # Perform spatial join with hex grid
        hexagrid_with_data = hexagrid.sjoin(points_used, predicate="intersects")

        # Aggregate statistics
        hexagrid_stats = hexagrid_with_data.groupby("grid_id")[attribute_column].agg(['mean', 'median', 'max', 'min']).reset_index()
        hexagrid_stats = hexagrid_stats.rename(columns={
            'mean': f'{attribute_column}_mean',
            'median': f'{attribute_column}_median',
            'max': f'{attribute_column}_max',
            'min': f'{attribute_column}_min'
        })
        hexagrid_stats[attribute_column] = hexagrid_stats[f'{attribute_column}_mean']

        # Merge
        hexagrid_avg = hexagrid.merge(hexagrid_stats, on="grid_id", how="left")
        hexagrid_avg = apply_tin_interpolation_v2(hexagrid_avg, f'{attribute_column}_mean')

        # Classification scheme and visualization
        scheme = st.sidebar.selectbox("Select Aggregation Scheme", options=[
            'BoxPlot', 'EqualInterval', 'FisherJenks', 'FisherJenksSampled', 'HeadTailBreaks',
            'JenksCaspall', 'JenksCaspallForced', 'JenksCaspallSampled', 'MaxP', 'MaximumBreaks',
            'NaturalBreaks', 'Quantiles', 'Percentiles', 'StdMean'
        ])
        cmap = st.sidebar.selectbox("Select Color Palette", options=["viridis", "plasma", "inferno", "magma", "cividis", "cubehelix", "Blues", "Greens", "Reds", "Purples", "Oranges", "coolwarm", "YlGnBu", "YlOrRd", "BuPu", "GnBu", "PuBu", "OrRd", "RdPu"])
        k = st.sidebar.slider("Select Number of Classes", min_value=2, max_value=10, value=2, step=1)
        # stroke_opacity = st.sidebar.slider("Select Line Transparency", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        weight = st.sidebar.slider("Select Line Weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1) 
        generate_button = st.sidebar.button("Generate Map", key="generate_map_button")

        if generate_button:
            st.session_state['generate_button'] = True

        if st.session_state['generate_button'] and hexagrid_avg is not None:
            m = hexagrid_avg.explore(
                column=f'{attribute_column}_mean',
                scheme=scheme,
                k=k,
                cmap=cmap, 
                style_kwds={
                    # 'opacity': 0.5,
                    'weight': weight,
                }
            ) 
            display_detail = st.radio("Display points and polygon?", ("No", "Yes"), key="display_points_polygon")    
            if display_detail == "Yes":
                points.explore(
                    m=m,
                    column=attribute_column,
                    name="Points",
                    marker_kwds={"radius": 5},
                    scheme=scheme,
                    k=k,
                    cmap=cmap
                ) 

                # polygon.explore(
                #     m=m,
                #     name="Polygon",
                #     color="black",
                #     weight=2,
                #     fill_opacity=0.1
                # )
            folium.LayerControl().add_to(m)
            st.session_state['map'] = m
            st.session_state['map_generated'] = True 
            save_button = st.button("Save Map", key="save_map_button")  
            if save_button:
                values = hexagrid_avg[f'{attribute_column}_mean'].fillna(0)
                # Create classification
                classifier = mapclassify.classify(values, scheme=scheme, k=k)

                # Normalize the values to match the colormap
                norm = colors.BoundaryNorm(classifier.bins, ncolors=plt.get_cmap(cmap).N)
                cmap_func = plt.get_cmap(cmap)

                # Map color codes
                hexagrid_avg['color'] = values.apply(lambda x: colors.to_hex(cmap_func(norm(x))) if pd.notnull(x) else '#d3d3d3') 
                hexagrid_avg['price_per_m'] = hexagrid_avg[f'{attribute_column}_mean']
                hexagrid_avg.to_crs("EPSG:4326", inplace=True)
                hexagrid_avg.to_file("hexagrid_avg.geojson", driver="GeoJSON", layer_options={"ID_GENERATE": "YES"}) 
               
                
                st.success("✅ Map saved successfully!")
    else:
        st.info("📍 Upload a point GeoJSON file to proceed with the analysis.")

# Display generated map if exists
if st.session_state.get('map') is not None:
    folium_static(st.session_state['map'], width=1000, height=1000)
