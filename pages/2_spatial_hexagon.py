import streamlit as st
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import folium
import numpy as np
from streamlit_folium import st_folium

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from streamlit_folium import folium_static



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

def apply_tin_interpolation(gdf, value_column):
    """Interpolate missing values using TIN (LinearNDInterpolator).
    
    Args:
        gdf: GeoDataFrame containing the data
        value_column: Name of the column with values to interpolate
        
    Returns:
        GeoDataFrame with interpolated values
    """
    # Check if there are any missing values
    if gdf[value_column].isna().sum() == 0:
        return gdf  # No missing values, return original
    
    known = gdf.dropna(subset=[value_column])
    unknown = gdf[gdf[value_column].isna()]

    # Need at least 3 known points to create a TIN
    if len(known) < 3:
        raise ValueError("At least 3 known points are required for TIN interpolation")

    # Get centroids of polygons as coordinates
    known_coords = np.array(list(known.geometry.centroid.apply(lambda geom: (geom.x, geom.y))))
    known_values = known[value_column].values

    interpolator = LinearNDInterpolator(known_coords, known_values)

    # Interpolate for unknown values
    unknown_coords = np.array(list(unknown.geometry.centroid.apply(lambda geom: (geom.x, geom.y))))
    interpolated_values = interpolator(unknown_coords)

    # Apply the interpolated values back to the original GeoDataFrame
    gdf.loc[gdf[value_column].isna(), value_column] = interpolated_values

    return gdf

def create_hex_grid(gdf=None, bounds=None, n_cells=10, overlap=False, crs="EPSG:4326"):
    """Create a hexagonal grid over the provided geometry."""
    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
    else:
        xmin, ymin, xmax, ymax = gdf.total_bounds

    unit = (xmax - xmin) / n_cells
    a = np.sin(np.pi / 3)  # Height of the hexagon

    cols = np.arange(np.floor(xmin), np.ceil(xmax), 3 * unit)
    rows = np.arange(np.floor(ymin) / a, np.ceil(ymax) / a, unit)

    hexagons = []
    for x in cols:
        for i, y in enumerate(rows):
            x0 = x + 1.5 * unit if i % 2 != 0 else x
            hexagons.append(Polygon([
                (x0, y * a),
                (x0 + unit, y * a),
                (x0 + (1.5 * unit), (y + unit) * a),
                (x0 + unit, (y + (2 * unit)) * a),
                (x0, (y + (2 * unit)) * a),
                (x0 - (0.5 * unit), (y + unit) * a),
            ]))

    grid = gpd.GeoDataFrame({'geometry': hexagons}, crs=crs)
    grid["grid_area"] = grid.area
    grid = grid.reset_index().rename(columns={"index": "grid_id"})

    if overlap:
        grid = grid.sjoin(gdf, how='inner').drop_duplicates('geometry')
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

        # Year filtering
        if "tahun" in raw_points.columns:
            try:
                raw_points["tahun"] = pd.to_numeric(raw_points["tahun"], errors='coerce').astype('Int64')
            except Exception as e:
                st.warning(f"Gagal mengonversi kolom tahun: {e}")

            valid_tahun = raw_points["tahun"].dropna()
            if not valid_tahun.empty:
                tahun_min = valid_tahun.min()
                tahun_max = valid_tahun.max()

                slider_tahun = st.sidebar.slider(
                    "Pilih data tahun",
                    min_value=int(tahun_min),
                    max_value=int(tahun_max),
                    value=(int(tahun_min), int(tahun_max)),
                    step=1,
                    format="%d"
                )

                # Filter from raw_points → so slider remains dynamic
                points = raw_points[raw_points["tahun"].between(slider_tahun[0], slider_tahun[1])]
                st.sidebar.write(f"📍 {len(points)} titik digunakan setelah filter.")
                st.success("✅ Data telah difilter berdasarkan tahun.")
            else:
                st.warning("⚠️ Tidak ada data tahun yang valid.")
                points = raw_points.copy()
        else:
            points = raw_points.copy()

        index_col = [items for items in points.columns if "index" in items] 
        points.drop(index_col, axis=1, inplace=True)

        # Dropdown to select attribute
        attribute_column = st.sidebar.selectbox("Select Attribute for Analysis", options=points.columns)

        st.subheader("📊 Hexagonal Grid Visualization")

        # Grid slider
        n_cells = st.sidebar.slider("Select Grid Size", min_value=50, max_value=100, step=2, value=10)
        hexagrid = create_hex_grid(gdf=polygon, n_cells=n_cells, overlap=True, crs=uniform_crs)

        hexa_col = [items for items in hexagrid.columns if "index" in items] 
        hexagrid.drop(hexa_col, axis=1, inplace=True) 
        if(filtered_data is not None):
            hexagrid_with_data = hexagrid.sjoin(filtered_data, predicate="intersects") 
        else :
            hexagrid_with_data = hexagrid.sjoin(points, predicate="intersects")

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
        cmap = st.sidebar.selectbox("Select Color Palette", options=["viridis", "plasma", "inferno", "magma", "cividis"])
        k = st.sidebar.slider("Select Number of Classes", min_value=2, max_value=10, value=2, step=1)
        generate_button = st.sidebar.button("Generate Map", key="generate_map_button")

        if generate_button:
            st.session_state['generate_button'] = True

        if st.session_state['generate_button'] and hexagrid_avg is not None:
            m = hexagrid_avg.explore(
                column=f'{attribute_column}_mean',
                scheme=scheme,
                k=k,
                cmap=cmap
            ) 
            display_detail = st.radio("Display points and polygon?", ("Yes", "No"), key="display_points_polygon")    
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
    else:
        st.info("📍 Upload a point GeoJSON file to proceed with the analysis.")

# Display generated map if exists
if st.session_state.get('map') is not None:
    folium_static(st.session_state['map'], width=1000, height=500)
