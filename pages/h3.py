import streamlit as st
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from scipy.spatial import Delaunay
import h3
import matplotlib.colors as mcolors
# from osgeo import gdal, ogr, osr
import tempfile
import os

st.set_page_config(
    page_title="Property Price Hexagrid",
    page_icon="🏝️",
    layout="wide"
)

st.title("Property Price Hexagrid")
st.markdown("""
This app performs TIN (Triangulated Irregular Network) interpolation on property price data.
Adjust the hexagon resolution and filters to see how different parameters affect the visualization.
""")

# Functions for data preparation
def prepare_property_data(place_gdf):
    # Ensure the data has the right CRS
    if place_gdf.crs != 'EPSG:32749':
        place_gdf = place_gdf.to_crs('EPSG:32749')
    return place_gdf

def prepare_place_boundary(place_area_gdf):
    # Ensure it has the right CRS
    if place_area_gdf.crs != 'EPSG:32749':
        place_area_gdf = place_area_gdf.to_crs('EPSG:32749')
    return place_area_gdf

# Create hexagonal grid 
def create_hexagonal_grid(place_boundary, resolution=8, buffer_deg=0.02):
    """
    Create a hexagonal grid using the H3 indexing system with fast per-feature clipping.
    
    Parameters:
    - place_boundary: GeoDataFrame with geometries in any CRS
    - resolution: H3 resolution level (default=8)
    - buffer_deg: Buffer in degrees for edge coverage
    
    Returns:
    - GeoDataFrame of clipped H3 hexagons covering the input geometries
    """
    # Step 1: Convert to WGS84 for H3
    gdf_wgs84 = place_boundary.to_crs("EPSG:4326")

    # Step 2: Merge all geometries into one first (fast)
    merged = gdf_wgs84.geometry.values[0] if len(gdf_wgs84) == 1 else gdf_wgs84.unary_union

    # Step 3: Apply buffer once to merged shape
    buffered_geom = merged.buffer(buffer_deg)

    # Step 4: Get bounds and sample grid
    minx, miny, maxx, maxy = buffered_geom.bounds
    lat_step = 0.001
    lng_step = 0.001

    hex_ids = set()
    for lat in np.arange(miny, maxy, lat_step):
        for lng in np.arange(minx, maxx, lng_step):
            pt = Point(lng, lat)
            if buffered_geom.contains(pt):
                h = h3.latlng_to_cell(lat, lng, resolution)
                hex_ids.add(h)

    # Step 5: Expand neighbors to avoid gaps
    all_cells = set(hex_ids)
    for h in hex_ids:
        all_cells.update(h3.grid_ring(h, 1))

    # Step 6: Convert to polygons
    hex_polygons = []
    hex_indices = []
    for h in all_cells:
        boundary = h3.cell_to_boundary(h)
        poly = Polygon([(lng, lat) for lat, lng in boundary])
        hex_polygons.append(poly)
        hex_indices.append(h)

    hex_gdf = gpd.GeoDataFrame(
        {"h3_index": hex_indices},
        geometry=hex_polygons,
        crs="EPSG:4326"
    )

    # Step 7: Project to UTM and clip once
    hex_gdf = hex_gdf.to_crs("EPSG:32749")
    place_utm = place_boundary.to_crs("EPSG:32749")
    clipped = gpd.overlay(hex_gdf, place_utm, how='intersection')
    clipped["area_km2"] = clipped.geometry.area / 1e6
    clipped = clipped[clipped["area_km2"] > 0.00001].reset_index(drop=True)

    return clipped

# Perform TIN interpolation using Delaunay triangulation
def perform_tin_interpolation(property_data, hex_grid):
    # Extract points and values
    points = np.array([(point.x, point.y) for point in property_data.geometry])
    values = np.array(property_data['hpm'])  # Using price per meter (hpm) column
    
    # Create Delaunay triangulation
    tri = Delaunay(points)
    
    # Get centroids of hexagons for interpolation
    hex_centroids = hex_grid.geometry.centroid
    target_points = np.array([(point.x, point.y) for point in hex_centroids])
    
    # Perform interpolation for each hexagon centroid
    interpolated_values = []
    
    for point in target_points:
        # Find the simplex (triangle) containing this point
        simplex_idx = tri.find_simplex(point)
        
        if simplex_idx != -1:  # If point is inside the triangulation
            # Get vertices of the triangle
            triangle_vertices = tri.simplices[simplex_idx]
            
            # Get coordinates of triangle vertices
            triangle_points = points[triangle_vertices]
            
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
            interpolated_value = np.sum(values[triangle_vertices] * b)
        else:
            # Point is outside the triangulation, use nearest neighbor
            distances = np.sqrt(np.sum((points - point)**2, axis=1))
            nearest_idx = np.argmin(distances)
            interpolated_value = values[nearest_idx]
        
        interpolated_values.append(interpolated_value)
    
    # Add interpolated values to hexagon grid
    hex_grid['price_per_m'] = interpolated_values
    
    return hex_grid

# Assign color codes based on values and classification scheme
def assign_color_codes(gdf, value_column, cmap_name, scheme, k):
    """
    Assigns color codes to a GeoDataFrame based on a classification scheme
    Returns the GeoDataFrame with a new 'color_code' column
    """
    try:
        import mapclassify
        
        # Get the values to classify
        values = gdf[value_column].values
        
        # Create the classification
        classifier = mapclassify.classify(
            y=values,
            scheme=scheme,
            k=k
        )
        
        # Get the bin assignments (0-based)
        bin_labels = classifier.yb
        
        # Create a colormap
        cmap = plt.cm.get_cmap(cmap_name, k)
        
        # Assign colors based on bin assignments
        colors = []
        for label in bin_labels:
            rgba = cmap(label)
            hex_color = mcolors.rgb2hex(rgba)
            colors.append(hex_color)
        
        # Add color column to the GeoDataFrame
        gdf['color_code'] = colors
        
        # Also store the bin labels for reference
        gdf['class_bin'] = bin_labels
        
        # Add bin edges information
        gdf['bin_edges'] = [classifier.bins for _ in range(len(gdf))]
        
        return gdf, classifier
    
    except ImportError:
        st.error("Please install mapclassify: pip install mapclassify")
        return gdf, None

# Function to format numbers in Indonesian Rupiah format
def format_rupiah(value):
    """
    Format a number to Indonesian Rupiah format:
    - Uses period (.) as thousand separator
    - No decimal places
    - Adds 'Rp' prefix
    
    Example: 3250000.04 -> Rp3.250.000
    """
    # Round to integer to remove decimal places
    value_int = int(round(value))
    
    # Format with period as thousand separator
    # We use the English locale and replace the comma with a period
    formatted = "{:,}".format(value_int).replace(",", ".")
    
    # Add Rp prefix
    return f"Rp {formatted}"

# Create interactive map with folium
def create_interactive_map(place_boundary, property_data, interpolated_grid, 
                          color_scheme="viridis", classification_scheme="Quantiles", 
                          k=5, basemap="CartoDB positron"):
    # Convert to WGS84 for web mapping
    boundary_wgs84 = place_boundary.to_crs('EPSG:4326')
    property_wgs84 = property_data.to_crs('EPSG:4326')
    grid_wgs84 = interpolated_grid.to_crs('EPSG:4326')
    
    # Apply the classification scheme and get color codes
    grid_wgs84, classifier = assign_color_codes(
        grid_wgs84, 
        'price_per_m', 
        color_scheme, 
        classification_scheme, 
        k
    )
    
    # Create base map
    m = folium.Map(
        location=[grid_wgs84.geometry.centroid.y.mean(), 
                 grid_wgs84.geometry.centroid.x.mean()],
        zoom_start=10,
        tiles=basemap
    )
    
    # Create feature groups for each layer (so they can be toggled)
    hexagon_layer = folium.FeatureGroup(name="Property Prices (Hexagons)")
    point_layer = folium.FeatureGroup(name="Property Points")
    boundary_layer = folium.FeatureGroup(name="Boundary")
    
    # Add hexagon grid with custom colors
    for idx, row in grid_wgs84.iterrows():
        # Create popup content
        popup_content = f"""
        <b>Price per m²:</b> {format_rupiah(row['price_per_m'])}<br>
        <b>Area:</b> {row['area_km2']:.4f} km²<br>
        <b>H3 Index:</b> {row['h3_index']}<br>
        <b>Color Code:</b> {row['color_code']}<br>
        <b>Class:</b> {row['class_bin'] + 1} of {k}
        """
        
        # Create polygon with tooltip and popup
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x, color=row['color_code']: {
                'fillColor': color,
                'fillOpacity': 0.7,
                'color': 'black',
                'weight': 1
            },
            tooltip=f"Price: {format_rupiah(row['price_per_m'])}",
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(hexagon_layer)
    
    # Add property points
    for idx, row in property_wgs84.iterrows():
        # Create popup content with property details
        point_popup = f"""
        <b>Price per m²:</b> {format_rupiah(row['hpm'])}<br>
        """
        
        # Add additional attributes if they exist
        if 'tahun' in row:
            point_popup += f"<b>Year:</b> {row['tahun']}<br>"
        if 'kondisi_wilayah_sekitar' in row:
            point_popup += f"<b>Area Condition:</b> {row['kondisi_wilayah_sekitar']}<br>"
        if 'luas_tanah' in row:
            point_popup += f"<b>Land Area:</b> {row['luas_tanah']} m²<br>"
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            tooltip=f"HPM: {row['hpm']}",
            popup=folium.Popup(point_popup, max_width=300)
        ).add_to(point_layer)
    
    # Add boundary
    folium.GeoJson(
        boundary_wgs84.geometry.__geo_interface__,
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 2
        }
    ).add_to(boundary_layer)
    
    # Add all layers to the map
    hexagon_layer.add_to(m)
    point_layer.add_to(m)
    boundary_layer.add_to(m)
    
    # Add layer control to toggle layers on/off
    folium.LayerControl().add_to(m)
    
    # Add a legend
    # Add a discrete color legend using Folium's built-in system
    if classifier:
        # Import branca for the colormap
        import branca.colormap as cm
        
        # Get the bin edges (classification breaks)
        bins = classifier.bins.tolist()
        
        # Get the exact colors used in the map
        colors = [mcolors.rgb2hex(plt.cm.get_cmap(color_scheme, k)(i)) for i in range(k)]
        
        # Create a StepColormap (discrete colors) instead of LinearColormap
        # This ensures distinct color blocks rather than a gradient
        colormap = cm.StepColormap(
            colors=colors,
            vmin=min(grid_wgs84['price_per_m']),
            vmax=max(grid_wgs84['price_per_m']),
            index=bins,  # Use the classifier bins as the index
            caption=f"Price per m² in {format_rupiah(1)[0:2]} ({classification_scheme})"
        )
        
        # Add the colormap to the map
        colormap.add_to(m)
    
    return m, grid_wgs84

# Main application logic
# File uploader
place_file = st.file_uploader("Upload property point data (GeoJSON)", type=["geojson", "json"])
place_area_file = st.file_uploader("Upload boundary data (GeoJSON)", type=["geojson", "json"])

# Only proceed if both files are uploaded
if place_file and place_area_file:
    # Save uploaded files to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp_place:
        tmp_place.write(place_file.getvalue())
        tmp_place_path = tmp_place.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp_area:
        tmp_area.write(place_area_file.getvalue())
        tmp_area_path = tmp_area.name
    
    # Load GeoDataFrames
    place = gpd.read_file(tmp_place_path)
    place_area = gpd.read_file(tmp_area_path)
    
    # Clean up temporary files
    os.unlink(tmp_place_path)
    os.unlink(tmp_area_path)
    
    # Check if the required columns exist
    if 'hpm' not in place.columns:
        st.error("The property data must have an 'hpm' column (price per meter).")
    else:
        # Sidebar for filters and settings
        st.sidebar.header("Settings")
        
        # Hexagon Resolution
        resolution = st.sidebar.slider(
            "Hexagon Resolution",
            min_value=6,
            max_value=10,
            value=8,
            step=1,
            help="Resolution 6 = Large hexagons (~36.13 km²), Resolution 10 = Small hexagons (~0.015 km²)"
        )
        
        # Color scheme options
        color_scheme = st.sidebar.selectbox(
            "Color Scheme",
            options=["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "YlOrRd", "YlGnBu", "RdYlBu", "RdBu"],
            index=0
        )
        
        # Classification scheme options
        classification_scheme = st.sidebar.selectbox(
            "Classification Scheme",
            options=["Quantiles", "EqualInterval", "NaturalBreaks", "FisherJenks", "HeadTailBreaks", "BoxPlot", "StdMean", "MaximumBreaks"],
            index=0
        )
        
        # Number of classes
        k_classes = st.sidebar.slider(
            "Number of Classes (k)",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Number of color classes to use in the visualization"
        )
        
        basemap = st.sidebar.selectbox(
            "Base Map",
            options=["CartoDB positron", "OpenStreetMap", "Stamen Terrain", "CartoDB dark_matter"],
            index=0
        )
        
        # Add filters section
        st.sidebar.header("Data Filters")
        
        # Filter by Year if 'tahun' column exists
        year_filter = None
        if 'tahun' in place.columns:
            try:
                # Extract years and convert to integers if needed
                years = pd.to_numeric(place['tahun'], errors='coerce').dropna().astype(int).unique()
                years = sorted(years)
                
                # Allow selecting all years or specific years
                year_filter_type = st.sidebar.radio(
                    "Filter by Year",
                    options=["All Years", "Select Years"]
                )
                
                if year_filter_type == "Select Years":
                    year_filter = st.sidebar.multiselect(
                        "Select Years",
                        options=years,
                        default=years
                    )
            except:
                st.sidebar.warning("Could not process 'tahun' column as years.")
        
        # Filter by Area Condition if the column exists
        condition_filter = None
        if 'kondisi_wilayah_sekitar' in place.columns:
            conditions = place['kondisi_wilayah_sekitar'].dropna().unique()
            
            # Allow selecting all conditions or specific ones
            condition_filter_type = st.sidebar.radio(
                "Filter by Area Condition",
                options=["All Conditions", "Select Conditions"]
            )
            
            if condition_filter_type == "Select Conditions":
                condition_filter = st.sidebar.multiselect(
                    "Select Area Conditions",
                    options=conditions,
                    default=conditions
                )
        
        # Filter by Land Area if the column exists
        land_area_filter_type = None
        land_area_min_max = None
        land_area_category = None
        
        if 'luas_tanah' in place.columns:
            try:
                # Convert to numeric
                place['luas_tanah_numeric'] = pd.to_numeric(place['luas_tanah'], errors='coerce')
                
                # Get min and max values
                min_area = place['luas_tanah_numeric'].min()
                max_area = place['luas_tanah_numeric'].max()
                
                # Land area filter type choice
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
                
                elif land_area_filter_type == "Category Filter":
                    land_area_category = st.sidebar.multiselect(
                        "Land Area Categories",
                        options=["< 1,000 m²", "1,000 - 10,000 m²", "> 10,000 m²"],
                        default=["< 1,000 m²", "1,000 - 10,000 m²", "> 10,000 m²"]
                    )
            except:
                st.sidebar.warning("Could not process 'luas_tanah' column as numeric values.")
        
        # Apply filters to the data
        filtered_data = place.copy()
        
        # Apply year filter if selected
        if year_filter and 'tahun' in filtered_data.columns:
            # Convert to numeric to ensure proper filtering
            filtered_data['tahun_numeric'] = pd.to_numeric(filtered_data['tahun'], errors='coerce')
            filtered_data = filtered_data[filtered_data['tahun_numeric'].isin(year_filter)]
        
        # Apply area condition filter if selected
        if condition_filter and 'kondisi_wilayah_sekitar' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['kondisi_wilayah_sekitar'].isin(condition_filter)]
        
        # Apply land area filters if selected
        if 'luas_tanah_numeric' in filtered_data.columns:
            if land_area_filter_type == "Range Filter" and land_area_min_max:
                min_val, max_val = land_area_min_max
                filtered_data = filtered_data[
                    (filtered_data['luas_tanah_numeric'] >= min_val) & 
                    (filtered_data['luas_tanah_numeric'] <= max_val)
                ]
            
            elif land_area_filter_type == "Category Filter" and land_area_category:
                # Create a mask for each category
                mask = pd.Series(False, index=filtered_data.index)
                
                if "< 1,000 m²" in land_area_category:
                    mask = mask | (filtered_data['luas_tanah_numeric'] < 1000)
                
                if "1,000 - 10,000 m²" in land_area_category:
                    mask = mask | ((filtered_data['luas_tanah_numeric'] >= 1000) & 
                                  (filtered_data['luas_tanah_numeric'] <= 10000))
                
                if "> 10,000 m²" in land_area_category:
                    mask = mask | (filtered_data['luas_tanah_numeric'] > 10000)
                
                filtered_data = filtered_data[mask]
        
        # Check if we still have data after filtering
        if len(filtered_data) == 0:
            st.error("No data remains after applying filters. Please adjust your filter settings.")
        else:
            # Prepare data
            st.subheader("Data Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Properties", len(place))
                st.metric("Filtered Properties", len(filtered_data))
                # st.write("Price per Meter (HPM) Statistics (Filtered Data):")
                # st.write(filtered_data['hpm'].describe())
            
            with col2:
                if resolution == 6:
                    hex_size = "~36.13 km²"
                elif resolution == 7:
                    hex_size = "~5.16 km²"
                elif resolution == 8:
                    hex_size = "~0.74 km²"
                elif resolution == 9:
                    hex_size = "~0.11 km²"
                elif resolution == 10:
                    hex_size = "~0.015 km²"
                
                st.metric("Hexagon Resolution", f"H3 Resolution {resolution} ({hex_size})")
                st.metric("Classification Scheme", classification_scheme)
                st.metric("Number of Classes", k_classes)
                
                # Display active filters
                active_filters = []
                if year_filter and year_filter_type == "Select Years":
                    active_filters.append(f"Years: {', '.join(map(str, year_filter))}")
                
                if condition_filter and condition_filter_type == "Select Conditions":
                    active_filters.append(f"Conditions: {', '.join(condition_filter)}")
                
                if land_area_filter_type == "Range Filter" and land_area_min_max:
                    active_filters.append(f"Land Area: {land_area_min_max[0]} - {land_area_min_max[1]} m²")
                elif land_area_filter_type == "Category Filter" and land_area_category:
                    active_filters.append(f"Land Area Categories: {', '.join(land_area_category)}")
                
                if active_filters:
                    st.markdown("**Active Filters:**")
                    for f in active_filters:
                        st.markdown(f"- {f}")
            
            # Process data
            with st.spinner("Processing data... This may take a moment."):
                property_data = prepare_property_data(filtered_data)
                place_boundary = prepare_place_boundary(place_area)
                
                # Create hexagonal grid with user-selected resolution
                hex_grid = create_hexagonal_grid(place_boundary, resolution=resolution)
                
                # Perform TIN interpolation
                interpolated_grid = perform_tin_interpolation(property_data, hex_grid)
                
                # Calculate statistics
                total_hexagons = len(hex_grid)
                avg_area = hex_grid['area_km2'].mean()
                
                st.metric("Total Hexagons Generated", total_hexagons)
                st.metric("Average Hexagon Area", f"{avg_area:.3f} km²")
                
                # Create quartile information
                q1 = interpolated_grid['price_per_m'].quantile(0.25)
                q2 = interpolated_grid['price_per_m'].quantile(0.5)
                q3 = interpolated_grid['price_per_m'].quantile(0.75)
                vmin = interpolated_grid['price_per_m'].min()
                vmax = interpolated_grid['price_per_m'].max()
                
                # Display quartile information
                st.subheader("Price per Meter Quartiles")
                cols = st.columns(5)
                cols[0].metric("Minimum", format_rupiah(vmin))
                cols[1].metric("Q1 (25%)", format_rupiah(q1))
                cols[2].metric("Median", format_rupiah(q2))
                cols[3].metric("Q3 (75%)", format_rupiah(q3))
                cols[4].metric("Maximum", format_rupiah(vmax))
                
                # Create interactive map
                st.subheader("Interactive Map")
                st.info("You can toggle different layers on/off using the layer control in the top right corner of the map.")
                m, grid_with_colors = create_interactive_map(
                    place_boundary, 
                    property_data, 
                    interpolated_grid,
                    color_scheme=color_scheme,
                    classification_scheme=classification_scheme,
                    k=k_classes,
                    basemap=basemap
                )
                
                # Display the map
                folium_static(m, width=1200, height=800)
                
                # Add color legend explanation
                st.subheader("Color Classification Information")
                st.markdown(f"""
                The map uses the **{classification_scheme}** classification scheme with **{k_classes}** classes 
                and the **{color_scheme}** color palette. Each hexagon is colored based on its price per meter value.
                
                Hover over or click on any hexagon to see its exact price value and color code.
                """)
                
                # Add download buttons for the data
                st.subheader("Download Results")
                
                # Create temporary files for download
                with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp_download:
                    grid_with_colors.to_crs('EPSG:4326').to_file(tmp_download.name, driver='GeoJSON')
                    with open(tmp_download.name, 'rb') as f:
                        download_data = f.read()
                    os.unlink(tmp_download.name)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Interpolated Grid (GeoJSON)",
                        data=download_data,
                        file_name=f"interpolated_grid_h3res{resolution}.geojson",
                        mime="application/json"
                    )
                
                with col2:
                    # Create a CSV with hexagon ID, price, and color code
                    if 'color_code' in grid_with_colors.columns:
                        csv_data = grid_with_colors[['h3_index', 'price_per_m', 'area_km2', 'color_code', 'class_bin']].to_csv(index=False)
                    else:
                        csv_data = grid_with_colors[['h3_index', 'price_per_m', 'area_km2']].to_csv(index=False)
                        
                    st.download_button(
                        label="Download Interpolated Values (CSV)",
                        data=csv_data,
                        file_name=f"place_interpolated_values_h3res{resolution}.csv",
                        mime="text/csv"
                    )
                
                # Download filtered data
                st.subheader("Download Filtered Data")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp_filtered:
                    property_data.to_crs('EPSG:4326').to_file(tmp_filtered.name, driver='GeoJSON')
                    with open(tmp_filtered.name, 'rb') as f:
                        filtered_data_download = f.read()
                    os.unlink(tmp_filtered.name)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Filtered Property Data (GeoJSON)",
                        data=filtered_data_download,
                        file_name=f"place_filtered_properties.geojson",
                        mime="application/json"
                    )
                
                with col2:
                    filtered_csv = filtered_data.drop(columns=['geometry']).to_csv(index=False)
                    st.download_button(
                        label="Download Filtered Property Data (CSV)",
                        data=filtered_csv,
                        file_name=f"place_filtered_properties.csv",
                        mime="text/csv"
                    )
else:
    st.info("Please upload both the property point data and boundary files to continue.")
    
    # # Show placeholder/demo image
    # st.subheader("Example Visualization")
    # st.image("https://storage.googleapis.com/kaggle-datasets-images/1862783/3101200/42e0af8ca2cbe5fd0bdfab77d13f7b3e/dataset-card.png", 
    #          caption="Example of hexagonal grid visualization (placeholder)")
    
    # Add information about hexagon sizes
    st.subheader("H3 Hexagon Size Reference")
    
    # Create a table showing hexagon sizes
    size_data = {
        "Resolution": list(range(6, 11)),
        "Avg Hex Area (km²)": ["36.13", "5.16", "0.74", "0.11", "0.015"],
        "Approx Edge Length (m)": ["3,229", "1,220", "461", "174", "65.9"]
    }
    
    st.table(pd.DataFrame(size_data))