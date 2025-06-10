import streamlit as st        
import pandas as pd        
import geopandas as gpd        
from shapely.geometry import Point        
import io        
import zipfile        
import os        
import tempfile      
# from helper import transform_data_to_geodataframe, clean_invalid_infinite_geometries       
from helper import *
from streamlit_folium import st_folium    , folium_static    
import folium        
  
# Set the header of the app        
st.title("Excel to SHP")        
      
# Initialize session state      
if 'df' not in st.session_state:      
    st.session_state.df = None      
if 'gdf' not in st.session_state:      
    st.session_state.gdf = None      
if 'lat_col' not in st.session_state:      
    st.session_state.lat_col = None      
if 'lon_col' not in st.session_state:      
    st.session_state.lon_col = None      
if 'map_created' not in st.session_state:      
    st.session_state.map_created = False      
  
# Upload Excel file        
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])        
  
@st.cache_data  
def load_data(uploaded_file):  
    return pd.read_excel(uploaded_file)    
  
if uploaded_file is not None:      
    # Read the Excel file and store it in session state      
    df = load_data(uploaded_file)     
    st.session_state.df = df      
    st.session_state.lat_col = None      
    st.session_state.lon_col = None      
    st.session_state.map_created = False  # Reset map creation flag when a new file is uploaded      
  
# Display the first few rows of the dataframe      
if st.session_state.df is not None:      
    st.write("Uploaded Data:")      
    st.dataframe(st.session_state.df.head())      
      
    # Select columns for latitude and longitude      
    columns = st.session_state.df.columns.tolist()      
    lat_col = st.selectbox("Select Longitude Column", columns, index=st.session_state.lat_col if st.session_state.lat_col in columns else 0)      
    lon_col = st.selectbox("Select Latitude Column", columns, index=st.session_state.lon_col if st.session_state.lon_col in columns else 0)      
      
    if lat_col and lon_col:      
        # Store selected columns in session state      
        st.session_state.lat_col = lat_col      
        st.session_state.lon_col = lon_col      
      
        # Convert the selected columns to a GeoDataFrame      
        try:      
            @st.cache_resource    
            def create_geodataframe(df, lat_col, lon_col):    
                gdf = transform_data_to_geodataframe(df, lat_col, lon_col)      
                gdf_2 = transform_data_to_geodataframe(df, lon_col, lat_col)   
                gdf = pd.concat([gdf, gdf_2])  
                gdf = clean_invalid_infinite_geometries(gdf)      
                gdf.to_crs(epsg=4326, inplace=True)      
                return gdf    
    
            gdf = create_geodataframe(st.session_state.df, lat_col, lon_col)    
    
            # Store the cleaned GeoDataFrame in session state      
            st.session_state.gdf = gdf      
            st.session_state.map_created = False  # Set to True to indicate map can be shown  
      
            # Provide a download button to download the file as an SHP file      
            if not st.session_state.map_created : 
                with tempfile.TemporaryDirectory() as temp_dir:      
                    # Write the GeoDataFrame to a shapefile in the temporary directory      
                    gdf.to_file(os.path.join(temp_dir, "output.shp"))      
            
                    # Create a zip file containing the shapefile      
                    zip_buffer = io.BytesIO()      
                    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:      
                        for file in os.listdir(temp_dir):      
                            zipf.write(os.path.join(temp_dir, file), file)      
            
                    # Seek to the beginning of the buffer      
                    zip_buffer.seek(0)      
            
                    # Provide a download button to download the file as a zip file      
                   
                st.session_state.map_created = True 
            st.download_button(      
                        label="Download SHP File",      
                        data=zip_buffer,      
                        file_name="output.zip",      
                        mime="application/zip"      
            )       
        
        except Exception as e:      
            st.error(f"An error occurred: {e}")      
  
# Function to create and cache the map  

def create_map(gdf):  
    if ('hpm' in gdf.columns) : 
        # Create a folium map object  
        m = gdf.explore(
            attr = 'google maps', 
            column = 'hpm', 
            scheme = 'quantiles', 
            k = 10
        )

    # Add the secondary GeoDataFrame with custom markers if provided
    # if other_data is not None:
    #    other_data.explore()
    else : 
        m = gdf.explore(
            attr = 'google maps', 
        )


    # Add alternative tile layers
    folium.TileLayer('Stamen Terrain',attr = 'google map', control=True).add_to(m)
    folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', control=True, attr='<a href=https://endless-sky.github.io/>Endless Sky</a>').add_to(m)
    folium.LayerControl().add_to(m)
    return m
  
# Show the interactive map if the flag is set    
if st.session_state.map_created and st.session_state.gdf is not None:    
    st.write("Map Visualization:")      
    # Create and display the map  
    m = create_map(st.session_state.gdf)  
    folium_static(m, width=700, height=500)      