import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium, folium_static
from sqlalchemy import create_engine
from folium.plugins import Draw
from shapely.geometry import shape
from helper import visualize_poi_by_wkt
from shapely import wkt
import  json
import requests
from shapely.ops import transform 
import plotly.express as px
import pyproj

# --- Database Engine ---
engine = create_engine(st.secrets["POSTGRES_ENGINE"])

# --- Load Cached Grid ---

# --- UI Tabs ---

st.title("Point Analysis Beta")
st.write("Click on map to view. analysis")

m = folium.Map(location=[-6.2, 106.8], zoom_start=11)
draw = Draw(
    export=True,
    draw_options={
        "polyline": False,
        "polygon": False,
        "circle": False,
        "rectangle": False,
        "circlemarker": False,
        "marker": True  # Only allow points
    },
    # edit_options={"edit": False}  # Optional: disable editing after placing
)
draw.add_to(m)
folium.LayerControl().add_to(m)

click_data = st_folium(m, width=1350, height=700)


if click_data and 'last_active_drawing' in click_data and click_data['last_active_drawing']:
    geom_geojson = click_data['last_active_drawing']['geometry']
    shapely_geom = shape(geom_geojson)
    selected_geom_wkt = shapely_geom.wkt
    clicked_point = wkt.loads(selected_geom_wkt) 
        # Define CRS
    wgs84 = pyproj.CRS("EPSG:4326")
    utm = pyproj.CRS("EPSG:32749")  # UTM zone 49S (correct for Jakarta/Bogor area)

    # Create transformers
    to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    to_wgs84 = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform

    # Transform point to UTM
    utm_point = transform(to_utm, clicked_point)

    # Buffer in meters
    buffer_utm = utm_point.buffer(1000)

    # Transform buffer back to WGS84
    buffer_wgs84 = transform(to_wgs84, buffer_utm)

    # Get WKT of buffer in WGS84
    buffer_1k = buffer_wgs84.wkt
    import streamlit as st
    import pandas as pd

    st.title("Property Input Form")

    # Input fields
    latitude = st.number_input("Latitude", format="%.6f", value = clicked_point.y)
    longitude = st.number_input("Longitude", format="%.6f", value = clicked_point.x)
    lebar_jalan_di_depan = st.number_input("Lebar Jalan di Depan (m)", min_value=0)
    luas_tanah = st.number_input("Luas Tanah (m²)", min_value=0)

    sertifikat = st.selectbox("Jenis Sertifikat", options=[
        (1, "SHM (Sertifikat Hak Milik)"),
        (2, "SHGB (Sertifikat Hak Guna Bangunan)"),
        (3, "Lainnya")
    ], format_func=lambda x: x[1])[0] 
    kondisi_wilayah_sekitar = st.selectbox("Kondisi Wilayah Sekitar", options = [
        (1, "Komersial"), 
        (0, "Perumahan")
    ], format_func=lambda x: x[1])[0]

    # Preprocessing
    is_SHM = 1 if sertifikat == 1 else 0
    SHM = 1 if sertifikat == 1 else 0
    is_SHGB = 1 if sertifikat == 2 else 0 
    kondisi_wilayah_sekitar_perumahan = 1 if kondisi_wilayah_sekitar==1 else 0

    # Combine all into a dict (can be used for model input)
    request_data = {
        "latitude": latitude,
        "longitude": longitude,
        "lebar_jalan_di_depan": lebar_jalan_di_depan,
        "luas_tanah": luas_tanah,
        "sertifikat": sertifikat,
        "is_SHM": is_SHM,
        "SHM": SHM,
        "is_SHGB": is_SHGB, 
        "kondisi_wilayah_sekitar_perumahan" : kondisi_wilayah_sekitar_perumahan, 
        'marking' : kondisi_wilayah_sekitar_perumahan, 
        "is_hook" : kondisi_wilayah_sekitar_perumahan, 
        "tapak_beraturan" : 1, 
        "jenis_objek_feat" : 1
    } 

    button_predict = st.button("Predict value now!") 
    if not (button_predict and latitude and longitude and lebar_jalan_di_depan and luas_tanah):
        st.error("Lebar Jalan di Depan, Luas Tanah, Latitude, and Longitude are required fields.") 
    else:
        st.success("Sending request to the API...")
        
        req = requests.post(st.secrets['PREDICT_API_URL'], json=request_data)
        
        if req.status_code == 200:
            response = req.json()  # Correct way to parse JSON from the response
            st.success("Prediction successful!")
            # st.write(response) 
            with st.expander(label = 'request detail') : 
                st.write(response) 
            model_preds = response["prediction"]["model_prediction"]
            df = pd.DataFrame([
                {
                    "Model": p["model_name"],
                    "Prediction (Rp)": round(p["prediction"][0])
                }
                for p in model_preds
            ])

            # Display table
            st.title("Model Prediction Results")
            st.dataframe(df)

            # Show range and summary stats
            st.markdown(f"**Predicted Price Range:** {response['prediction']['range']}")
            st.markdown(f"**Median:** Rp {round(response['prediction']['median']):,}")
            st.markdown(f"**Standard Deviation:** Rp {round(response['prediction']['std']):,}")  
            # st.secrets['admin_api_url']
            macro_analysis = requests.get(st.secrets["ADMIN_API_URL"], params = {"lon" : longitude, "lat" : latitude}) 
            
 
            
            if (macro_analysis.status_code ==200) : 
                st.title("Spatial Analysis Results : ")
                macro_analysis_response = macro_analysis.json() 
                # Extract POI name and distance pairs
                poi_data = []
                                
                # --- Group 1: Administrative Location ---
                with st.expander("📍 Administrative Location"):
                    st.write(pd.DataFrame({
                        "Provinsi": macro_analysis_response["Provinsi"],
                        "Kota/Kabupaten": macro_analysis_response["Kota/Kabupaten"],
                        "Kecamatan": macro_analysis_response["Kecamatan"],
                        "Kelurahan/Desa": macro_analysis_response["Kelurahan/Desa"],
                    }, index=[0])) 
                
             
                # --- Group 2: Amenity Within ---
                with st.expander("🏢 Amenity Within"):
                    amenity_data = {
                        "School": (macro_analysis_response["Nearest School"], macro_analysis_response["Distance to Nearest School (m)"]),
                        "Retail": (macro_analysis_response["Nearest Retail"], macro_analysis_response["Distance to Nearest Retail (m)"]),
                        "Hotel": (macro_analysis_response["Nearest Hotel"], macro_analysis_response["Distance to Nearest Hotel (m)"]),
                        "Restaurant": (macro_analysis_response["Nearest Restaurant"], macro_analysis_response["Distance to Nearest Restaurant (m)"]),
                        "Cafe/Resto": (macro_analysis_response["Nearest Cafe/Resto"], macro_analysis_response["Distance to Nearest Cafe/Resto (m)"]),
                        "Mall": (macro_analysis_response["Nearest Mall"], macro_analysis_response["Distance to Nearest Mall (m)"]),
                        "Government Institution": (macro_analysis_response["Nearest Government Institution"], macro_analysis_response["Distance to Nearest Government (m)"]),
                        "Convenience Store": (macro_analysis_response["Nearest Retail"], macro_analysis_response["Distance to Nearest Retail (m)"]),  # assuming same as Retail
                    }
                    df_amenities = pd.DataFrame([
                        {"Amenity": k, "Name": v[0], "Distance (m)": v[1]}
                        for k, v in amenity_data.items()
                    ])
                    st.dataframe(df_amenities)

                # --- Group 3: Access to Public Transportation ---
                with st.expander("🚉 Access to Public Transportation"):
                    transport_data = {
                        "Train Station": (macro_analysis_response["Nearest Train Station"], macro_analysis_response["Distance to Nearest Train Station (m)"]),
                        "Airport": (macro_analysis_response["Nearest Airport"], macro_analysis_response["Distance to Nearest Airport (m)"]),
                        "Bus Stop": (macro_analysis_response["Nearest Bus Stop"], macro_analysis_response["Distance to Nearest Bus Stop (m)"]),
                    }
                    df_transport = pd.DataFrame([
                        {"Transportation": k, "Name": v[0], "Distance (m)": v[1]}
                        for k, v in transport_data.items()
                    ])
                    st.dataframe(df_transport)

                # --- Group 4: Negative Factor ---
                with st.expander("⚠️ Negative Factor"):
                    st.write({
                        "Nearest Cemetery": macro_analysis_response["Nearest Cemetery"],
                        "Distance to Nearest Cemetery (m)": macro_analysis_response["Distance to Nearest Cemetery (m)"]
                    })

            st.subheader("Detailed View for Selected Area")
            map_viz, bar_fig, land_price_kde, building_price_kde, yearly_price_development, surrounding_environment = visualize_poi_by_wkt(
                buffer_1k, engine
            )

            if map_viz:
                folium_static(map_viz, width=1400, height=500)
            else:
                st.warning("No 1000m grid intersects with your selection.")

            # Access group: POI Count & Surrounding Environment
            with st.expander("Access (POI and Surrounding Environment)", expanded=False):
                cols = st.columns(2)
                with cols[0]:
                    if bar_fig:
                        st.plotly_chart(bar_fig, use_container_width=True)
                    else:
                        st.info("POI data is not available.")
                with cols[1]:
                    if surrounding_environment:
                        st.plotly_chart(surrounding_environment, use_container_width=True)
                    else:
                        st.info("Surrounding environment data is not available.")

            # Price group: Land Price KDE & Building Price KDE
            with st.expander("Price (Property and Land)", expanded=False):
                cols = st.columns(2)
                with cols[0]:
                    if land_price_kde:
                        st.plotly_chart(land_price_kde, use_container_width=True)
                    else:
                        st.info("Land price data is not available.")
                with cols[1]:
                    if building_price_kde:
                        st.plotly_chart(building_price_kde, use_container_width=True)
                    else:
                        st.info("Building price data is not available.")

            # Demand group: Yearly Median Price Development
            with st.expander("Demand (Yearly Price Trend)", expanded=False):
                if yearly_price_development:
                    st.plotly_chart(yearly_price_development, use_container_width=True)
                else:
                    st.info("Yearly price trend data is not available.")
        else:
            st.error(f"Prediction failed with status code {req.status_code}")
            st.write(req.text) 





    st.subheader("Preprocessed Output")
    

