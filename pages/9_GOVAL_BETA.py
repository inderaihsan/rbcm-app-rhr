import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import json

st.set_page_config(page_title="GOVAL Engine Predictor", layout="wide")

st.title("🏢 GOVAL Engine Land Value Predictor")

# Initialize session state for coordinates
if 'latitude' not in st.session_state:
    st.session_state.latitude = -6.211824861547755
if 'longitude' not in st.session_state:
    st.session_state.longitude = 106.82838818022269

# Sidebar for input parameters
st.sidebar.header("📋 Input Parameters")

coordinate_input = st.sidebar.text_input("Coordinates (lat, lon)", value=f"{st.session_state.latitude:.10f}, {st.session_state.longitude:.10f}")
manual_lat = float(coordinate_input.split(",")[0].strip())
manual_lon = float(coordinate_input.split(",")[1].strip())

land_area = st.sidebar.number_input("Land Area (m²)", min_value=1, value=285)
road_width = st.sidebar.number_input("Road Width (m)", min_value=1, value=6)
radius_buffer = st.sidebar.number_input("Radius Buffer (m)", min_value=1, value=150)
agga_land_use = st.sidebar.selectbox("AGGA Land Use", options=["green", "slum to normal", "normal premium", "premium", "industri", "komersial"], index=0) 
bentuk_tapak = st.sidebar.selectbox("Bentuk Tapak", options=["Persegi", "Lainnya"], index=0) 
tahun = st.sidebar.selectbox("Tahun", options=range(2010, 2026), index=4)

api_url = st.sidebar.text_input("API URL", value="http://192.168.90.115:8000/predict-using-goval-engine")

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🚀 Get Prediction", type="primary", use_container_width=True)

# Main content area
if predict_button:
    # Construct API URL with parameters
    params = {
        "longitude": manual_lon,
        "latitude": manual_lat,
        "land_area": land_area,
        "road_width": road_width,
        "radius_buffer_meters": radius_buffer, 
        "agga_landuse": agga_land_use, 
        "bentuk_tapak": bentuk_tapak,
        "tahun": tahun
    }
    
    full_url = f"{api_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
    
    with st.spinner("Fetching prediction..."):
        try:
            response = requests.get(full_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Display main prediction results prominently
            st.success("✅ Prediction Successful!")
            
            # Key metrics in 3 rows of 2 columns
            st.markdown("### 🎯 Model Predictions")
            
            row1 = st.columns(2)
            with row1[0]:
                st.metric("RERF Prediction", f"Rp {data.get('rerf_prediction', 0):,.0f}")
            with row1[1]:
                st.metric("RF Prediction", f"Rp {data.get('rf_pred', 0):,.0f}")
            
            row2 = st.columns(2)
            with row2[0]:
                st.metric("GBDT Prediction", f"Rp {data.get('gbdt_prediction', 0):,.0f}")
            with row2[1]:
                st.metric("LASSO Prediction", f"Rp {data.get('lasso_prediction', 0):,.0f}")
            
            row3 = st.columns(2)
            with row3[0]:
                st.metric("RIDGE Prediction", f"Rp {data.get('ridge_prediction', 0):,.0f}")
            with row3[1]:
                st.metric("SVR Prediction", f"Rp {data.get('svr_prediction', 0):,.0f}")

            row4 = st.columns(1)
            with row3[0]:
                st.metric("XG-Boost Prediction", f"Rp {data.get('xgboost_prediction', 0):,.0f}") 

            soft_mean_prediction = data.get('soft_mean_prediction', 0)  
            soft_median_prediction = data.get('soft_median_prediction', 0) 
            bottom_whisker_mean = data.get('bottom_whisker_mean', 0)
            bottom_whisker_median = data.get('bottom_whisker_mad', 0)
            top_whisker_mean = data.get('top_whisker_mean', 0)  
            top_whisker_median = data.get('top_whisker_mad', 0)  
            formatted_mean_value = f"Rp {soft_mean_prediction:,.0f}"
            formatted_median_value = f"Rp {soft_median_prediction:,.0f}"
            formatted_range_value_mean = f"Rp {bottom_whisker_mean:,.0f} - Rp {top_whisker_mean:,.0f}"
            formatted_range_value_median = f"Rp {bottom_whisker_median:,.0f} - Rp {top_whisker_median:,.0f}"

            with st.expander("📊 Detailed Model Predictions"):
                with st.container():
                    st.markdown(
                        f"""
                        <div style='font-size:22px; color:green; font-weight:bold; font-style:italic;'>
                            Model Average Prediction: {formatted_mean_value}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div style='font-size:22px; color:green; font-weight:bold; font-style:italic;'>
                            Model Median Prediction: { formatted_median_value }

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div style='font-size:22px; color:green; font-weight:bold; font-style:italic;'>
                            Range Mean <->: { formatted_range_value_mean }

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div style='font-size:22px; color:green; font-weight:bold; font-style:italic;'>
                            Range Median <->: { formatted_range_value_median }

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    

                st.markdown("---")
                
            # Secondary information in compact format
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.markdown("### 📍 Location")
                st.write(f"**Province:** {data.get('wadmpr', 'N/A')}")
                st.write(f"**City:** {data.get('wadmkk', 'N/A')}")
                st.write(f"**District:** {data.get('wadmkc', 'N/A')}")
                st.write(f"**Sub-district:** {data.get('wadmkd', 'N/A')}")
            
            with info_col2:
                # Nearby places
                if data.get('name_result') and data.get('distance_result'):
                    st.markdown("### 🏢 Nearby Places")
                    nearby_count = 0
                    for name, dist in zip(data['name_result'], data['distance_result']):
                        if name and nearby_count < 5:  # Limit to 5 places
                            st.write(f"• {name} ({dist:.0f}m)")
                            nearby_count += 1
            
            # Show full JSON in expander
            with st.expander("📄 View Full API Response"):
                st.json(data)
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error fetching prediction: {str(e)}")
        except json.JSONDecodeError:
            st.error("❌ Error parsing API response")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

else:
    # Show map when no prediction is running
    st.markdown("### 📍 Select Location on Map")
    st.info(f"**Current Coordinates:** {manual_lat:.6f}, {manual_lon:.6f}")
    
    # Create map centered on current coordinates
    m = folium.Map(
        location=[manual_lat, manual_lon],
        zoom_start=15,
        tiles="OpenStreetMap"
    )
    
    # Add marker at current location
    folium.Marker(
        [manual_lat, manual_lon],
        popup="Selected Location",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    # Display map and capture clicks
    map_data = st_folium(m, width=None, height=500, key="map")
    
    # Update coordinates if map was clicked
    if map_data and map_data.get("last_clicked"):
        st.session_state.latitude = map_data["last_clicked"]["lat"]
        st.session_state.longitude = map_data["last_clicked"]["lng"]
        st.rerun()
    
    st.markdown("💡 **Tip:** Click on the map to select a location, adjust parameters in the sidebar, then click 'Get Prediction'") 