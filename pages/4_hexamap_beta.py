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

# --- Database Engine ---
engine = create_engine(st.secrets["POSTGRES_ENGINE"])

# --- Load Cached Grid ---
@st.cache_data
def load_1000_grid():
    gdf = gpd.read_postgis("SELECT * FROM public.jakarta_1000_grid;", engine, geom_col='geometry')
    gdf = gdf.to_crs(epsg=4326)
    return gdf

gdf_1000 = load_1000_grid()

# --- UI Tabs ---
tab_draw, tab_result = st.tabs(["🖌️ Draw Area", "📊 Results"])

with tab_draw:
    st.title("Hexamap Beta")
    st.write("Use the Draw tool to select an area.")

    m = folium.Map(location=[-6.2, 106.8], zoom_start=11)
    Draw(export=True).add_to(m)

    def style_function(feature):
        return {'fillOpacity': 0.2, 'weight': 1, 'color': 'black'}

    grid_layer = folium.GeoJson(
        gdf_1000,
        name="Jakarta 1000m Grid",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=["grid_id"]),
    )
    grid_layer.add_to(m)
    folium.LayerControl().add_to(m)

    click_data = st_folium(m, width=700, height=500)

    if click_data and 'last_active_drawing' in click_data and click_data['last_active_drawing']:
        geom_geojson = click_data['last_active_drawing']['geometry']
        shapely_geom = shape(geom_geojson)
        selected_geom_wkt = shapely_geom.wkt
        st.session_state["selected_geom_wkt"] = selected_geom_wkt
        st.success("Area selected! Switch to the '📊 Results' tab to see the details.")
    
    if st.button("🔄 Clear Selection"):
        st.session_state.pop("selected_geom_wkt", None)
        st.experimental_rerun()

with tab_result:
    if "selected_geom_wkt" in st.session_state:
        st.subheader("Detailed View for Selected Area")
        map_viz, bar_fig, land_price_kde, building_price_kde, yearly_price_development, surrounding_environment = visualize_poi_by_wkt(
            st.session_state["selected_geom_wkt"], engine
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
        st.info("No area selected yet. Go to the '🖌️ Draw Area' tab and draw a shape.")
