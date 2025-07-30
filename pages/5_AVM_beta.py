import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium, folium_static
from sqlalchemy import create_engine
from folium.plugins import Draw
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from helper import visualize_poi_by_wkt
import json
import requests
import pyproj
from openai import OpenAI

# ——— CONFIGURATION ——— 


st.set_page_config(layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
engine = create_engine(
    f"postgresql://{st.secrets['database']['user']}:{st.secrets['database']['password']}@{st.secrets['database']['host']}:{st.secrets['database']['port']}/{st.secrets['database']['name']}"
)

# ——— CACHES FOR PERFORMANCE ———
@st.cache_resource(show_spinner=False)
def geocode(address: str):
    url = (
        "https://maps.googleapis.com/maps/api/geocode/json"
        f"?address={requests.utils.quote(address)}"
        f"&key={GOOGLE_API_KEY}"
    )
    r = requests.get(url, timeout=5).json()
    if r.get("results"):
        loc = r["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    return None, None

@st.cache_resource(show_spinner=False)
def fetch_admin(lon: float, lat: float) -> dict:
    r = requests.get(
        "https://apiavm.rhr.co.id/geo/get-administrative-area",
        params={"lon": lon, "lat": lat},
        verify=False,
        timeout=10
    )
    return r.json() if r.status_code == 200 else {}

@st.cache_resource(show_spinner=False)
def fetch_visuals(buffer_wkt: str):
    # visualize_poi_by_wkt now returns 6 items: (map_viz, bar_fig, land_price_kde, yearly_price_development, surrounding_environment, prop_df)
    return visualize_poi_by_wkt(buffer_wkt, engine)




# ——— SESSION STATE INITIALIZATION ———

st.session_state['authorized']=False
if "selected_geojson" not in st.session_state:
    st.session_state.selected_geojson = None
if "map_center" not in st.session_state:
    st.session_state.map_center = (-6.2, 106.8)
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": (
            "You are a seasoned property consultant in Indonesia. "
            "Based on the data provided, give clear, concise recommendations."
        )}
    ] 
if st.session_state['authorized']==False : 
    st.warning("This is a classified project, kindly contact inderaihsan@gmail.com for accessing") 
    password_ = st.text_input(type='password', label='input password') 
    if (password_ == st.secrets['ADMIN_PASSWORD']) : 
        st.session_state.authorized = True 
    else : 
        st.error("Unauthorized access..")
        st.session_state.authorized = False
        st.stop()
# pinned_context holds the permanent data_summary
if "pinned_context" not in st.session_state:
    st.session_state.pinned_context = None

# ——— APP UI ———
st.title("🏘️ Property Analysis Tool")
st.write("Search an address or draw a marker on the map, then chat below without re-running analysis.")

# SEARCH BAR
search = st.text_input("🔎 Search location", placeholder="e.g. Senayan, Jakarta")
if search:
    lat, lng = geocode(search)
    if lat and lng:
        st.session_state.map_center = (lat, lng)
    else:
        st.warning("Location not found. Try another search term.")

# MAP + DRAW
with st.container():
    m = folium.Map(location=st.session_state.map_center, zoom_start=13)
    if search:
        folium.Marker(
            st.session_state.map_center,
            popup=search,
            icon=folium.Icon(color="blue", icon="search")
        ).add_to(m)
    Draw(draw_options={"marker": True}).add_to(m)
    folium.LayerControl().add_to(m)
    click_data = st_folium(m, width=1350, height=500, use_container_width=True)

    if click_data and click_data.get("last_active_drawing"):
        st.session_state.selected_geojson = click_data["last_active_drawing"]["geometry"]
        pt = shape(st.session_state.selected_geojson)
        st.session_state.map_center = (pt.y, pt.x)
        st.session_state.analysis_done = False

# RUN SPATIAL ANALYSIS ONCE PER SELECTION
if st.session_state.selected_geojson and not st.session_state.analysis_done:
    shp = shape(st.session_state.selected_geojson)
    lat, lng = shp.y, shp.x

    # 1km buffer via UTM
    wgs84 = pyproj.CRS("EPSG:4326")
    utm   = pyproj.CRS("EPSG:32749")
    to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    to_wgs = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform
    utm_pt = shp_transform(to_utm, shp)
    buf = utm_pt.buffer(1000)
    buffer_wkt = shp_transform(to_wgs, buf).wkt

    # fetch remote data
    admin = fetch_admin(lng, lat)
    visuals = fetch_visuals(buffer_wkt)
    map_viz, bar_fig, land_price_kde, yearly_price_development, surrounding_environment, prop_df = visuals

    # store in session_state
    st.session_state.analysis = {
        "lat": lat, "lng": lng,
        "admin": admin,
        "map_viz": map_viz,
        "bar_fig": bar_fig,
        "land_price_kde": land_price_kde,
        "yearly_price_development": yearly_price_development,
        "surrounding_environment": surrounding_environment,
        "prop_df": prop_df
    }
    # build chat_summary and pinned_context once
    price_summary = prop_df["price"].describe().to_dict() if "price" in prop_df else {}
    summary = {
        "latitude": lat,
        "longitude": lng,
        "administrative": admin,
        "price_summary": price_summary,
        "properties": prop_df.to_dict(orient="records")
    }
    st.session_state.chat_summary = summary
    ctx = json.dumps(summary, indent=2)
    st.session_state.pinned_context = {"role":"system","content": f"Data for discussion:\n{ctx}"}

    st.session_state.analysis_done = True

# DISPLAY ANALYSIS
if st.session_state.analysis_done:
    a = st.session_state.analysis
    lat, lng = a["lat"], a["lng"]

    st.subheader("📍 Selected Point")
    c1, c2 = st.columns(2)
    c1.write(f"**Latitude:** {lat:.6f}")
    c2.write(f"**Longitude:** {lng:.6f}")

    with st.expander("📍 Administrative Location"):
        df_admin = pd.DataFrame([{ 
            "Provinsi": a["admin"].get("Provinsi",""),
            "Kota/Kabupaten": a["admin"].get("Kota/Kabupaten",""),
            "Kecamatan": a["admin"].get("Kecamatan",""),
            "Kelurahan/Desa": a["admin"].get("Kelurahan/Desa","")
        }])
        st.dataframe(df_admin, use_container_width=True)

    with st.expander("🏢 Amenity & Transport & Factors"):
        tabs = st.tabs(["🏢 Amenities","🚉 Transport","⚠️ Negative"])
        resp = a["admin"]
        # amenities
        amen = {
            "School": (resp.get("Nearest School"), resp.get("Distance to Nearest School (m)")),
            "Retail": (resp.get("Nearest Retail"), resp.get("Distance to Nearest Retail (m)")),
            "Hotel":  (resp.get("Nearest Hotel"),  resp.get("Distance to Nearest Hotel (m)")),
        }
        df_amen = pd.DataFrame([{"Amenity":k,"Name":v[0],"Dist(m)":v[1]} for k,v in amen.items()])
        tabs[0].dataframe(df_amen, use_container_width=True)
        # transport
        trans = {
            "Train Station": (resp.get("Nearest Train Station"), resp.get("Distance to Nearest Train Station (m)")),
            "Bus Stop":      (resp.get("Nearest Bus Stop"),    resp.get("Distance to Nearest Bus Stop (m)")),
        }
        df_trans = pd.DataFrame([{"Transport":k,"Name":v[0],"Dist(m)":v[1]} for k,v in trans.items()])
        tabs[1].dataframe(df_trans, use_container_width=True)
        # negative
        neg = {"Cemetery": (resp.get("Nearest Cemetery"), resp.get("Distance to Nearest Cemetery (m)"))}
        df_neg = pd.DataFrame([{"Factor":k,"Name":v[0],"Dist(m)":v[1]} for k,v in neg.items()])
        tabs[2].dataframe(df_neg, use_container_width=True)

    with st.expander("Detailed View for Selected Area"):
        mt, bt, pt, dt = st.tabs([
            "Map visualization","📍 Access (POI Count)",
            "📊 Price Group","💰 Demand Analysis"
        ])
        with mt:
            folium_static(a["map_viz"], width=1400, height=500)
        with bt:
            cols = st.columns(2)
            if a["bar_fig"]: cols[0].plotly_chart(a["bar_fig"], use_container_width=True)
            if a["surrounding_environment"]: cols[1].plotly_chart(a["surrounding_environment"], use_container_width=True)
        with pt:
            if a["land_price_kde"]: st.plotly_chart(a["land_price_kde"], use_container_width=True)
            else: st.info("Land price data not available.")
        with dt:
            if a["yearly_price_development"]: st.plotly_chart(a["yearly_price_development"], use_container_width=True)
            else: st.info("Yearly price trend data not available.")

    with st.expander("🏘️ Surrounding Properties"):
        st.dataframe(a["prop_df"], use_container_width=True)

    # RENDER PAST CHAT
    for msg in st.session_state.messages[1:]:
        st.chat_message(msg["role"]).write(msg["content"])

    # CHAT INPUT & STREAMING WITH FULL CONTEXT
    st.subheader("Chat with AI Agent")
    user_q = st.chat_input("💬 Ask about this location…")
    if user_q:
        # append user
        st.session_state.messages.append({"role":"user","content":user_q})
        st.chat_message("user").write(user_q)

        # build history: system, pinned_context, then all messages[1:]
        history = [st.session_state.messages[0]]
        if st.session_state.pinned_context:
            history.append(st.session_state.pinned_context)
        history += st.session_state.messages[1:]

        # assistant streaming
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_text = ""
            # reserve slot
            st.session_state.messages.append({"role":"assistant","content":""})
            for chunk in client.chat.completions.create(
                model="gpt-4.1-mini", messages=history, stream=True, temperature=0.7
            ):
                delta = chunk.choices[0].delta.content or ""
                full_text += delta
                placeholder.write(full_text)
            # save final
            st.session_state.messages[-1]["content"] = full_text
else:
    st.info("Draw a marker or search above, then analysis will appear.")
