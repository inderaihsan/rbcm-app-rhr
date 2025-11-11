import streamlit as st
from openai import AsyncOpenAI
import googlemaps
from dotenv import load_dotenv
import os
import asyncio
import pandas as pd
from datetime import datetime
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
import os 
import logging
from typing import Dict, List, Any
import asyncio
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import create_engine, text
import pandas as pd
import math
import googlemaps
import json
import time




# Import your AgenticView class
# from your_module import AgenticView

db_user = st.secrets["database"]["user"]
db_password = st.secrets["database"]["password"]
db_host = st.secrets["database"]["host"] 
db_port = st.secrets["database"]["port"] 
db_name = st.secrets["database"]["name"]

class AgenticView :
  
  def __init__(self, google_client, gpt_client_async):
        """
        """
        self.google_client = google_client
        self.gpt_client_async = gpt_client_async
        self.engine  = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

  def validate_locations(self, parameter : dict)->dict :
      try:
          # First attempt: parse lon/lat directly
          lon = float(parameter["longitude"])
          lat = float(parameter["latitude"])

      except Exception as e1:
          try:
              # Second attempt: use geocoding from address
              alamat = parameter["alamat"]
              geocode_result = self.google_client.geocode(alamat)

              lat = float(geocode_result[0]['geometry']['location']['lat'])
              lon = float(geocode_result[0]['geometry']['location']['lng'])

          except Exception as e2:
              # Final fallback if both attempts fail
              print("Failed to get coordinates.")
              print("First error:", e1)
              print("Second error:", e2)
              lon, lat = None, None
              return None
      parameter["longitude"] = lon
      parameter["latitude"] = lat
      return parameter


  async def create_gdf(self, parameter : dict)-> gpd.GeoDataFrame:
      """
      """
      jenis_objek = parameter.get("jenis_objek", None)
      pemberi_tugas = parameter.get("pemberi_tugas", None)
      nomor_kontrak = parameter.get("nomor_kontrak", None)
      luas_tanah = parameter.get("luas_tanah", None)
      luas_bangunan = parameter.get("luas_bangunan", None)
      tahun = parameter.get("tahun", None)
      tujuan_penilaian = parameter.get("tujuan_penilaian", None)
      jenis_transaksi = parameter.get("jenis_transaksi", None)
      alamat_lokasi = parameter.get("alamat_lokasi", None)
      lon = parameter.get("longitude", None)
      lat = parameter.get("latitude", None)
      row = {
          "longitude": lon,
          "latitude": lat,
          "jenis_objek": jenis_objek,
          "pemberi_tugas": pemberi_tugas,
          "nomor_kontrak": nomor_kontrak,
          'tahun' : tahun,
          "luas_tanah": luas_tanah,
          "luas_bangunan": luas_bangunan,
          "tujuan_penilaian": tujuan_penilaian,
          "jenis_transaksi": jenis_transaksi,
          "alamat_lokasi": alamat_lokasi,
          "geometry": Point(lon, lat),

      }

      # GeoDataFrame dari list of dicts; tetapkan CRS WGS84
      gdf = gpd.GeoDataFrame([row], crs="EPSG:4326")
      return gdf

  async def find_neighbour(self, distance_m, lon, lat) :
    query = f"""WITH q AS (
                                    SELECT ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)::geography AS pt
                                )
                                SELECT
                                    t.pemberi_tugas,
                                    t.jenis_objek_text,
                                    t. cabang_text,
                                    t.divisi,
                                    t.tahun_kontrak,
                                    t.alamat_lokasi,
                                    t.keterangan,
                                    t.geometry,
                                    ST_Distance(t.geog, q.pt) AS distance_m
                                FROM objek_penilaian t, q
                                WHERE ST_DWithin(t.geog, q.pt, {distance_m})  -- radius 10 km
                                  AND t.longitude <> 0
                                  AND ST_Distance(t.geog, q.pt) > 0    -- exclude titik pusat
                                ORDER BY distance_m
                                LIMIT 20;   """

    df = pd.read_sql(query, self.engine)
    return df



  async def get_llm_response_of_object(self, df : pd.DataFrame, gdf_from_params : gpd.GeoDataFrame)  -> str :
    fetched_context = df.to_json(orient = 'records') if df is not None else None
    prospected_jobs = gdf_from_params.to_dict(orient = 'records')
    response = await self.gpt_client_async.responses.create(
      model="gpt-4.1-mini",
      max_output_tokens = 2000,
      input=f'''Anda adalah asisten berbahasa Indonesia yang bertugas membantu perusahaan penilaian untuk:
      1. Mencegah terjadinya konflik kepentingan,
      2. Menghindari duplikasi pekerjaan,
      3. Tidak melakukan penilaian ulang (revaluasi) pada objek yang sama.

      Berikut adalah data prospek penugasan baru (data prospect) :
      {prospected_jobs}

      Dan berikut adalah data penugasan yang sudah pernah dilakukan perusahaan (data existing):
      {fetched_context}

      Tugas Anda:
      - Bandingkan setiap objek pada `data prospect` dengan daftar di `data existing`.
      - Identifikasi jika ada objek yang berpotensi sama, mirip, atau menimbulkan konflik kepentingan.
      - Jelaskan alasan kemiripan (misalnya alamat mirip, koordinat berdekatan, nama pemberi tugas sama, tujuan penilaian sama, atau jenis transaksinya sama. dsb). 
      - Jika ada, Tampilkan data data yang kemungkinan mirip tersebut dalam bentuk tabel (nama pemberi tugas, alamat, jarak, tujuan penilaian, jenis transaksi, tahun kontrak)

      '''
      )

    return response.output_text

  async def get_llm_response_of_task_giver(self, task_giver : str) -> str :
    response = await self.gpt_client_async.responses.create(
      model="gpt-4.1-mini",
      max_output_tokens = 500,
      tools = [{"type" : "web_search"}],
      input=f'''Anda adalah asisten berbahasa Indonesia yang bertugas membantu perusahaan penilaian untuk:
      Mencegah terjadinya Penerimaan pekerjaan yang mendukung korupsi,kolusi dan benturan kepentingan, atau isu negatif lain yang membahayakan entitas kita.
      Berikut adalah pemberi tugas {task_giver} yang memberikan pekerjaan lakukan pencarian web mengenai hal negatif yang mungkin menjadi masalah. jika tidak terdapat indikasi
      korupsi, cukup katakan bahwa tidak ada informasi relevan mengenai korupsi atau tindakan negatif dari pemberi tugas
      '''
    )

    return response.output_text

  async def get_result(self, parameter):
    time_start = time.time()
    parameter = self.validate_locations(parameter)
    print('total running time of validate locations : ', time.time() - time_start)
    time_start = time.time()
    gdf_from_params = await self.create_gdf(parameter)
    print('total running time of gdf params : ', time.time() - time_start)
    time_start = time.time()
    neighbour = await self.find_neighbour(
        10000,
        float(parameter["longitude"]),
        float(parameter["latitude"]),
    )
    print('total running time of find neighbour : ', time.time() - time_start)
    time_start = time.time()

    tasks = [self.get_llm_response_of_task_giver(parameter["pemberi_tugas"]),
            self.get_llm_response_of_object(neighbour, gdf_from_params)]
    print('total running time of llm : ', time.time() - time_start)
    results = await asyncio.gather(*tasks)
    return {'task_giver_analysis' : results[0], 'object_analysis' : results[1]}




# Page configuration
st.set_page_config(
    page_title="Agentic View (BETA)",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Initialize clients
@st.cache_resource
def init_clients():
    """Initialize Google Maps and OpenAI clients"""

    google_api_key = st.secrets["GOOGLE_API_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    google_client = googlemaps.Client(key=google_api_key)
    gpt_client_async = AsyncOpenAI(api_key=openai_api_key)
    
    # Uncomment when AgenticView is available
    agentic_view = AgenticView(google_client, gpt_client_async)
    return agentic_view
    # except Exception as e:
    #     st.error(f"Error initializing clients: {str(e)}")
    #     return None, None

# Header
st.markdown('<div class="main-header">🏢 Agentic View 1.0 - BETA</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Informasi Sistem")
    st.markdown("""
    Sistem ini membantu mencegah:
    - ✅ Konflik kepentingan
    - ✅ Duplikasi pekerjaan
    - ✅ Revaluasi objek yang sama
    - ✅ Pekerjaan berisiko tinggi
    """)
    
    st.divider()
    
    st.header("🔧 Pengaturan")
    search_radius = st.slider(
        "Radius Pencarian (meter)",
        min_value=1000,
        max_value=20000,
        value=10000,
        step=1000,
        help="Jarak maksimum untuk mencari objek penilaian terdekat"
    )
    
    st.divider()
    
    if st.session_state.analysis_history:
        st.header("📊 Riwayat Analisis")
        st.metric("Total Analisis", len(st.session_state.analysis_history))

# Main content
tab1, tab2, tab3 = st.tabs(["📝 Input Data", "📊 Hasil Analisis", "📜 Riwayat"])

with tab1:
    st.header("Input Data Penugasan Baru")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informasi Umum")
        jenis_objek = st.selectbox(
            "Jenis Objek *",
            [
                "Kios",
                "Bisnis Unit",
                "Kapal",
                "Rumah Sakit",
                "Unit Mesin",
                "Rumah Tinggal",
                "Pembangkit Listrik",
                "Perkebunan Kelapa Sawit",
                "Ruko",
                "Perkebunan Hutan Tanaman Industri",
                "Alat Berat",
                "Stok Barang",
                "Pabrik",
                "Lainnya",
                "Tanah dan Bangunan Sederhana",
                "Pabrik Kelapa Sawit",
                "Tanah Kosong",
                "Pembangkit",
                "Tangki Timbun (Bulking Station)",
                "Gedung Kantor",
                "Serviced Apartemen",
                "Aset Tak Berwujud",
                "Tower",
                "SPBU",
                "Tanah dan Bangunan Gudang atau Pabrik",
                "Perkebunan Nanas & Komoditi Lain",
                "Mesin dan Peralatan",
                "Biogas",
                "Saham",
                "Villa",
                "Perkebunan Hortikultur",
                "Pendapat Kewajaran",
                "Hotel",
                "Soho",
                "Entitas",
                "Unit Kendaraan",
                "Transaksi",
                "Pipeline",
                "Ruang Kantor",
                "Kondominium",
                "Mall",
                "Perkebunan Kelapa Sawit Plasma",
                "Bangunan Saja"
            ],
            help="Pilih jenis objek yang akan dinilai"
        )
        
        pemberi_tugas = st.text_input(
            "Pemberi Tugas *",
            placeholder="Contoh: PT Bank ABC",
            help="Nama institusi/perusahaan yang memberikan tugas"
        )
        
        # nomor_kontrak = st.text_input(
        #     "Nomor Kontrak",
        #     placeholder="Contoh: 001/PEN/2024"
        # )
        
        tahun = st.number_input(
            "Tahun Kontrak *",
            min_value=2000,
            max_value=datetime.now().year,
            value=datetime.now().year
        )
        
        tujuan_penilaian = st.selectbox(
            "Tujuan Penilaian *",
            [
                "Pelaporan Keuangan",
                "Audit Support / Review",
                "Asuransi",
                "Investasi / Pendanaan",
                "Akuisisi / Penggabungan Usaha / Divestasi",
                "Jual Beli / Sewa Menyewa",
                "Penghapusan Aset / Hibah / Lelang",
                "IPO / Keterbukaan Informasi Publik",
                "Penjaminan Utang",
                "Pengadaan Tanah / Kompensasi",
                "Kajian Nilai / Studi Kelayakan",
                "Rencana Kerjasama / Internal Manajemen",
                "Pemanfaatan Ruang / Kesesuaian Tata Ruang"
            ]
        )
        
        jenis_transaksi = st.selectbox(
            "Jenis Transaksi",
            [
                "Monitoring" , 
                "Advisory" , 
                "Konsultansi" , 
                "Penilaian Saham" , 
                "Penilaian Aset" , 
                "Others"
            ]
        )
    
    with col2:
        st.subheader("Informasi Lokasi")
        alamat_lokasi = st.text_area(
            "Alamat Lokasi *",
            placeholder="Masukkan alamat lengkap objek penilaian",
            height=100
        )
        
        coord_col1, coord_col2 = st.columns(2)
        with coord_col1:
            longitude = st.text_input(
                "Longitude",
                placeholder="106.8456",
                help="Opsional jika alamat sudah diisi"
            )
        with coord_col2:
            latitude = st.text_input(
                "Latitude",
                placeholder="-6.2088",
                help="Opsional jika alamat sudah diisi"
            )
        
        st.info("💡 Koordinat akan otomatis dicari dari alamat jika tidak diisi")
        
        st.subheader("Informasi Properti")
        
        luas_col1, luas_col2 = st.columns(2)
        with luas_col1:
            luas_tanah = st.number_input(
                "Luas Tanah (m²)",
                min_value=0.0,
                value=0.0,
                step=10.0
            )
        with luas_col2:
            luas_bangunan = st.number_input(
                "Luas Bangunan (m²)",
                min_value=0.0,
                value=0.0,
                step=10.0
            )
    
    st.divider()
    
    # Submit button
    col_button1, col_button2, col_button3 = st.columns([1, 1, 2])
    with col_button1:
        submit_button = st.button("🔍 Analisis Sekarang", type="primary", use_container_width=True)
    with col_button2:
        clear_button = st.button("🗑️ Bersihkan Form", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if submit_button:
        # Validation
        if not pemberi_tugas or not alamat_lokasi:
            st.error("⚠️ Mohon isi semua field yang wajib (*)")
        else:
            # Prepare parameter
            parameter = {
                "jenis_objek": jenis_objek,
                "pemberi_tugas": pemberi_tugas,
                "nomor_kontrak": None,
                "luas_tanah": luas_tanah,
                "luas_bangunan": luas_bangunan,
                "tahun": tahun,
                "tujuan_penilaian": tujuan_penilaian,
                "jenis_transaksi": jenis_transaksi,
                "alamat_lokasi": alamat_lokasi,
                "alamat": alamat_lokasi,
                "longitude": longitude if longitude else None,
                "latitude": latitude if latitude else None
            }
            
            # Show loading
            with st.spinner("🔄 Sedang menganalisis..."):
                try:
                    # Initialize clients
                
                        agentic_view_processor = init_clients()
                        results = asyncio.run(agentic_view_processor.get_result(parameter))
                        
                        # # Mock results for demonstration
                        # results = {
                        #     'task_giver_analysis': 'Analisis pemberi tugas menunjukkan tidak ada indikasi negatif.',
                        #     'object_analysis': 'Ditemukan 3 objek serupa dalam radius 10km. Perlu peninjauan lebih lanjut.'
                        # }
                        
                        st.session_state.analysis_results = results
                        st.session_state.analysis_history.append({
                            'timestamp': datetime.now(),
                            'pemberi_tugas': pemberi_tugas,
                            'alamat': alamat_lokasi,
                            'results': results
                        })
                        
                        st.success("✅ Analisis selesai! Lihat hasil di tab 'Hasil Analisis'")
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {str(e)}")

with tab2:
    st.header("Hasil Analisis")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Task Giver Analysis
        # st.subheader("🔍 Analisis Pemberi Tugas")
        # st.markdown(f'<div class="info-box">{results["task_giver_analysis"]}</div>', unsafe_allow_html=True)
        
        # st.divider()
        
        # Object Analysis
        st.subheader("🏘️ Analisis Objek Penilaian")
        st.markdown(f'<div class="info-box">{results["object_analysis"]}</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Download results
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download Hasil (JSON)", use_container_width=True):
                st.download_button(
                    label="Unduh JSON",
                    data=str(results),
                    file_name=f"analisis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        with col2:
            if st.button("📄 Download Hasil (TXT)", use_container_width=True):
                report = f"""LAPORAN ANALISIS KONFLIK PENILAIAN
Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ANALISIS PEMBERI TUGAS:
{results['task_giver_analysis']}

ANALISIS OBJEK PENILAIAN:
{results['object_analysis']}
"""
                st.download_button(
                    label="Unduh TXT",
                    data=report,
                    file_name=f"analisis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    else:
        st.info("📝 Belum ada hasil analisis. Silakan input data di tab 'Input Data' terlebih dahulu.")

with tab3:
    st.header("Riwayat Analisis")
    
    if st.session_state.analysis_history:
        for idx, item in enumerate(reversed(st.session_state.analysis_history)):
            with st.expander(f"📌 {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {item['pemberi_tugas']}"):
                st.write(f"**Alamat:** {item['alamat']}")
                st.write("**Hasil Analisis Pemberi Tugas:**")
                st.info(item['results']['task_giver_analysis'])
                st.write("**Hasil Analisis Objek:**")
                st.info(item['results']['object_analysis'])
        
        if st.button("🗑️ Hapus Semua Riwayat"):
            st.session_state.analysis_history = []
            st.rerun()
    else:
        st.info("📝 Belum ada riwayat analisis.")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>Agentic View (BETA) v1.0 | 
        Pastikan semua API keys telah dikonfigurasi dengan benar</small>
    </div>
""", unsafe_allow_html=True)