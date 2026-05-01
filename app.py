import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process, fuzz

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="RAD PS - DSS System", page_icon="🎮", layout="wide")
st.title("🎮 Sistem Pendukung Keputusan RAD Playstation")
st.markdown("Sistem Rekomendasi Penambahan dan Manajemen Aset Game Berbasis *Item-Based Collaborative Filtering*")

# ==========================================
# 2. CACHE DATA & MODEL (Agar website tidak lemot)
# ==========================================
@st.cache_data
def load_data_and_model():
    df = pd.read_excel("Dataset_Final_DSS_Rental.xlsx")
    
    # Preprocessing
    df_model = df.copy()
    if 'Local_Multiplayer' in df_model.columns:
        # Menangani berbagai format input agar tidak error
        df_model['Local_Multiplayer'] = df_model['Local_Multiplayer'].replace({'Yes': 1, 'No': 0, '1': 1, '0': 0, 'ya': 1, 'tidak': 0})
        df_model['Local_Multiplayer'] = pd.to_numeric(df_model['Local_Multiplayer'], errors='coerce').fillna(0)
        
    if 'Genre' in df_model.columns:
        genres_split = df_model['Genre'].astype(str).str.get_dummies(sep=', ')
        df_model = pd.concat([df_model, genres_split], axis=1)
        
    fitur_numerik = ['Rating_Global', 'Waktu_Main_Jam', 'Size_GB', 'Local_Multiplayer']
    fitur_lengkap = fitur_numerik + list(genres_split.columns)
    
    matriks_mentah = df_model[fitur_lengkap].fillna(0)
    
    # Normalisasi & Hitung Cosine Similarity
    scaler = MinMaxScaler()
    matriks_normalisasi = scaler.fit_transform(matriks_mentah)
    similarity_matrix = cosine_similarity(matriks_normalisasi)
    sim_df = pd.DataFrame(similarity_matrix, index=df['Judul'], columns=df['Judul'])
    
    return df, sim_df

# Load fungsi di atas
with st.spinner("Memuat Database Game & Menghitung Matriks Kemiripan..."):
    df, sim_df = load_data_and_model()

# ==========================================
# 3. NAVIGASI SIDEBAR
# ==========================================
menu = st.sidebar.selectbox("Pilih Menu:", ["📊 Evaluasi Aset (Inventory)", "🔍 Cari Rekomendasi Game Baru"])

# ==========================================
# 4. HALAMAN 1: EVALUASI ASET SAAT INI
# ==========================================
if menu == "📊 Evaluasi Aset (Inventory)":
    st.header("Status Inventaris RAD Playstation")
    
    df_rental = df[df['Status_Inventaris'] == 'Ada'].copy()
    
    def tentukan_status_aset(row):
        median_sewa = df_rental['Total_Sewa'].median()
        q1_sewa = df_rental['Total_Sewa'].quantile(0.25)
        
        if row['Total_Sewa'] > median_sewa or row['Rating_Global'] >= 4.5:
            return "🟢 PERTAHANKAN"
        elif row['Size_GB'] > 80 and row['Total_Sewa'] < q1_sewa:
            return "🔴 HAPUS"
        else:
            return "🟡 MONITOR"
            
    df_rental['Saran_DSS'] = df_rental.apply(tentukan_status_aset, axis=1)
    
    # Menampilkan metrik ringkas
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Game Dimiliki", len(df_rental))
    col2.metric("Saran Hapus (Hemat Storage)", len(df_rental[df_rental['Saran_DSS'] == '🔴 HAPUS']))
    col3.metric("Rata-rata Rating Aset", round(df_rental['Rating_Global'].mean(), 2))
    
    # Menampilkan tabel interaktif
    st.dataframe(df_rental[['Judul', 'Genre', 'Size_GB', 'Rating_Global', 'Total_Sewa', 'Saran_DSS']].sort_values(by='Total_Sewa', ascending=False), use_container_width=True)

# ==========================================
# 5. HALAMAN 2: REKOMENDASI PENAMBAHAN
# ==========================================
elif menu == "🔍 Cari Rekomendasi Game Baru":
    st.header("Cari Rekomendasi Penambahan Game")
    st.info("Sistem akan mencari game PS4/PS5 yang **paling mirip** dengan input Anda, dan **belum dimiliki** oleh rental.")
    
    kata_kunci = st.text_input("Masukkan nama game referensi (Misal: FC 26, Tekken, GTA):")
    top_n = st.slider("Jumlah Rekomendasi:", min_value=1, max_value=10, value=5)
    
    if st.button("Cari Rekomendasi") and kata_kunci:
        # A. Fuzzy Search
        daftar_judul = df['Judul'].astype(str).tolist()
        best_match, score = process.extractOne(kata_kunci, daftar_judul, scorer=fuzz.token_set_ratio)
        
        if score < 70:
            st.error(f"❌ Game '{kata_kunci}' tidak dikenali di database.")
        else:
            judul_resmi = best_match
            st.success(f"🔎 Game Acuan Terdeteksi: **{judul_resmi}** (Akurasi Pencarian: {score}%)")
            
            # B. Ambil Skor & Filter
            skor_kemiripan = sim_df[judul_resmi].reset_index()
            skor_kemiripan.columns = ['Judul', 'Skor_Kemiripan']
            df_kandidat = pd.merge(skor_kemiripan, df[['Judul', 'Status_Inventaris', 'Bisa_PS4', 'Bisa_PS5', 'Genre', 'Size_GB']], on='Judul', how='left')
            df_kandidat = df_kandidat[df_kandidat['Judul'] != judul_resmi]
            
            # Terapkan Filter Bisnis
            df_kandidat = df_kandidat[df_kandidat['Status_Inventaris'].astype(str).str.strip().str.lower() != 'ada']
            valid_values = ['1', '1.0', 'yes', 'true', 'ya']
            kondisi_ps4 = df_kandidat['Bisa_PS4'].astype(str).str.strip().str.lower().isin(valid_values)
            kondisi_ps5 = df_kandidat['Bisa_PS5'].astype(str).str.strip().str.lower().isin(valid_values)
            df_kandidat = df_kandidat[kondisi_ps4 | kondisi_ps5]
            
            hasil_rekomendasi = df_kandidat.sort_values(by='Skor_Kemiripan', ascending=False).head(top_n)
            
            if hasil_rekomendasi.empty:
                st.warning("⚠️ Tidak ada rekomendasi. Mungkin semua game yang mirip sudah dimiliki oleh rental, atau bukan game PS4/PS5.")
            else:
                hasil_rekomendasi = hasil_rekomendasi.reset_index(drop=True)
                hasil_rekomendasi.index += 1
                hasil_rekomendasi['Skor_Kemiripan'] = (hasil_rekomendasi['Skor_Kemiripan'] * 100).round(2).astype(str) + "%"
                
                # Fungsi pelabelan platform
                def label_platform(row):
                    platforms = []
                    if str(row['Bisa_PS4']).strip().lower() in valid_values: platforms.append('PS4')
                    if str(row['Bisa_PS5']).strip().lower() in valid_values: platforms.append('PS5')
                    return ', '.join(platforms)
                    
                hasil_rekomendasi['Platform'] = hasil_rekomendasi.apply(label_platform, axis=1)
                
                # Tampilkan hasil dalam bentuk tabel yang cantik
                st.table(hasil_rekomendasi[['Judul', 'Skor_Kemiripan', 'Genre', 'Platform', 'Size_GB']])