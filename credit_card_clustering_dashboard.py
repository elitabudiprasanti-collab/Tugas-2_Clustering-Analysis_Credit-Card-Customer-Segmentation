import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)

from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Card Clustering Dashboard",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background-color: #F7F9FC;
}

[data-testid="stSidebar"] {
    background-color: #E8F0FE;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 1. TITLE ; Tujuan Dashboard
# =========================================================
st.title("💳 Customer Segmentation Dashboard")

st.markdown("---")

st.markdown("""

### 📌 Tentang Dashboard

Dashboard ini digunakan untuk **melakukan segmentasi pelanggan kartu kredit
berdasarkan pola penggunaan kartu**, seperti saldo, transaksi, pembayaran,
dan limit kredit.

Melalui teknik clustering, pelanggan dengan karakteristik yang serupa
akan dikelompokkan ke dalam cluster yang sama sehingga perusahaan dapat
lebih memahami profil dan perilaku pelanggan.

### 🎯 Tujuan Analisis

* Mengidentifikasi **kelompok pelanggan** berdasarkan perilaku penggunaan kartu kredit.
* Memahami **karakteristik** masing-masing segmen pelanggan.
* Mendukung penyusunan **strategi pemasaran** yang lebih tepat sasaran.
* Membantu pengembangan **program loyalitas dan layanan pelanggan**.

### 💡 Manfaat

* Menemukan pelanggan premium dengan nilai tinggi.
* Mengidentifikasi pelanggan yang aktif bertransaksi.
* Mendeteksi pelanggan dengan aktivitas penggunaan rendah.
* Mendukung pengambilan keputusan bisnis berbasis data.
  """)

# =========================================================
# 2. Cara Membaca Dashboard
# =========================================================
st.info("""
### 📖 Cara membaca dashboard:

1. Pilih **algoritma clustering** pada panel kiri.
2. Tentukan **jumlah cluster** yang diinginkan.
3. Lihat **Silhouette Score** untuk mengevaluasi kualitas cluster.
4. Perhatikan **Profil Cluster** untuk memahami karakteristik setiap kelompok pelanggan.
5. Gunakan Rekomendasi Bisnis sebagai dasar pengambilan keputusan.
""")

# =========================================================
# LOAD DATASET
# =========================================================
df = pd.read_csv("CC GENERAL.csv")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Pengaturan Dashboard")

algorithm = st.sidebar.selectbox(
    "Pilih Algoritma Clustering",
    [
        "KMeans",
        "Hierarchical",
        "DBSCAN",
        "GMM"
    ]
)

k = st.sidebar.slider(
    "Jumlah Cluster",
    min_value=2,
    max_value=10,
    value=4
)

# =========================================================
# 3. Ringkasan Dashboard (KPI)
# =========================================================
st.header("📈 Ringkasan Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Jumlah Pelanggan",
    df.shape[0]
)

col2.metric(
    "Jumlah Variabel",
    df.shape[1]
)

col3.metric(
    "Algoritma",
     algorithm
)

col4.metric(
    "Jumlah Cluster",
    k if algorithm != "DBSCAN" else "Otomatis"
)

# =========================================================
# 4. DATASET OVERVIEW
# =========================================================
st.header("📌 Dataset Overview")

with st.expander("📖 Penjelasan Dataset"):
    st.write("""
Dataset Credit Card Customer Segmentation berisi informasi perilaku
    penggunaan kartu kredit oleh pelanggan.

Penjelasan beberapa variabel utama:
- BALANCE       : saldo kartu kredit saat ini
- PURCHASES     : total transaksi pembelian
- CASH_ADVANCE  : penarikan tunai dari kartu kredit
- CREDIT_LIMIT  : batas kredit pelanggan
- PAYMENTS      : total pembayaran yang dilakukan
- TENURE        : Lama pelanggan menggunakan kartu kredit
""")
    
st.dataframe(df.head())

# =========================================================
# 5. UKURAN DATASET
# =========================================================

st.header("📏 Ukuran Dataset")


col1, col2 = st.columns(2)

col1.metric(
    "Jumlah Baris",
    df.shape[0]
)

col2.metric(
    "Jumlah Kolom",
    df.shape[1]
)

# =========================================================
# 6. NAMA VARIABEL
# =========================================================

st.header("🧾 Nama Variabel")

st.write(df.columns)

# =========================================================
# 7. MISSING VALUE
# =========================================================
st.header("🧹 Missing Value")

with st.expander("📖 Penjelasan Missing Value"):

    st.write("""
    Missing Value adalah **data yang kosong** atau belum terisi.

    Sebelum proses clustering dilakukan, data yang kosong perlu **ditangani**
    agar hasil analisis menjadi lebih akurat.

    Pada dashboard ini, missing value digantikan menggunakan **nilai rata-rata**
    dari masing-masing variabel.
    """)

missing_total = df.isnull().sum().sum()

st.metric(
    "Total Missing Value",
    missing_total
)
if missing_total == 0:
    st.success("✅ Tidak ditemukan missing value pada dataset.")
else:
    st.markdown(
    f"""
    <div style="
        padding:12px;
        border-radius:10px;
        background-color:#FDECEC;
        color:#C62828;
        border-left:5px solid #E57373;
        margin-bottom:10px;">
        ⚠️ Ditemukan <b>{missing_total}</b> missing value yang akan ditangani sebelum proses clustering.
    </div>
    """,
    unsafe_allow_html=True
)
st.dataframe(df.isnull().sum())

# =========================================================
# 8. STATISTIK DESKRIPTIF
# =========================================================

st.header("📊 Statistik Deskriptif")

with st.expander("📖 Penjelasan Statistik Deskriptif"):

    st.write("""
    Statistik deskriptif digunakan untuk memberikan **gambaran umum** mengenai data.

    Tabel berikut menampilkan informasi seperti nilai rata-rata (mean),
    nilai minimum, nilai maksimum, serta tingkat penyebaran data
    pada setiap variabel.
    """)
    
st.dataframe(df.describe())

# =========================================================
# HANDLE MISSING VALUE
# =========================================================

df.fillna(
    df.mean(numeric_only=True),
    inplace=True
)

# =========================================================
# DROP CUST_ID
# =========================================================

if "CUST_ID" in df.columns:

    df = df.drop("CUST_ID", axis=1)

# =========================================================
# STANDARDIZATION
# =========================================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)


# =========================================================
# 9. ELBOW METHOD
# =========================================================
st.header("📉 Elbow Method")

with st.expander("📖 Apa itu Elbow Method?"):

    st.write("""
    Elbow Method digunakan untuk membantu menentukan **jumlah cluster yang optimal**.

    Metode ini mengevaluasi perubahan nilai WCSS (Within Cluster Sum of Squares)
    untuk berbagai jumlah cluster yang dicoba.
    """)

st.warning("""
⚠️ Semakin kecil **nilai WCSS**, semakin baik data dalam cluster.

📌Titik yang membentuk pola seperti 'siku' (elbow) 
    biasanya dipilih sebagai **jumlah cluster yang paling optimal**.
""")

wcss = []

for i in range(1,11):

    km = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    km.fit(scaled_data)

    wcss.append(km.inertia_)

fig1, ax1 = plt.subplots(figsize=(8,5))

ax1.plot(
    range(1,11),
    wcss,
    marker='o'
)

ax1.set_title("Elbow Method")

ax1.set_xlabel("Jumlah Cluster")

ax1.set_ylabel("WCSS")

st.pyplot(fig1)

# =========================================================
# CLUSTERING
# =========================================================

if algorithm == "KMeans":

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster = model.fit_predict(
        scaled_data
    )

elif algorithm == "Hierarchical":

    model = AgglomerativeClustering(
        n_clusters=k
    )

    cluster = model.fit_predict(
        scaled_data
    )

elif algorithm == "DBSCAN":

    model = DBSCAN(
        eps=2,
        min_samples=5
    )

    cluster = model.fit_predict(
        scaled_data
    )

else:

    model = GaussianMixture(
        n_components=k,
        random_state=42
    )

    cluster = model.fit_predict(
        scaled_data
    )

# =========================================================
# ADD CLUSTER
# =========================================================

df["Cluster"] = cluster

# =========================================================
# 10. SILHOUETTE SCORE
# =========================================================
st.header("⭐ Silhouette Score")

with st.expander("📖 Apa itu Silhouette Score?"):

    st.write("""
    Silhouette Score digunakan untuk mengukur **kualitas hasil clustering**.

    Nilai ini menunjukkan **seberapa baik** data dalam suatu cluster
    dibandingkan dengan cluster lainnya.
    """)

st.info("""
📊 Semakin tinggi nilai Silhouette Score, 
    semakin baik pemisahan antar cluster yang terbentuk.
""")

    
if len(set(cluster)) > 1:

    score = silhouette_score(
        scaled_data,
        cluster
    )

    st.success(
        f"Silhouette Score : {score:.4f}"
    )

    if score > 0.7:
        st.success("Interpretasi: Kualitas cluster sangat baik.")
    elif score > 0.5:
        st.success("Interpretasi: Kualitas cluster baik.")
    elif score > 0.3:
        st.warning("Interpretasi: Kualitas cluster cukup baik.")
    else:
        st.error("Interpretasi: Kualitas cluster kurang baik.")

    st.info("""
    📊 Panduan Interpretasi Silhouette Score
    - 0.70 – 1.00 : Kualitas cluster sangat baik
    - 0.50 – 0.70 : Kualitas cluster baik
    - 0.30 – 0.50 : Kualitas cluster cukup baik
    - < 0.30      : Kualitas cluster kurang baik
    """)

else:

    st.warning(
        "Silhouette Score tidak dapat dihitung."
    )

# =========================================================
# 11. PCA VISUALIZATION
# =========================================================
st.header("🧭 PCA 2D Projection")
   
with st.expander("📖 Penjelasan Grafik PCA"):

    st.write("""
    PCA (Principal Component Analysis) digunakan untuk **mereduksi** (menyederhanakan)
    jumlah variabel menjadi dua dimensi sehingga hasil clustering
    dapat divisualisasikan dengan lebih mudah.
    """)

st.info("""
📊 Setiap titik mewakili pelanggan,
🖌️ Warna menunjukkan cluster.

📌Semakin **terpisah** antar kelompok pada grafik, 
    semakin **baik** hasil clustering yang diperoleh.
""")


pca = PCA(n_components=2)

pca_data = pca.fit_transform(
    scaled_data
)

df["PCA1"] = pca_data[:,0]

df["PCA2"] = pca_data[:,1]

fig2, ax2 = plt.subplots(figsize=(10,6))

sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="Set2",
    data=df,
    s=80,
    ax=ax2
)

ax2.set_title(
    f"PCA Visualization - {algorithm}"
)

st.pyplot(fig2)

# =========================================================
# 12. DISTRIBUSI CLUSTER
# =========================================================
st.header("📦 Distribusi Cluster")
  
with st.expander("📖 Apa yang ditampilkan pada bagian ini?"):

    st.write("""
    Distribusi cluster menunjukkan **jumlah pelanggan pada setiap cluster**
    yang terbentuk dari proses clustering.

    Tabel menampilkan jumlah dan persentase pelanggan pada masing-masing cluster,
    sedangkan grafik batang memudahkan perbandingan ukuran antar cluster.
    """)

st.info("""
📊 Semakin **besar** suatu cluster, semakin banyak pelanggan yang memiliki
    **karakteristik serupa** dalam kelompok tersebut.
""")

cluster_count = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_percent = (
    cluster_count / len(df) * 100
).round(2)

st.dataframe(
    pd.DataFrame({
        "Jumlah": cluster_count,
        "Persentase (%)": cluster_percent
    })
)

fig3, ax3 = plt.subplots(figsize=(8,5))

sns.barplot(
    x=cluster_count.index,
    y=cluster_count.values,
    ax=ax3
)

ax3.set_xlabel("Cluster")

ax3.set_ylabel("Jumlah Data")

st.pyplot(fig3)

# =========================================================
# 13. Profil dan Interpretasi Cluster (RINGKASAN CLUSTER)
# =========================================================
st.header("📊 Profil Pelanggan Berdasarkan Cluster")

with st.expander("📖 Penjelasan Profil Cluster"):

    st.write("""
    Bagian ini menampilkan **profil setiap cluster** yang terbentuk.

    Setiap cluster menunjukkan kelompok pelanggan dengan **karakteristik**
    dan pola penggunaan kartu kredit yang **berbeda**.
    """)

cluster_summary = (
    df.groupby("Cluster")
    .mean(numeric_only=True)
)
st.dataframe(cluster_summary)

st.subheader("🏷️ Profil dan Interpretasi Cluster")

for c in cluster_summary.index:

    purchases = cluster_summary.loc[c, "PURCHASES"]
    balance = cluster_summary.loc[c, "BALANCE"]
    credit = cluster_summary.loc[c, "CREDIT_LIMIT"]

    if purchases > cluster_summary["PURCHASES"].quantile(0.75):
        nama = "💳 High Spender"

    elif credit > cluster_summary["CREDIT_LIMIT"].quantile(0.75):
        nama = "👑 Premium Customer"

    elif balance < cluster_summary["BALANCE"].median():
        nama = "🌱 Low Activity Customer"

    else:
        nama = "📈 Regular Customer"

    st.markdown(f"### Cluster {c} → {nama}")

    st.write(f"""
    Rata-rata saldo (BALANCE)             : **{balance:.2f}**

    Rata-rata transaksi (PURCHASES)       : **{purchases:.2f}**

    Rata-rata limit kredit (CREDIT_LIMIT) : **{credit:.2f}**
    """)

    if nama == "💳 High Spender":

        st.success("""
        **Karakteristik**
        - Transaksi **tinggi**
        - **Aktif** menggunakan kartu kredit

        **Rekomendasi**
        - Program **cashback dan reward**
        - Prioritas **promosi loyalitas**
        """)

    elif nama == "👑 Premium Customer":
    
        st.info("""
        **Karakteristik**
        - Limit kredit **tinggi**
        - Potensi nilai pelanggan tinggi

        **Rekomendasi**
        - Tawarkan produk **premium**
        - Berikan **layanan eksklusif**
        """)

    elif nama == "🌱 Low Activity Customer":

        st.warning("""
        **Karakteristik**
        - Aktivitas transaksi **rendah**
        - Penggunaan kartu kredit masih minim

        **Rekomendasi**
        - Promo khusus untuk **meningkatkan penggunaan**
        - Program reaktivasi pelanggan
        """)

    else:

        st.info("""
        **Karakteristik**
        - Penggunaan kartu kredit relatif **stabil**

        **Rekomendasi**
        - Pertahankan **engagement pelanggan**
        - Berikan **promosi berkala**
        """)

    st.markdown("---")

# =========================================================
# 14. HEATMAP CLUSTER
# =========================================================
st.header("🔥 Heatmap Karakteristik Cluster")

with st.expander("📖 Penjelasan Heatmap"):

    st.write("""
    Heatmap membantu **membandingkan karakteristik** antar cluster.

    📏 Setiap baris menunjukkan cluster.
    📐 Setiap kolom menunjukkan variabel.

    🔴 Warna merah menunjukkan nilai rata-rata yang relatif tinggi.

    🔵 Warna biru menunjukkan nilai rata-rata yang relatif rendah.
    
    Dengan grafik ini, perbedaan karakteristik antar kelompok
    pelanggan dapat terlihat dengan lebih mudah.
    """)


cluster_summary = (
    df.groupby("Cluster")
    .mean(numeric_only=True)
)

fig4, ax4 = plt.subplots(figsize=(16,8))

sns.heatmap(
    cluster_summary,
    cmap="coolwarm",
    ax=ax4
)

ax4.set_title(
    "Heatmap Karakteristik Cluster"
)

st.pyplot(fig4)

# =========================================================
# 15. RADAR CHART
# =========================================================
st.header("🕸️ Radar Chart Tiap Cluster")

with st.expander("📖 Penjelasan Radar Chart"):

    st.write("""
    Radar Chart menunjukkan **profil masing-masing cluster** berdasarkan
    saldo (BALANCE), transaksi (PURCHASES), limit kredit (CREDIT_LIMIT),
    dan pembayaran (PAYMENTS).
    """)

st.info("""
📊 Semakin **luas are**a suatu cluster, 
    semakin **tinggi karakteristik** variabel yang dimiliki dibandingkan cluster lainnya.
""")

selected_features = [
    'BALANCE',
    'PURCHASES',
    'CREDIT_LIMIT',
    'PAYMENTS'
]

radar_data = cluster_summary[
    selected_features
]

# Normalisasi
radar_data = (
    radar_data - radar_data.min()
) / (
    radar_data.max() - radar_data.min()
)

labels = radar_data.columns

num_vars = len(labels)

angles = np.linspace(
    0,
    2*np.pi,
    num_vars,
    endpoint=False
).tolist()

angles += angles[:1]

fig5, ax5 = plt.subplots(
    figsize=(8,8),
    subplot_kw=dict(polar=True)
)

for i in range(len(radar_data)):

    values = radar_data.iloc[i].tolist()

    values += values[:1]

    ax5.plot(
        angles,
        values,
        linewidth=2,
        label=f'Cluster {i}'
    )

    ax5.fill(
        angles,
        values,
        alpha=0.1
    )

ax5.set_xticks(angles[:-1])

ax5.set_xticklabels(labels)

plt.legend(
    loc='upper right',
    bbox_to_anchor=(1.3,1.1)
)

st.pyplot(fig5)


# =========================================================
# Hasil Evaluasi Clustering
# =========================================================
st.header("📊 Hasil Evaluasi Clustering")

with st.expander("📖 Penjelasan Evaluasi Clustering"):

    st.write("""
    Tabel berikut menampilkan **metode clustering yang digunakan**
    serta **Silhouette Score** untuk menilai kualitas hasil clustering.
    """)

st.info("""
📊 Semakin **tinggi Silhouette Score**, 
    semakin **baik** pemisahan antar cluster yang terbentuk.
""")

comparison = pd.DataFrame({
    "Metode": [
        algorithm
    ],

    "Silhouette Score": [
        score if len(set(cluster)) > 1 else np.nan
    ]
})

st.dataframe(comparison)

# =========================================================
# 16. KESIMPULAN
# =========================================================
st.header("📌 Kesimpulan")

if score >= 0.7:
    kualitas = "sangat baik"
elif score >= 0.5:
    kualitas = "baik"
elif score >= 0.3:
    kualitas = "cukup"
else:
    kualitas = "kurang baik"

st.write(f"""
✅ Metode clustering yang digunakan adalah **{algorithm}**.

✅ Jumlah cluster yang terbentuk sebanyak **{len(set(cluster))}** cluster.

✅ Nilai Silhouette Score sebesar **{score:.3f}**

✅ Kualitas clustering tergolong **{kualitas}**.
""")
# =========================================================
# 17. Insight Hasil Clustering
# =========================================================
st.header("💡 Insight Hasil Clustering")

st.write("""
1. Pelanggan terbagi ke dalam beberapa kelompok dengan **karakteristik yang berbeda** 
   sehingga memerlukan strategi **pelayanan yang berbeda** pula.

2. Kelompok pelanggan dengan transaksi dan limit kredit **tinggi** 
   berpotensi menjadi **pelanggan premium**.

3. Kelompok dengan aktivitas transaksi **rendah** dapat menjadi **target promosi khusus** 
   untuk meningkatkan penggunaan kartu.

4. Hasil segmentasi pelanggan membantu perusahaan menyusun **strategi pemasaran** 
   yang lebih  efektif dan tepat sasaran.
""")

# =========================================================
# 18. DOWNLOAD DATA
# =========================================================

st.header("⬇️ Download Hasil Clustering")

csv = df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="hasil_clustering.csv",
    mime="text/csv"
)