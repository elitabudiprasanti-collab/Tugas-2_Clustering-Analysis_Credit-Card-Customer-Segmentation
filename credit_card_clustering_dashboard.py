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

# =========================================================
# 1. TITLE ; Tujuan Dashboard
# =========================================================
st.title("💳 Customer Segmentation Dashboard")

st.markdown("---")

st.markdown("""

### 📌 Tentang Dashboard

Dashboard ini digunakan untuk melakukan segmentasi pelanggan kartu kredit
berdasarkan pola penggunaan kartu, seperti saldo, transaksi, pembayaran,
dan limit kredit.

Melalui teknik clustering, pelanggan dengan karakteristik yang serupa
akan dikelompokkan ke dalam cluster yang sama sehingga perusahaan dapat
lebih memahami profil dan perilaku pelanggan.

### 🎯 Tujuan Analisis

* Mengidentifikasi kelompok pelanggan berdasarkan perilaku penggunaan kartu kredit.
* Memahami karakteristik masing-masing segmen pelanggan.
* Mendukung penyusunan strategi pemasaran yang lebih tepat sasaran.
* Membantu pengembangan program loyalitas dan layanan pelanggan.

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
📖 Cara membaca dashboard:

1. Pilih algoritma clustering pada panel kiri.
2. Tentukan jumlah cluster yang diinginkan.
3. Lihat nilai Silhouette Score untuk mengevaluasi kualitas cluster.
4. Perhatikan Profil Cluster untuk memahami karakteristik setiap kelompok pelanggan.
5. Gunakan Rekomendasi Bisnis sebagai dasar pengambilan keputusan.
""")

# =========================================================
# LOAD DATASET
# =========================================================
df = pd.read_csv("CC GENERAL.csv")

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
- BALANCE : saldo kartu kredit saat ini
- PURCHASES : total transaksi pembelian
- CASH_ADVANCE : penarikan tunai dari kartu kredit
- CREDIT_LIMIT : batas kredit pelanggan
- PAYMENTS : total pembayaran yang dilakukan
- TENURE : Lama pelanggan menggunakan kartu kredit
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

st.info("""
Missing Value adalah data yang kosong atau tidak memiliki nilai pada suatu variabel.

Data yang hilang dapat mempengaruhi hasil analisis dan kualitas clustering.
Oleh karena itu, sebelum proses clustering dilakukan, missing value perlu
diidentifikasi dan ditangani terlebih dahulu.

Pada dashboard ini, missing value akan digantikan menggunakan nilai rata-rata
(mean) dari masing-masing variabel numerik.
""")

missing_total = df.isnull().sum().sum()

st.metric(
    "Total Missing Value",
    missing_total
)
if missing_total == 0:
    st.success("✅ Tidak ditemukan missing value pada dataset.")
else:
    st.warning(
        f"⚠️ Ditemukan {missing_total} missing value yang akan ditangani sebelum proses clustering."
    )
st.dataframe(df.isnull().sum())

# =========================================================
# 8. STATISTIK DESKRIPTIF
# =========================================================

st.header("📊 Statistik Deskriptif")

st.info("""
Tabel berikut menampilkan ringkasan kondisi data, seperti nilai rata-rata,
nilai terkecil, nilai terbesar, dan tingkat penyebaran data pada setiap variabel.
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
# 9. ELBOW METHOD
# =========================================================
st.header("📉 Elbow Method")

st.info("""
Elbow Method digunakan untuk membantu menentukan jumlah cluster yang optimal.

Semakin kecil nilai WCSS, semakin baik data dalam cluster.
Titik yang membentuk 'siku' (elbow) biasanya dipilih sebagai jumlah cluster terbaik.
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

    st.caption("""
    Panduan Silhouette Score:
    - 0.70 - 1.00 : Sangat Baik
    - 0.50 - 0.70 : Baik
    - 0.30 - 0.50 : Cukup
    - < 0.30 : Kurang Baik
    """)

else:

    st.warning(
        "Silhouette Score tidak dapat dihitung."
    )

# =========================================================
# 11. PCA VISUALIZATION
# =========================================================
st.header("🧭 PCA 2D Projection")

st.info("""
PCA digunakan untuk mereduksi banyak variabel menjadi dua dimensi 
sehingga hasil clustering dapat divisualisasikan.

Semakin terpisah antar kelompok pada grafik PCA, 
semakin baik kualitas hasil clustering yang terbentuk.
""")

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

st.info("""
Bagian ini menjelaskan karakteristik masing-masing cluster yang terbentuk
berdasarkan pola penggunaan kartu kredit pelanggan. Setiap cluster
merepresentasikan kelompok pelanggan dengan perilaku yang berbeda.
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

    st.write(
        f"""
        - Rata-rata saldo (BALANCE): {balance:.2f}
        - Rata-rata transaksi (PURCHASES): {purchases:.2f}
        - Rata-rata limit kredit (CREDIT_LIMIT): {credit:.2f}
        """
    )
    
if nama == "💳 High Spender":

    st.success("""
    Karakteristik:
    • Frekuensi dan nilai transaksi tinggi
    • Aktif menggunakan kartu kredit

    Rekomendasi:
    • Berikan program cashback dan reward
    • Target utama promosi dan loyalty program
    """)

elif nama == "👑 Premium Customer":

    st.success("""
    Karakteristik:
    • Memiliki limit kredit tinggi
    • Potensi nilai pelanggan tinggi

    Rekomendasi:
    • Tawarkan produk premium
    • Berikan layanan eksklusif dan prioritas
    """)

elif nama == "🌱 Low Activity Customer":

    st.warning("""
    Karakteristik:
    • Aktivitas transaksi relatif rendah
    • Pemanfaatan kartu kredit masih rendah

    Rekomendasi:
    • Berikan promo khusus untuk meningkatkan penggunaan kartu
    • Lakukan kampanye reaktivasi pelanggan
    """)

else:

    st.info("""
    Karakteristik:
    • Penggunaan kartu kredit relatif stabil
    • Tidak terlalu tinggi maupun terlalu rendah

    Rekomendasi:
    • Pertahankan engagement pelanggan
    • Berikan promosi berkala untuk menjaga aktivitas transaksi
    """)
    
    st.markdown("---")
    
# =========================================================
# 14. HEATMAP CLUSTER
# =========================================================
st.header("🔥 Heatmap Karakteristik Cluster")

st.info("""
Heatmap menunjukkan rata-rata karakteristik setiap cluster.

Setiap baris menunjukkan cluster,
setiap kolom menunjukkan variabel.

Warna merah berarti nilai rata-rata tinggi,
warna biru berarti nilai rata-rata rendah.
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

st.info("""
Radar Chart menunjukkan profil masing-masing cluster berdasarkan
saldo, transaksi, limit kredit, dan pembayaran.

Semakin luas area suatu cluster, semakin tinggi karakteristik
variabel yang dimiliki dibandingkan cluster lainnya.
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

st.info("""
Bagian ini menampilkan metode clustering yang digunakan
serta nilai Silhouette Score yang diperoleh.
Nilai Silhouette Score digunakan untuk mengevaluasi kualitas
pemisahan cluster yang terbentuk.
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
st.header("📌 Kesimpulan Dashboard")

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
• Pelanggan terbagi ke dalam beberapa kelompok dengan karakteristik yang berbeda sehingga memerlukan strategi pelayanan yang berbeda pula.

• Kelompok pelanggan dengan transaksi dan limit kredit tinggi berpotensi menjadi pelanggan premium.

• Kelompok dengan aktivitas transaksi rendah dapat menjadi target promosi khusus untuk meningkatkan penggunaan kartu.

• Hasil segmentasi pelanggan membantu perusahaan menyusun strategi pemasaran yang lebih  efektif dan tepat sasaran.
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