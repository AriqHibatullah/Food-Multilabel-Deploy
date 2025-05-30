import joblib
import time
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fungsi import prediksi_gizi, hitung_akg, rekomendasi_makanan

# === Load Model ===
model = joblib.load("model/Model.pkl")
threshold = joblib.load("model/Thresholds.pkl")
df_gizi = pd.read_csv("dataset/Nilai Gizi.csv")
asumsi = df_gizi[['nama', 'takaran saji']]
df_makanan = pd.read_csv("dataset/Dataset Makanan.csv")
satuan = ["kj", "kkal", "gram", "gram", "gram"]

pca = joblib.load("scaler/pca_hog.pkl")
scale_hist = joblib.load("scaler/scaler histogram.pkl")
scale_hog = joblib.load("scaler/scaler hog.pkl")
scale_lbp = joblib.load("scaler/scaler lbp.pkl")
scale_glcm = joblib.load("scaler/scaler glcm.pkl")

# === Fungsi ===
def resize_image(image, max_height):
    width, height = image.size
    if height <= max_height:
        return image
    scale = max_height / height
    new_height = max_height
    new_width = int(width * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

# === UI/UX ===
st.set_page_config(layout="wide")
col1, col_pad, col2 = st.columns([1.2, 0.1, 2.6])

if "gizi_gambar" not in st.session_state:
    st.session_state.gizi_gambar = None
if "pred_labels" not in st.session_state:
    st.session_state.pred_labels = []
if "no_bg" not in st.session_state:
    st.session_state.no_bg = None
if "rekom" not in st.session_state:
    st.session_state.rekom = None
if "last_uploaded_image" not in st.session_state:
    st.session_state.last_uploaded_image = None

with col1 :
    st.title("Prediksi Gizi dari Foto Makanan")
    st.write("Prediksi makanan dan gizinya berbasis Machine Learning menggunakan model Multilabel Classification.")
    st.write(" ")

    st.write("Model yang digunakan")
    genre = st.radio(
        "Pilih model:",
        ('SVC', 'SVC Pro (coming soon)')
    )

    st.write(" ")
    st.write("Informasi lebih lanjut")
    with st.expander("Tentang pengembang"):
        st.markdown("""
            Dikembangkan oleh :\n
            - Muhammad Ariq Hibatullah
            - Firdaini Azmi
            - Reva Deshinta Isyana
        """)

    with st.expander("Tentang model"):
        st.markdown("""
            jelaskan tentang model
        """)

    with st.expander("Tentang perhitungan AKG"):
        st.markdown("""
            jelaskan tentang akg
        """)

with col2 :
    st.write("Silahkan upload foto makanan anda untuk mengetahui kandungan gizinya!")
    image = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])
    if image != st.session_state.last_uploaded_image:
        # Reset state kalau gambar berubah
        st.session_state.gizi_gambar = None
        st.session_state.pred_labels = []
        st.session_state.no_bg = None
        st.session_state.last_uploaded_image = image

    if st.button("Kirim"):
        if image is not None:
            with st.spinner("⏳ Sedang memproses..."):
                img = Image.open(image)

                time.sleep(5)
                gizi_gambar, pred_labels, img_nobg = prediksi_gizi(img, model, threshold, df_gizi, pca, scale_hist, scale_hog, scale_lbp, scale_glcm)
                if gizi_gambar is not None :
                    st.session_state.gizi_gambar = gizi_gambar
                    st.session_state.pred_labels = pred_labels
                    st.session_state.no_bg = img_nobg
                    st.session_state.show_result = True
                else:
                    st.warning('⚠️ Gagal memproses gambar.')
        else :
            st.warning('⚠️ Silahkan input gambar terlebih dahulu.')

    tab1, tab2, tab3 = st.tabs(["📊 Kandungan Gizi", "🩺 Kebutuhan AKG", "🍕 Rekomendasi Makanan"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.write("Rincian Gizi")
            if st.session_state.gizi_gambar is not None :
                img = Image.open(image)
                img_resized = resize_image(img, max_height=200)
                st.image(img_resized, caption="Gambar yang diupload")
                gizi_gambar = st.session_state.gizi_gambar
                pred_labels = st.session_state.pred_labels

                if not pred_labels :
                    st.error("❌ Tidak ada makanan yang terdeteksi.")
                else :
                    st.success("Makanan anda telah terdeteksi.")
                    st.write(f"✅ Makanan terdeteksi:", ", ".join(pred_labels))

                    st.write(" ")
                    st.write(" ")
                    st.write(" ")
                    st.write(" ")
                    st.write("🔎 Dengan asumsi:")
                    asumsi_terpilih = asumsi[asumsi['nama'].isin(pred_labels)]
                    for nama, takaran in zip(asumsi_terpilih['nama'], asumsi_terpilih['takaran saji']):
                        st.write(f"- **{nama}**: {takaran}")

                    st.write(" ")
                    st.write(" ")
                    st.write(" ")
                    st.write(" ")
                    with st.expander("Prediksi makanan salah?"):
                        st.markdown("""
                            Kesalahan pada prediksi biasanya terjadi karena proses 
                            penyiapan data khususnya di remove background. 
                            Silahkan cek hasil remove background dibawah ini :
                        """)
                        if st.session_state.no_bg is not None :
                            hasil_nobg = st.session_state.no_bg[:, :, ::-1]
                            st.image(hasil_nobg,caption="Hasil Remove Background", use_container_width=200)
                        st.markdown("""
                            Jika hasil gambar tersebut tidak menangkap visualisasi makanan,
                            maka model tidak dapat memprediksi label dengan benar. Tapi
                            jika hasil tersebut menunjukan visualisasi makanan yang jelas, 
                            maka itu berarti model kami belum bisa memprediksi gambar input
                            dengan benar.
                        """)

            else :
                st.info("🖼️ Belum ada input gambar.")

        with col2:
            if st.session_state.gizi_gambar is not None :
                karbo = gizi_gambar['karbohidrat'] * 4
                protein = gizi_gambar['protein'] * 4
                lemak = gizi_gambar['lemak'] * 9

                fig = px.pie(
                    names=["Protein", "Karbo", "Lemak"],
                    values=[protein, karbo, lemak],
                    hole=0.6,
                    title="Komposisi Gizi",
                    color=["Protein", "Karbo", "Lemak"],
                    color_discrete_map={
                        "Protein": "#1f77b4",
                        "Karbo": "#c1121f",
                        "Lemak": "#e63946"
                    }
                )
                st.plotly_chart(fig, key="komposisi_gizi_aktif")

                st.write("📊 Perkiraan Total Nilai Gizi:")
                for nutrisi, nilai, satuan_gizi in zip(gizi_gambar.index, gizi_gambar.values, satuan):
                    st.write(f"- **{nutrisi.capitalize()}**: {nilai:.2f} {satuan_gizi}")
            else:
                fig_kosong = px.pie(
                    names=["Belum ada data"],
                    values=[100],
                    hole=0.6,
                    title="Komposisi Gizi"
                )
                fig_kosong.update_traces(marker=dict(colors=["#d3d3d3"]))
                st.plotly_chart(fig_kosong, key="komposisi_gizi_kosong")
    
    with tab2:
        st.write("Masukkan data diri Anda untuk menghitung AKG:")
        col1, col2, col3 = st.columns([1, 1, 2])

        if "form_data" not in st.session_state:
            st.session_state.form_data = {
                "gender": "pria",
                "berat": 60,
                "aktivitas": "Tidak aktif",
                "usia": 30,
                "tinggi": 170
            }

        with col1:
            gender = st.selectbox("Jenis kelamin:", ["Pria", "Wanita"])
            berat = st.number_input("Berat badan (kg):", min_value=30, max_value=100, value=60, step=1)
            aktivitas = st.selectbox("Aktivitas fisik anda:", ["Tidak aktif", "Sedikit aktif", "Cukup aktif", "Aktif", "Sangat aktif"])

            if st.button("Update"):
                st.session_state.form_data = {
                    "gender": gender,
                    "berat": berat,
                    "aktivitas": aktivitas
                }
                st.session_state.show_result_akg = True

        with col2:
            usia = st.number_input("Usia:", min_value=20, max_value=150, value=30, step=1)
            tinggi = st.number_input("Tinggi badan (cm):", min_value=100, max_value=220, value=170, step=1)
            with st.expander("ℹ️ Penjelasan pilihan aktivitas"):
                st.markdown("""
                - **Tidak aktif**: Jarang olahraga  
                - **Sedikit aktif**: Olahraga 1-3 kali/minggu  
                - **Cukup aktif**: Olahraga 3-5 kali/minggu  
                - **Aktif**: Olahraga 6-7 kali/minggu  
                - **Sangat aktif**: Olahraga 2 kali sehari  
                """)

            if st.session_state.get("show_result_akg", False):
                st.session_state.form_data["usia"] = usia
                st.session_state.form_data["tinggi"] = tinggi

        with col3:
            # Buat cek value Aktual
            aktual_k = 0
            aktual_p = 0
            aktual_l = 0
            aktual_kalori = 0
            if st.session_state.get("gizi_gambar") is not None:
                aktual_k = round(st.session_state.gizi_gambar['karbohidrat'], 2)
                aktual_p = round(st.session_state.gizi_gambar['protein'], 2)
                aktual_l = round(st.session_state.gizi_gambar['lemak'], 2)
                aktual_kalori = round(st.session_state.gizi_gambar['kalori'], 2)
                data = {
                    "Kategori": ["Karbohidrat", "Protein", "Lemak"],
                    "Aktual": [aktual_k, aktual_p, aktual_l]
                }
            else :
                data = {
                    "Kategori": ["Karbohidrat", "Protein", "Lemak"],
                    "Aktual": [0, 0, 0]
                }

            # Buat cek value Target
            if st.session_state.get("show_result_akg", False):
                data_user = st.session_state.form_data
                akg, kebutuhan_k, kebutuhan_p, kebutuhan_l = hitung_akg(
                    data_user["gender"], data_user["berat"], data_user["tinggi"], data_user["usia"], data_user["aktivitas"]
                )
                if None in (kebutuhan_k, kebutuhan_p, kebutuhan_l):
                    data.update({"Target": [0, 0, 0]})
                    st.session_state["rekomendasi"] = None
                else :
                    data.update({"Target": [kebutuhan_k, kebutuhan_p, kebutuhan_l]})
                    if st.session_state.rekom is None :
                        st.session_state.rekom = rekomendasi_makanan(df_makanan, kebutuhan_k, kebutuhan_p, kebutuhan_l, aktual_k, aktual_p, aktual_l)
                    
                    target_akg = round(akg, 2)
                    rekomendasi = st.session_state.rekom
                    st.session_state["rekomendasi"] = rekomendasi

                    # Zona warna
                    zone1 = target_akg * 0.6
                    zone2 = target_akg * 0.8
                    zone3 = target_akg * 1.2

                    fig = go.Figure()

                    fig.update_layout(
                        shapes=[
                            dict(type="rect", x0=0, x1=zone1, y0=0, y1=1, fillcolor="#eb1425", line_width=0, layer='below'),
                            dict(type="rect", x0=zone1, x1=zone2, y0=0, y1=1, fillcolor="#fffb08", line_width=0, layer='below'),
                            dict(type="rect", x0=zone2, x1=zone3 + 300, y0=0, y1=1, fillcolor="#29bf12", line_width=0, layer='below'),
                            dict(type="line", x0=target_akg, x1=target_akg, y0=0, y1=1, line=dict(color="white", width=2, dash="dash"))
                        ],
                        barmode="overlay"
                    )

                    fig.add_trace(go.Bar(
                        x=[aktual_kalori],
                        y=[0.5],
                        orientation='h',
                        marker=dict(color="white"),
                        text=f"{aktual_kalori} kkal",
                        textfont=dict(color="white"),
                        textposition="outside",
                        name="Kalori didapat",
                        width=0.6
                    ))

                    fig.add_annotation(
                        x=target_akg + 50,
                        y=0.3,
                        text=f"{target_akg} kkal",
                        showarrow=False,
                        font=dict(size=12, color="white"),
                        align='left'
                    )

                    fig.update_layout(
                        height=120,
                        margin=dict(l=30, r=30, t=40, b=30),
                        title="Kebutuhan AKG",
                        xaxis=dict(title="Kalori yang didapat VS kebutuhan AKG (kkal)", range=[0, target_akg + 300]),
                        yaxis=dict(showticklabels=False),
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True, key="bullet")

                    st.write(" ")
                    st.write("〽️ Kebutuhan AKG yang diperlukan :")
                    st.write(f"- Kalori: {akg} kkal")
                    st.write(f"- Karbohidrat: {kebutuhan_k} gram")
                    st.write(f"- Protein: {kebutuhan_p} gram")
                    st.write(f"- Lemak: {kebutuhan_l} gram")

            else :
                data.update({"Target": [0, 0, 0]})

    with tab3:
        data.update({"Rekomendasi": [aktual_k, aktual_p, aktual_l]})
        if st.session_state.gizi_gambar is not None :
            st.write("Rekomendasi makanan untuk memenuhi AKG")
            st.caption("Dengan asumsi takaran saji 100 gram.")
            if "rekomendasi" in st.session_state and st.session_state["rekomendasi"] is not None:
                daftar_rekomendasi = pd.DataFrame(st.session_state["rekomendasi"])
                tampilkan = ['Nama', 'karbohidrat', 'protein', 'lemak', 'kalori', 'energi', 'label']
                st.dataframe(daftar_rekomendasi[tampilkan].reset_index(drop=True))

                col1, col2 = st.columns([1,2])
                with col1:
                    st.write(' ')
                    st.write(' ')
                    choose = st.multiselect("Pilih makanan rekomendasi:", 
                                            options=daftar_rekomendasi["Nama"].tolist(), 
                                            key="rek",
                                            help="Pilih makanan yang direkomendasikan untuk memunculkan nilai gizi"
                                            )

                    st.write(' ')
                    st.write('📊 Total asupan gizi:')
                    if choose:
                        pilihan_df = daftar_rekomendasi[daftar_rekomendasi["Nama"].isin(choose)]
                        predik_k = round(pilihan_df["karbohidrat"].sum(), 2) + aktual_k
                        predik_p = round(pilihan_df["protein"].sum(), 2) + aktual_p
                        predik_l = round(pilihan_df["lemak"].sum(), 2) + aktual_l
                        predik_kalori = round(pilihan_df["kalori"].sum(), 2) + aktual_kalori

                        predik_k = round(predik_k, 2)
                        predik_p = round(predik_p, 2)
                        predik_l = round(predik_l, 2)
                        predik_kalori = round(predik_kalori, 2)

                        data.update({"Rekomendasi": [predik_k, predik_p, predik_l]})

                        st.write(f"- Karbohidrat: {predik_k} g")
                        st.write(f"- Protein: {predik_p} g")
                        st.write(f"- Lemak: {predik_l} g")
                        st.write(f"- Kalori: {predik_kalori} kkal")

                with col2:
                    df_user = pd.DataFrame(data)
                    df_melted = df_user.melt(
                        id_vars="Kategori", 
                        value_vars=["Target", "Rekomendasi", "Aktual"],
                        var_name="Tipe", 
                        value_name="Nilai"
                    )

                    fig = px.bar(
                        df_melted, x="Kategori", y="Nilai", color="Tipe",
                        barmode="group", text="Nilai",
                        title="Gizi dari rekomendasi",
                        color_discrete_map={
                            "Aktual": "#fee440",
                            "Rekomendasi" : "#f15bb5",
                            "Target": "#9b5de5"
                        }
                    )
        
                    fig.update_traces(textposition='outside')
                    fig.update_layout(yaxis_range=[0, max(df_melted["Nilai"]) * 1.2])
                    st.plotly_chart(fig, key = "progress")

            else:
                st.info("💡 Rekomendasi akan muncul setelah Anda mengisi data AKG dan upload gambar makanan.")
        else: 
            st.info("💡 Rekomendasi akan muncul setelah Anda mengisi data AKG dan upload gambar makanan.")
