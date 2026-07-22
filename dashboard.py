import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Klasifikasi Penyakit Daun Tomat", 
    layout="wide", 
    page_icon="🍅"
)

# Memuat model
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('mobilenetv2_tomat_terbaik.keras', compile=False)
    return model

model = load_my_model()

# Database informasi dan penanganan penyakit
descriptions = {
    'Tomato___Bacterial_spot': {
        'penjelasan': "**Penyebab:** Bakteri *Xanthomonas campestris pv. vesicatoria*.\n\n**Gejala Visual:** Ditandai dengan kemunculan lesi atau bercak kecil berukuran melingkar tidak beraturan yang tampak basah (*water-soaked*) dan berwarna cokelat tua hingga hitam pada permukaan bawah daun. Seiring berkembangnya fase infeksi, bercak tersebut akan mengering dan dikelilingi oleh area klorosis berupa lingkaran kuning pucat (*halo*), yang menyebabkan jaringan tengah daun berlubang hingga mengalami kerontokan prematur (*defoliation*).",
        'penanganan': "**Cara Mengatasi:**\n* **Mekanis:** Segera pangkas dan musnahkan daun atau tanaman yang terinfeksi parah. Lakukan rotasi tanaman dan hindari penyiraman *overhead* yang membasahi daun.\n* **Kimiawi:** Aplikasikan bakterisida atau fungisida berbahan dasar tembaga (*copper*) secara berkala."
    },
    'Tomato___Early_blight': {
        'penjelasan': "**Penyebab:** Infeksi organisme jamur *Alternaria solani*.\n\n**Gejala Visual:** Menyerang area daun dengan bercak melingkar berwarna cokelat gelap hingga hitam. Ciri pembeda utama dari penyakit ini adalah terbentuknya pola cincin konsentris yang menyerupai papan target di pusat bercak. Akumulasi bercak yang meluas dapat segera membunuh seluruh helai daun.",
        'penanganan': "**Cara Mengatasi:**\n* **Mekanis:** Pastikan jarak tanam yang cukup untuk sirkulasi udara yang baik. Gunakan mulsa tanah untuk mencegah spora jamur memercik ke daun.\n* **Kimiawi:** Penyemprotan fungisida berbahan aktif klorotalonil atau mankozeb pada awal gejala terlihat."
    },
    'Tomato___Late_blight': {
        'penjelasan': "**Penyebab:** *Oomycete* (jamur air) *Phytophthora infestans*.\n\n**Gejala Visual:** Muncul bercak berukuran besar dengan bentuk tidak beraturan yang menyebar dari tepi atau ujung daun, dan berwarna kelabu gelap kehitaman. Pada kelembapan tinggi dan tahap penyakit lanjut, permukaan bawah daun sering tertutup dengan miselium atau jamur bakteri putih halus yang menyerupai kapas.",
        'penanganan': "**Cara Mengatasi:**\n* **Mekanis:** Penyakit ini sangat destruktif. Segera cabut tanaman yang terinfeksi dan bakar. Jaga area kebun tetap kering.\n* **Kimiawi:** Aplikasikan fungisida sistemik spektrum luas secara preventif saat cuaca dingin dan basah."
    },
    'Tomato___Leaf_Mold': {
        'penjelasan': "**Penyebab:** Aktivitas jamur *Passalora fulva* (bersinonim dengan *Cladosporium fulvum*).\n\n**Gejala Visual:** Penyakit ini umumnya berkembang pesat pada area budidaya dengan sirkulasi udara terbatas seperti rumah kaca, diawali dengan bercak klorosis berwarna kuning pucat di permukaan atas daun.",
        'penanganan': "**Cara Mengatasi:**\n* **Mekanis:** Tingkatkan ventilasi dan sirkulasi udara, terutama di dalam rumah kaca (*greenhouse*). Kurangi kelembapan lingkungan.\n* **Kimiawi:** Penggunaan fungisida berbahan aktif tembaga atau klorotalonil."
    },
    'Tomato___Septoria_leaf_spot': {
        'penjelasan': "**Penyebab:** Hifa infeksius dari jamur *Septoria lycopersici*.\n\n**Gejala Visual:** Ditandai oleh munculnya banyak bercak melingkar kecil pada seluruh helai daun. Ciri morfologi unik dari penyakit ini adalah tepi luarnya yang tegas berwarna cokelat tua dengan bintik-bintik hitam kecil di pusat atau tengah bercak.",
        'penanganan': "**Cara Mengatasi:**\n* **Mekanis:** Singkirkan sisa tanaman di akhir musim. Hindari menyiram air langsung ke dedaunan.\n* **Kimiawi:** Semprotkan fungisida pelindung secara rutin, terutama selama periode basah."
    },
    'Tomato___Spider_mites/Two-spotted_spider_mite': {
        'penjelasan': "**Penyebab:** Gangguan hama dari spesies *Tetranychus urticae*.\n\n**Gejala Visual:** Gejala awal berupa bintik-bintik kuning keputihan (*stippling*) pada permukaan atas daun. Hama ini menyebabkan kerusakan klorofil secara signifikan melalui aktivitas penusukan dan penghisapan cairan sel parenkim daun.",
        'penanganan': "**Cara Mengatasi:**\n* **Mekanis:** Tungau menyukai kondisi kering. Menyemprot dedaunan dengan semburan air yang kuat secara rutin dapat mengusir tungau dan merusak jaringnya.\n* **Kimiawi:** Gunakan sabun insektisida, minyak nimba (*neem oil*), atau mitisida khusus."
    },
    'Tomato___Target_Spot': {
        'penjelasan': "**Penyebab:** Patogen jamur *Corynespora cassiicola*.\n\n**Gejala Visual:** Penyakit ini memiliki kemiripan morfologi dengan Hawar Dini (*Early Blight*). Gejalanya dimulai dengan munculnya lesi nekrotik berukuran kecil yang meluas menjadi bercak konsentris cokelat terang hingga gelap.",
        'penanganan': "**Cara Mengatasi:**\n* **Mekanis:** Kurangi durasi daun dalam keadaan basah dengan menyiram pada pangkalnya di pagi hari. Tingkatkan jarak tanam.\n* **Kimiawi:** Fungisida spektrum luas dapat membantu mengendalikan penyebarannya."
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'penjelasan': "**Penyebab:** Virus TYLCV yang ditularkan secara eksklusif oleh vektor serangga kutu kebul (*Bemisia tabaci*).\n\n**Gejala Visual:** Menyebabkan anomali pertumbuhan yang sangat parah, ditunjukkan oleh ukuran daun yang mengecil secara signifikan. Terjadi penguningan (klorosis) yang mencolok pada bagian tepi dan sela urat daun, serta deformasi helai daun yang menggulung kaku ke atas menyerupai bentuk mangkuk.",
        'penanganan': "**Cara Mengatasi:**\n* **Mekanis:** Tidak ada obat untuk penyakit virus. Segera musnahkan tanaman yang terinfeksi.\n* **Vektor:** Fokus pada pengendalian Kutu Kebul menggunakan perangkap kuning (*yellow sticky traps*) atau insektisida."
    },
    'Tomato___Tomato_mosaic_virus': {
        'penjelasan': "**Penyebab:** Infeksi penyakit dari golongan *Tobamovirus* yang sangat menular.\n\n**Gejala Visual:** Infeksi virus ini menyebabkan diskolorasi sistemik berupa pola belang atau mosaik antara hijau terang dan hijau tua pada helaian daun. Selain kelainan warna, daun yang terinfeksi biasanya mengalami malformasi struktural seperti pengerutan dan penyempitan helaian daun.",
        'penanganan': "**Cara Mengatasi:**\n* **Mekanis:** Virus ini mudah menular melalui kontak mekanis. Cabut dan hancurkan tanaman bergejala.\n* **Sanitasi:** Cuci tangan dan sterilkan alat setelah menangani tanaman. Hindari produk tembakau di dekat kebun."
    },
    'Tomato___healthy': {
        'penjelasan': "**Kondisi:** Daun Sehat (*Healthy*).\n\n**Penjelasan:** Mengindikasikan kondisi integritas jaringan seluler dan epidermis daun tomat yang utuh tanpa adanya intervensi atau manifestasi klinis dari agen patogen. Karakteristik visualnya dicirikan oleh tingkat pigmentasi klorofil yang ideal, menghasilkan distribusi warna hijau yang konsisten di seluruh permukaan helai daun.",
        'penanganan': "**Saran Pemeliharaan:**\n* Teruskan penyiraman secara teratur di area perakaran, hindari membasahi daun.\n* Berikan pupuk NPK berimbang, dengan penekanan pada Fosfor dan Kalium saat berbunga.\n* Lakukan pengecekan visual mingguan."
    }
}

labels = list(descriptions.keys())

# Sidebar
with st.sidebar:
    st.image("logo.png", width=255)
    st.header("Cara Penggunaan")
    st.markdown("""
    1. Pastikan objek adalah **Daun Tomat**.
    2. Ambil atau pilih foto daun yang pencahayaannya **terang dan fokus**.
    3. Tekan **Browse files** atau *drag-and-drop* gambar pada area unggah.
    4. Tunggu sistem memproses gambar dan memunculkan hasil.
    """)
    st.write("---")
    st.header("Tentang Aplikasi")
    st.markdown("""
    Dashboard ini merupakan implementasi dari arsitektur MobileNetV2 untuk memfasilitasi klasifikasi kondisi daun tomat.
    
    **Pengembang:**
    * **Nama:** Revata Octathio
    * **Instansi:** Teknik Informatika, Universitas Gunadarma
    """)

# Konten Utama
st.title("Klasifikasi Penyakit Daun Tomat")
st.markdown("Silakan unggah foto daun tomat untuk dianalisis. Pastikan daun terlihat jelas dan terang.")

file = st.file_uploader("Format yang didukung: JPG, JPEG, PNG", type=["jpg", "png", "jpeg"])

# Fungsi pra-pemrosesan dan prediksi
def predict(image_data, model):
    size = (224, 224)    
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]

    img_final = preprocess_input(img_reshape) 
    prediction = model.predict(img_final)
    return prediction

# Menampilkan hasil prediksi
if file is not None:
    image = Image.open(file)
    with st.spinner('Sedang menganalisis citra...'):
        predictions = predict(image, model)
        result_index = np.argmax(predictions)
        class_name = labels[result_index]
        confidence = np.max(predictions) * 100
        
    nama_penyakit_bersih = class_name.replace('Tomato___', '').replace('_', ' ')
    BATAS_KEYAKINAN = 80.0
    col1, col2 = st.columns([1, 1.5]) 
    
    with col1:
        st.image(image, caption="Foto yang Diunggah", width=400)
        if confidence >= BATAS_KEYAKINAN:
            if class_name == 'Tomato___healthy':
                st.markdown(f"**Hasil Deteksi Gambar:** `Daun Sehat`")
            else:
                st.markdown(f"**Hasil Deteksi Gambar:** `Penyakit {nama_penyakit_bersih}`")
        else:
            st.markdown(f"**Hasil Deteksi Gambar:** `Citra Tidak Dikenali`")
            
    with col2:
        if confidence < BATAS_KEYAKINAN:
            st.error("⚠️ **SISTEM MENOLAK CITRA (Objek Tidak Dikenali)**")
            st.warning(f"Probabilitas klasifikasi terlalu rendah (`{confidence:.2f}%`).")
            st.write("Sistem mendeteksi bahwa citra yang diunggah tidak memiliki kecocokan fitur visual dengan dataset latih. Objek kemungkinan besar **bukan daun tomat yang valid**, tidak fokus, atau memiliki latar belakang yang rumit. Silakan unggah ulang foto daun tomat yang benar.")
        else:
            if class_name == 'Tomato___healthy':
                st.success(f"**HASIL DETEKSI CITRA:** Daun Sehat ({nama_penyakit_bersih})")
            else:
                st.error(f"**HASIL DETEKSI:** Terindikasi Penyakit {nama_penyakit_bersih}")
                
            # Akurasi / Probabilitas
            st.markdown(f"**Probabilitas Klasifikasi:** `{confidence:.2f}%`")
            st.progress(int(confidence) / 100)
            st.write("---")
            # Penjelasan penyakit dan solusi
            st.subheader("Penjelasan dan Cara Mengatasi")
            st.info(descriptions[class_name]['penjelasan'])
            st.warning(descriptions[class_name]['penanganan'])
