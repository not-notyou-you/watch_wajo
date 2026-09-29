# Watch Wajo

Aplikasi Streamlit untuk memprediksi luas genangan air di Kabupaten Wajo pada observasi berikutnya. Masukannya adalah riwayat observasi multisensor dari Sentinel-1, MODIS, dan GPM. Model yang dipakai adalah 1D-CNN.

## Tentang model

| Parameter | Nilai |
|---|---|
| Model | 1D-CNN · seed 43 |
| Lookback | 6 observasi |
| MAE uji | 33,64 km² |
| RMSE uji | 44,52 km² |
| R² uji | 0,292 |
| Skill vs persistence | 0,469 |

## Menjalankan secara lokal

Yang dibutuhkan: Git dan Python 3.10 sampai 3.12.

**1. Clone repo**

```bash
git clone https://github.com/<username>/<nama-repo>.git
cd <nama-repo>
```

**2. Buat virtual environment**

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependensi**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Jalankan aplikasi**

```bash
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`.

## File CSV uji coba

Di root repo ada enam file CSV untuk mencoba aplikasi. Setiap file mewakili satu kasus yang berbeda.

**Cara memakai**

1. Jalankan `streamlit run app.py`.
2. Buka tab **Data riwayat**, lalu pilih **Unggah CSV**.
3. Unggah salah satu file di folder **study_case**.
4. Tekan **Jalankan prediksi**, lalu lihat hasilnya di tab **Ringkasan**.

| File | Kasus | Yang diharapkan |
|---|---|---|
| `kasus1_kemarau_stabil.csv` | Musim kemarau (Agu–Des 2024). Genangan kecil dan stabil sekitar 42–59 km², curah hujan rendah. | Prakiraan tetap rendah, status **Normal**. |
| `kasus2_banjir_naik.csv` | Awal musim hujan (Des 2024–Jun 2025). Curah hujan naik tajam dan genangan membesar dari sekitar 70 km² menjadi 285 km². | Prakiraan naik, status berpotensi **Waspada** atau **Bahaya**. |
| `kasus3_banjir_surut.csv` | Pasca-banjir (Jun–Des 2024). Genangan memuncak di sekitar 350 km², lalu surut ke sekitar 80 km² seiring hujan berkurang. | Prakiraan melanjutkan tren turun. |
| `kasus4_data_bolong_acak.csv` | Data tidak rapi: urutan baris acak, ada satu tanggal ganda, dan beberapa sel NDVI, NDWI, GPM, dan VV kosong. | Aplikasi mengurutkan tanggal, menghapus duplikat, dan mengisi sel kosong dengan klimatologi bulanan. Muncul peringatan, prediksi tetap berjalan. |
| `kasus5_ekstrem_luar_rentang.csv` | Kejadian ekstrem: genangan mencapai 470 km² dan hujan 7 hari 420 mm. Ada juga satu baris dengan orbit 61 yang tidak dikenal model. | Muncul peringatan nilai di luar rentang historis dan orbit tidak dikenal. Hasilnya perlu dibaca dengan hati-hati. |
| `kasus6_riwayat_kurang.csv` | Hanya 4 observasi. | Prediksi ditolak dengan pesan galat riwayat minimal. |

Semua data di file ini sintetis dan dibuat hanya untuk uji coba. Data ini bukan hasil observasi satelit asli.

## Catatan

- Folder `artefak/` harus ada dan lengkap. Tanpa folder itu, model tidak bisa dimuat.
- Kalau muncul error `torch` saat install, periksa versi Python kamu. PyTorch 2.5.1 hanya mendukung Python 3.9 sampai 3.12.