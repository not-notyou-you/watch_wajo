# inferensi.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def muat_artefak(folder):
    folder = Path(folder)
    meta = json.loads((folder / 'meta.json').read_text(encoding='utf-8'))
    model = torch.jit.load(str(folder / 'model.ts'), map_location='cpu')
    model.eval()
    with np.load(folder / 'freq_map.npz') as z:
        peta = {k: z[k] for k in z.files}
    sampel = pd.read_csv(folder / 'sample_history.csv')
    verify = json.loads((folder / 'verify.json').read_text(encoding='utf-8'))
    return meta, model, peta, sampel, verify


def turunkan_fitur(riwayat, baris_target, meta, y_hist_awal):
    A = float(meta['luas_irisan_km2'])
    L = int(meta['lookback'])
    df = pd.concat([pd.DataFrame(riwayat), pd.DataFrame([baris_target])], ignore_index=True)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    R = len(df) - 1
    if R < L + 1:
        raise ValueError(f'Riwayat minimal {L + 1} baris, diberikan {R}')
    for k in meta['kolom_input'][1:]:
        df[k] = pd.to_numeric(df[k], errors='coerce').astype(float)
    gap = (df['Tanggal'].diff().dt.total_seconds() / 86400).to_numpy(dtype=float, copy=True)
    gap[0] = float(meta['median_gap_hari'])
    turun = df['Orbit'].isin(meta['orbit_turun']).astype(float).values
    doy = df['Tanggal'].dt.dayofyear.values
    doy_sin, doy_cos = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    ndvi = df['NDVI'].where(df['NDVI_cov'] >= 0.1).values
    ndwi = df['NDWI'].where(df['NDWI_cov'] >= 0.1).values
    ndwi_valid = (df['NDWI_cov'] >= 0.5).astype(float).values
    rf7d = df['GPM_7d_mm'].values
    rf7d_35h = df.set_index('Tanggal')['GPM_7d_mm'].rolling('35D').mean().values
    hujan_antar = rf7d * np.minimum(gap, 7) / 7
    frac = df['Genangan_km2'].values / A
    xs = np.column_stack([frac, df['VV_mean_dB'].values, df['VV_p10_dB'].values, rf7d, ndwi, ndwi_valid,
                          gap, turun])[R - L:R]
    y = frac[:R]
    hist_mean = (float(y_hist_awal['jumlah']) + y.sum()) / (float(y_hist_awal['n']) + R)
    orbit = df['Orbit'].values
    sama = [j for j in range(R - 1, -1, -1) if orbit[j] == orbit[R]]
    y_sama = y[sama[0]] if sama else y[R - 1]
    xe = np.array([gap[R], turun[R], doy_sin[R], doy_cos[R],
                   df['GPM_24h_mm'].values[R], df['GPM_72h_mm'].values[R], rf7d[R], df['GPM_7d_max_mm'].values[R],
                   rf7d_35h[R], hujan_antar[R], ndvi[R], ndwi[R], df['NDWI_cov'].values[R],
                   df['NDWI_umur_hari'].values[R], ndwi_valid[R], hist_mean, y[R - 1] - y[R - 2], y_sama, A],
                  dtype=np.float64)
    return xs.astype(np.float64), xe


def normalisasi(xs, xe, meta):
    nx, nt = meta['norm_x'], meta['norm_x_target']
    xs = np.where(np.isnan(xs), np.array(nx['median']), xs)
    xe = np.where(np.isnan(xe), np.array(nt['median']), xe)
    xs = (xs - np.array(nx['mean'])) / np.array(nx['std'])
    xe = (xe - np.array(nt['mean'])) / np.array(nt['std'])
    return xs.astype(np.float32), xe.astype(np.float32)


def pulihkan_km2(out, meta):
    y = float(np.asarray(out).reshape(-1)[0]) * meta['norm_y']['std'] + meta['norm_y']['mean']
    if meta['target_mode'] == 'log':
        y = float(np.expm1(y))
    return float(np.clip(y, 0.0, meta['luas_irisan_km2']))


def prediksi_km2(model, riwayat, baris_target, meta, y_hist_awal):
    xs, xe = turunkan_fitur(riwayat, baris_target, meta, y_hist_awal)
    xs_n, xe_n = normalisasi(xs, xe, meta)
    with torch.no_grad():
        out = model(torch.from_numpy(xs_n[None]), torch.from_numpy(xe_n[None]))
    return pulihkan_km2(out.numpy(), meta)


def orbit_berikut(orbit_terakhir, meta):
    urutan = [int(o) for o in meta['urutan_orbit']]
    if orbit_terakhir not in urutan:
        return urutan[0]
    return urutan[(urutan.index(orbit_terakhir) + 1) % len(urutan)]


def forecast_rekursif(model, riwayat, meta, y_hist_awal, K):
    r = pd.DataFrame(riwayat).copy()
    r['Tanggal'] = pd.to_datetime(r['Tanggal'])
    hasil = []
    for _ in range(K):
        tgl = r['Tanggal'].iloc[-1] + pd.Timedelta(days=int(round(meta['median_gap_hari'])))
        baru = {'Tanggal': tgl, 'Genangan_km2': np.nan, 'Orbit': orbit_berikut(int(r['Orbit'].iloc[-1]), meta)}
        baru.update({k: meta['klimatologi_bulanan'][k][str(tgl.month)] for k in meta['klimatologi_bulanan']})
        p = prediksi_km2(model, r, baru, meta, y_hist_awal)
        baru['Genangan_km2'] = p
        r = pd.concat([r, pd.DataFrame([baru])], ignore_index=True)
        hasil.append({'Tanggal': tgl, 'Orbit': baru['Orbit'], 'Prediksi_km2': p})
    return pd.DataFrame(hasil)


def cek_verify(model, meta, verify):
    riwayat = pd.DataFrame(verify['input_rows'])
    target = {k: (np.nan if v is None else v) for k, v in verify['baris_target'].items()}
    p = prediksi_km2(model, riwayat, target, meta, verify['y_hist_awal'])
    return p, abs(p - verify['prediksi_km2']) <= verify['toleransi_km2']


def validasi_riwayat(df, meta):
    galat, peringatan = [], []
    kolom = meta['kolom_input']
    hilang = [k for k in kolom if k not in df.columns]
    if hilang:
        return None, [f'Kolom tidak ditemukan: {", ".join(hilang)}'], peringatan
    d = df[kolom].copy()
    d['Tanggal'] = pd.to_datetime(d['Tanggal'], errors='coerce')
    if d['Tanggal'].isna().any():
        galat.append(f'{int(d["Tanggal"].isna().sum())} baris memiliki Tanggal kosong atau tidak valid (format YYYY-MM-DD)')
        d = d.dropna(subset=['Tanggal'])
    for k in kolom[1:]:
        d[k] = pd.to_numeric(d[k], errors='coerce')
    d = d.sort_values('Tanggal', kind='stable')
    duplikat = int(d['Tanggal'].duplicated().sum())
    if duplikat:
        d = d.drop_duplicates('Tanggal', keep='last')
        peringatan.append(f'{duplikat} tanggal ganda dihapus, baris terakhir dipakai')
    d = d.reset_index(drop=True)
    if d['Genangan_km2'].isna().any():
        galat.append('Genangan_km2 wajib terisi pada semua baris')
    if d['Orbit'].isna().any():
        galat.append('Orbit wajib terisi pada semua baris')
    else:
        asing = sorted(set(d['Orbit'].astype(int)) - set(int(o) for o in meta['urutan_orbit']))
        if asing:
            peringatan.append(f'Orbit {asing} tidak dikenal model; diperlakukan sebagai orbit naik')
    terisi = 0
    for k, klim in meta['klimatologi_bulanan'].items():
        kosong = d[k].isna()
        if kosong.any():
            d.loc[kosong, k] = [klim[str(b)] for b in d.loc[kosong, 'Tanggal'].dt.month]
            terisi += int(kosong.sum())
    if terisi:
        peringatan.append(f'{terisi} sel kosong diisi klimatologi bulanan')
    luar = []
    for k, (lo, hi) in meta['rentang_input'].items():
        if k in d and ((d[k] < lo) | (d[k] > hi)).any():
            luar.append(k)
    if luar:
        peringatan.append(f'Nilai di luar rentang historis: {", ".join(luar)}')
    n_min = int(meta['min_baris_riwayat'])
    if len(d) < n_min:
        galat.append(f'Riwayat minimal {n_min} observasi, tersedia {len(d)}')
    return d, galat, peringatan


def y_hist_efektif(df, meta):
    awal = meta['y_hist_awal']
    batas = pd.Timestamp(awal['tanggal_batas'])
    sebelum = df[df['Tanggal'] < batas]
    jumlah = float(awal['jumlah']) - float((sebelum['Genangan_km2'] / meta['luas_irisan_km2']).sum())
    n = int(awal['n']) - len(sebelum)
    if n <= 0:
        return {'jumlah': 0.0, 'n': 0}
    return {'jumlah': max(jumlah, 0.0), 'n': n}


def galat_langkah(k, meta):
    g = meta['galat_per_langkah_km2']
    return float(g[k - 1]) if k <= len(g) else float(g[-1]) * k / len(g)


def status_luas(luas, meta):
    if luas >= meta['status']['bahaya_km2']:
        return 'Bahaya'
    if luas >= meta['status']['waspada_km2']:
        return 'Waspada'
    return 'Normal'


def jalankan_forecast(model, df, meta):
    K = max(1, len(df) // 3)
    fc = forecast_rekursif(model, df, meta, y_hist_efektif(df, meta), K)
    A = float(meta['luas_irisan_km2'])
    fc['Langkah'] = np.arange(1, K + 1)
    fc['Galat_km2'] = [galat_langkah(k, meta) for k in fc['Langkah']]
    fc['Bawah_km2'] = (fc['Prediksi_km2'] - fc['Galat_km2']).clip(0, A)
    fc['Atas_km2'] = (fc['Prediksi_km2'] + fc['Galat_km2']).clip(0, A)
    fc['Status'] = [status_luas(p, meta) for p in fc['Prediksi_km2']]
    return fc


def estimasi_sebaran(peta, luas_km2):
    frek = peta['frekuensi'].ravel()
    area = peta['px_area'].ravel().astype(np.float64)
    idx = np.flatnonzero(peta['irisan'].ravel())
    urut = idx[np.argsort(-frek[idx], kind='stable')]
    kum = np.cumsum(area[urut])
    n = 0 if luas_km2 <= 0 else min(int(np.searchsorted(kum, luas_km2, side='left')) + 1, len(urut))
    hasil = np.full(frek.shape, np.nan)
    hasil[idx] = 0.0
    hasil[urut[:n]] = 1.0
    return hasil.reshape(peta['frekuensi'].shape), float(kum[n - 1]) if n else 0.0


def _rgb(hex_warna):
    h = hex_warna.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def gambar_frekuensi(peta, lo='#f7f8f3', hi='#3d6150', latar='#fbfbf8'):
    f = np.nan_to_num(peta['frekuensi'] / 100.0, nan=0.0)[..., None]
    img = _rgb(lo) * (1 - f) + _rgb(hi) * f
    img[~peta['irisan']] = _rgb(latar)
    return img.astype(np.uint8)


def gambar_sebaran(sebaran, basah='#6f9480', kering='#e6e8e1', latar='#fbfbf8'):
    img = np.empty(sebaran.shape + (3,))
    img[:] = _rgb(latar)
    img[sebaran == 0] = _rgb(kering)
    img[sebaran == 1] = _rgb(basah)
    return img.astype(np.uint8)
