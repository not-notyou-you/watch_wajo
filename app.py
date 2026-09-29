# app.py
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from inferensi import (cek_verify, estimasi_sebaran, gambar_frekuensi, gambar_sebaran, jalankan_forecast,
                       muat_artefak, validasi_riwayat)

AKAR = Path(__file__).parent
FOLDER_ARTEFAK = AKAR / 'artefak'

WARNA = {'bg': '#f4f5f0', 'card': '#fbfbf8', 'line': '#e2e4dc', 'tx': '#2e3a33', 'mut': '#7b867e',
         'acc': '#5f7f6b', 'hi': '#3d6150', 'wet': '#6f9480', 'dry': '#e6e8e1', 'lo': '#f7f8f3'}
WARNA_STATUS = {'Normal': '#5f7f6b', 'Waspada': '#b08a3e', 'Bahaya': '#a8574a'}

IKON_TETES = ('<svg class="logo" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
              'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3.5c-3 4.2-6 7.6-6 11a6 6 0 0 0 12 0'
              'c0-3.4-3-6.8-6-11z"/></svg>')
IKON_SLIDER = ('<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
               'stroke-linecap="round"><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/>'
               '<circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/></svg>')
IKON_INFO = ('<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
             'stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h0"/></svg>')

st.set_page_config(page_title='Prediksi Genangan Wajo', page_icon=str(AKAR / 'assets' / 'ikon.svg'), layout='wide')
st.markdown(f'<style>{(AKAR / "assets" / "style.css").read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)


@st.cache_resource(show_spinner='Memuat model...')
def sumber_daya():
    meta, model, peta, sampel, verify = muat_artefak(FOLDER_ARTEFAK)
    p_verify, lolos = cek_verify(model, meta, verify)
    gambar_frek = gambar_frekuensi(peta, WARNA['lo'], WARNA['hi'], WARNA['card'])
    return meta, model, peta, sampel, verify, p_verify, lolos, gambar_frek


def html(isi):
    st.markdown(isi, unsafe_allow_html=True)


def kartu():
    wadah = st.container()
    wadah.markdown('<span class="penanda-kartu"></span>', unsafe_allow_html=True)
    return wadah


def fmt(x, d=1):
    return f'{x:,.{d}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def tabel_editor(df, meta):
    d = df.copy()
    d['Tanggal'] = pd.to_datetime(d['Tanggal'], errors='coerce').dt.date
    konfig = {'Tanggal': st.column_config.DateColumn('Tanggal', format='YYYY-MM-DD', required=True),
              'Genangan_km2': st.column_config.NumberColumn('Genangan_km2', format='%.2f', min_value=0.0,
                                                            max_value=float(meta['luas_irisan_km2'])),
              'Orbit': st.column_config.SelectboxColumn('Orbit', options=[int(o) for o in meta['urutan_orbit']],
                                                        required=True)}
    for k in meta['kolom_input'][3:]:
        konfig[k] = st.column_config.NumberColumn(k, format='%.4f')
    return konfig, d


try:
    META, MODEL, PETA, SAMPEL, VERIFY, P_VERIFY, LOLOS_VERIFY, GAMBAR_FREK = sumber_daya()
except Exception as e:
    st.error(f'Artefak di folder "artefak/" gagal dimuat: {e}')
    st.stop()

teks_status = 'Model siap' if LOLOS_VERIFY else 'Verifikasi gagal'
kelas_status = '' if LOLOS_VERIFY else ' gagal'
html(f'''
<div class="top">
  <div class="brand">{IKON_TETES}
    <div class="title">
      <div class="h1">Prediksi Luas Genangan Kabupaten Wajo</div>
      <p>Riwayat observasi Sentinel-1, GPM, dan MODIS, lalu prakiraan luas genangan pada observasi berikutnya</p>
    </div>
  </div>
  <div class="status{kelas_status}"><span class="dot"></span>{teks_status}</div>
</div>''')

bar = st.container()
pesan = st.container()

if not LOLOS_VERIFY:
    st.warning(f'Hasil pipeline app ({P_VERIFY:.4f} km²) berbeda dari notebook ({VERIFY["prediksi_km2"]:.4f} km²) '
               f'melebihi toleransi {VERIFY["toleransi_km2"]} km². Periksa kecocokan versi artefak.')

tab_hasil, tab_data, tab_detail = st.tabs(['Ringkasan', 'Data riwayat', 'Detail & model'])

with tab_data:
    c1, c2 = st.columns([1.3, 2], gap='large')
    with c1:
        sumber = st.radio('Sumber data riwayat', ['Data contoh', 'Unggah CSV'], horizontal=True)
    df_awal = SAMPEL
    with c2:
        if sumber == 'Unggah CSV':
            u1, u2 = st.columns([2.2, 1], vertical_alignment='bottom')
            berkas = u1.file_uploader('Berkas CSV riwayat', type=['csv'])
            template = ','.join(META['kolom_input']) + '\n'
            u2.download_button('Unduh template', template, 'template_riwayat_wajo.csv', 'text/csv',
                               use_container_width=True)
            if berkas is not None:
                try:
                    df_awal = pd.read_csv(berkas)
                except Exception as e:
                    html(f'<div class="catatan err">CSV tidak terbaca: {e}</div>')
                    df_awal = pd.DataFrame(columns=META['kolom_input'])
            else:
                df_awal = pd.DataFrame(columns=META['kolom_input'])
    kolom_ada = [k for k in META['kolom_input'] if k in df_awal.columns]
    konfig, df_tampil = tabel_editor(df_awal[kolom_ada] if kolom_ada else df_awal, META)
    html('<div class="field-label">Tabel riwayat (dapat diedit, tambah baris di bagian bawah)</div>')
    df_edit = st.data_editor(df_tampil, column_config=konfig, num_rows='dynamic', hide_index=True,
                             use_container_width=True, height=460, key=f'editor_{sumber}')

n_baris = int(df_edit['Tanggal'].notna().sum()) if 'Tanggal' in df_edit else 0
tanda = df_edit.to_csv(index=False)

with bar:
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1.3], vertical_alignment='center')
    b1.markdown(f'<div class="stat"><span>Observasi riwayat</span><b>{n_baris}</b></div>', unsafe_allow_html=True)
    b2.markdown(f'<div class="stat"><span>Langkah prakiraan</span><b>{max(1, n_baris // 3)}</b></div>',
                unsafe_allow_html=True)
    b3.markdown(f'<div class="stat"><span>Riwayat minimal</span><b>{META["min_baris_riwayat"]}</b></div>',
                unsafe_allow_html=True)
    jalankan = b4.button('Jalankan prediksi', type='primary', use_container_width=True)

if jalankan or 'hasil' not in st.session_state:
    with pesan:
        proses = st.status('Prediksi sedang berjalan...', expanded=False)
    with proses:
        st.write('Memvalidasi tabel riwayat...')
        df_valid, galat, peringatan = validasi_riwayat(df_edit, META)
        galat = list(galat or [])
        if not galat and df_valid is None:
            galat = ['Data riwayat tidak dapat diproses.']
        hasil_baru = None
        if not galat:
            st.write(f'Menjalankan prakiraan untuk {len(df_valid)} observasi riwayat...')
            try:
                hasil_baru = {'riwayat': df_valid, 'fc': jalankan_forecast(MODEL, df_valid, META)}
            except Exception as e:
                galat.append(f'Model gagal menghasilkan prakiraan: {type(e).__name__}: {e}')
    if galat:
        proses.update(label=f'Prediksi gagal ({len(galat)} masalah ditemukan)', state='error', expanded=False)
        st.toast('Prediksi gagal. Lihat penyebabnya di bawah tombol.', icon=':material/error:')
    else:
        n_fc = len(hasil_baru['fc'])
        proses.update(label=f'Prediksi selesai: {n_fc} langkah prakiraan dihasilkan', state='complete', expanded=False)
        if jalankan:
            st.toast('Prediksi selesai.', icon=':material/check_circle:')
    st.session_state['hasil'] = hasil_baru
    st.session_state['galat'] = galat
    st.session_state['peringatan'] = peringatan
    st.session_state['tanda'] = tanda
    st.session_state['status_terakhir'] = 'gagal' if galat else 'selesai'

with pesan:
    galat_ada = st.session_state.get('galat', [])
    if galat_ada:
        daftar = ''.join(f'<li>{g}</li>' for g in galat_ada)
        html(f'<div class="catatan err"><b>Prediksi gagal.</b> Penyebab:<ul class="daftar-galat">{daftar}</ul></div>')
    elif not jalankan and st.session_state.get('status_terakhir') == 'selesai':
        html('<div class="catatan">Menampilkan hasil prediksi terakhir.</div>')
    for p in st.session_state.get('peringatan', []):
        html(f'<div class="catatan warn"><b>Peringatan:</b> {p}</div>')
    if st.session_state.get('tanda') != tanda:
        html('<div class="catatan warn">Tabel riwayat berubah sejak prediksi terakhir. Tekan <b>Jalankan prediksi</b> '
             'untuk memperbarui hasil.</div>')

hasil = st.session_state.get('hasil')

if not hasil:
    with tab_hasil:
        html(f'''<div class="kosong">{IKON_INFO}<div><b>Belum ada hasil.</b><br>
Prediksi belum berhasil dijalankan. Periksa penyebabnya di atas, lalu buka tab Data riwayat, isi atau unggah minimal {META["min_baris_riwayat"]} observasi, lalu tekan Jalankan prediksi.</div></div>''')
    st.stop()

with tab_hasil:
    riwayat, fc = hasil['riwayat'], hasil['fc']
    terakhir = riwayat.iloc[-1]
    p1 = fc.iloc[0]
    selisih = p1['Prediksi_km2'] - terakhir['Genangan_km2']
    tanda_selisih = '+' if selisih >= 0 else '−'
    persen_irisan = p1['Prediksi_km2'] / META['luas_irisan_km2'] * 100
    warna_status = WARNA_STATUS[p1['Status']]
    html(f'''
<div class="metrics">
  <div class="m"><span>Prakiraan observasi berikutnya · {p1["Tanggal"]:%d %b %Y}</span>
    <b>{fmt(p1["Prediksi_km2"])} km²</b>
    <small>± {fmt(p1["Galat_km2"])} km² · {fmt(persen_irisan)}% area irisan</small></div>
  <div class="m"><span>Perubahan dari observasi terakhir · {terakhir["Tanggal"]:%d %b %Y}</span>
    <b>{tanda_selisih}{fmt(abs(selisih))} km²</b>
    <small>terakhir {fmt(terakhir["Genangan_km2"])} km²</small></div>
  <div class="m"><span>Status genangan</span>
    <b style="color:{warna_status}">{p1["Status"]}</b>
    <small>waspada ≥ {fmt(META["status"]["waspada_km2"], 0)} · bahaya ≥ {fmt(META["status"]["bahaya_km2"], 0)} km²</small></div>
</div>''')

    g_kiri, g_kanan = st.columns([1.9, 1], gap='medium')
    with g_kiri, kartu():
        html('<h3>Riwayat dan prakiraan luas genangan</h3>')
        d_hist = riwayat[['Tanggal', 'Genangan_km2']].rename(columns={'Genangan_km2': 'Luas'})
        sambung = pd.DataFrame({'Tanggal': [terakhir['Tanggal']], 'Prediksi_km2': [terakhir['Genangan_km2']],
                                'Bawah_km2': [terakhir['Genangan_km2']], 'Atas_km2': [terakhir['Genangan_km2']]})
        d_fc = pd.concat([sambung, fc[['Tanggal', 'Prediksi_km2', 'Bawah_km2', 'Atas_km2']]], ignore_index=True)
        d_ambang = pd.DataFrame({'y': [META['status']['waspada_km2'], META['status']['bahaya_km2']],
                                 'label': ['Waspada', 'Bahaya']})
        sumbu_x = alt.X('Tanggal:T', title=None, axis=alt.Axis(format='%d %b %Y', labelAngle=0, tickCount=6))
        sumbu_y = lambda kolom: alt.Y(kolom, title='Luas genangan (km²)', scale=alt.Scale(zero=False))
        pita = alt.Chart(d_fc).mark_area(color=WARNA['acc'], opacity=0.16).encode(
            x=sumbu_x, y=sumbu_y('Bawah_km2:Q'), y2='Atas_km2:Q')
        garis_hist = alt.Chart(d_hist).mark_line(color='#b9c1b7', strokeWidth=1.4).encode(x=sumbu_x, y=sumbu_y('Luas:Q'))
        titik_hist = alt.Chart(d_hist).mark_circle(color=WARNA['tx'], size=34).encode(
            x=sumbu_x, y=sumbu_y('Luas:Q'), tooltip=[alt.Tooltip('Tanggal:T', format='%d %b %Y'),
                                            alt.Tooltip('Luas:Q', title='Observasi (km²)', format=',.1f')])
        garis_fc = alt.Chart(d_fc).mark_line(color=WARNA['acc'], strokeWidth=2, strokeDash=[5, 4]).encode(
            x=sumbu_x, y=sumbu_y('Prediksi_km2:Q'))
        titik_fc = alt.Chart(d_fc.iloc[1:]).mark_circle(color=WARNA['acc'], size=60).encode(
            x=sumbu_x, y=sumbu_y('Prediksi_km2:Q'),
            tooltip=[alt.Tooltip('Tanggal:T', format='%d %b %Y'),
                     alt.Tooltip('Prediksi_km2:Q', title='Prakiraan (km²)', format=',.1f'),
                     alt.Tooltip('Bawah_km2:Q', title='Batas bawah', format=',.1f'),
                     alt.Tooltip('Atas_km2:Q', title='Batas atas', format=',.1f')])
        ambang = alt.Chart(d_ambang).mark_rule(strokeDash=[2, 4], color=WARNA['mut'], opacity=0.7).encode(y=sumbu_y('y:Q'))
        label_ambang = alt.Chart(d_ambang).mark_text(align='left', dx=4, dy=-6, fontSize=11, color=WARNA['mut']).encode(
            y=sumbu_y('y:Q'), x=alt.value(0), text='label:N')
        grafik = (pita + ambang + label_ambang + garis_hist + titik_hist + garis_fc + titik_fc).properties(height=340)
        grafik = grafik.configure(font='Inter, Segoe UI, sans-serif', background=WARNA['card']).configure_view(stroke=None).configure_axis(
            labelColor=WARNA['mut'], titleColor=WARNA['mut'], gridColor=WARNA['line'], domainColor=WARNA['line'],
            tickColor=WARNA['line'], labelFontSize=11, titleFontSize=11, titleFontWeight='normal')
        st.altair_chart(grafik, use_container_width=True)
        html(f'''<div class="legend"><span class="sw tx"></span>observasi
<span class="sw acc"></span>prakiraan rekursif<span class="sw band"></span>pita galat uji per langkah</div>''')

    with g_kanan, kartu():
        html('<h3>Estimasi sebaran genangan</h3>')
        opsi = [f'{r.Langkah} · {r.Tanggal:%d %b %Y}' for r in fc.itertuples()]
        pilih = st.selectbox('Langkah prakiraan', opsi, label_visibility='collapsed') if len(opsi) > 1 else opsi[0]
        baris = fc.iloc[opsi.index(pilih)]
        sebaran, luas_peta = estimasi_sebaran(PETA, baris['Prediksi_km2'])
        st.image(gambar_sebaran(sebaran, WARNA['wet'], WARNA['dry'], WARNA['card']), use_container_width=True)
        html(f'''<div class="legend"><span class="sw wet"></span>tergenang<span class="sw dry"></span>kering
<span class="spacer"></span>{fmt(luas_peta)} km²</div>''')
    html('<div class="catatan kecil">Peta sebaran adalah estimasi berbasis frekuensi genangan historis: piksel yang '
         'paling sering tergenang diisi lebih dulu sampai luasnya sama dengan angka prakiraan. Peta ini bukan '
         'keluaran spasial model.</div>')

with tab_detail:
    d_kiri, d_kanan = st.columns([1.9, 1], gap='medium')
    with d_kanan, kartu():
        html('<h3>Frekuensi genangan historis</h3>')
        st.image(GAMBAR_FREK, use_container_width=True)
        html('<div class="legend"><span class="sw lo"></span>jarang<span class="sw hi"></span>selalu tergenang</div>')
    with d_kiri:
        with kartu():
            html('<h3>Tabel prakiraan</h3>')
            tabel = fc[['Langkah', 'Tanggal', 'Orbit', 'Prediksi_km2', 'Bawah_km2', 'Atas_km2', 'Status']].copy()
            tabel['Tanggal'] = tabel['Tanggal'].dt.strftime('%Y-%m-%d')
            st.dataframe(tabel, hide_index=True, use_container_width=True, column_config={
                'Prediksi_km2': st.column_config.NumberColumn('Prakiraan (km²)', format='%.1f'),
                'Bawah_km2': st.column_config.NumberColumn('Batas bawah (km²)', format='%.1f'),
                'Atas_km2': st.column_config.NumberColumn('Batas atas (km²)', format='%.1f')})
            st.download_button('Unduh prakiraan (CSV)', tabel.to_csv(index=False), 'prakiraan_genangan_wajo.csv',
                               'text/csv')

        mt = META['metrik_test']
        with kartu():
            html(f'''<h3>Tentang model</h3>
<div class="info-grid">
  <div><span>Model</span><b>{META["nama_model"]} · seed {META.get("seed", "-")}</b></div>
  <div><span>Lookback</span><b>{META["lookback"]} observasi</b></div>
  <div><span>MAE uji</span><b>{fmt(mt["MAE"], 2)} km²</b></div>
  <div><span>RMSE uji</span><b>{fmt(mt["RMSE"], 2)} km²</b></div>
  <div><span>R² uji</span><b>{fmt(mt["R2"], 3)}</b></div>
  <div><span>Skill vs persistence</span><b>{fmt(mt["skill_vs_persistence"], 3)}</b></div>
</div>
<p class="catatan kecil">Prakiraan bersifat rekursif: setiap langkah memakai hasil langkah sebelumnya, dengan tanggal
maju {round(META["median_gap_hari"])} hari, orbit bergantian, serta curah hujan dan MODIS dari klimatologi bulanan.
Lebar pita adalah MAE prosedur yang sama pada data uji. Luas dihitung di dalam area irisan
{fmt(META["luas_irisan_km2"])} km².</p>''')

html(f'<div class="foot">Skripsi · Prediksi Luas Genangan Air di Kabupaten Wajo · Sentinel-1 + MODIS + GPM · '
     f'{META["nama_model"]} · kontrak {META["versi_kontrak"]}</div>')