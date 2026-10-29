## 2026-10-23 — Usulan aturan QC sinyal (Orang 2) — BELUM DISETUJUI

**Konteks.** Aturan PROJECT_BRIEF 10.2 (gagal jika ada lead dengan > 20% NaN, atau >= 3 lead datar) diterapkan pada 300 rekaman sampel (fold 0-16, acak: 100 positif + 200 negatif) dan 20 rekaman pilot. Hasil: 299 dari 300 sampel lolos.

**Temuan.** 4 dari 319 rekaman (1,3%) memiliki satu lead dengan artefak besar (maksimum 9,8-22,5 mV; p99 5,5-18,7 mV) sementara 11 lead lain normal (maksimum <= 2,4 mV). Aturan brief meloloskan keempatnya; setelah clip +-3 mV, lead itu menjadi derau jenuh. Ambang amplitudo 20 mV hanya menangkap 2 dari 4. Dua rekaman bervoltase tinggi yang tampak sah (maksimum 3,8 dan 5,2 mV, beberapa lead tinggi sekaligus) tidak termasuk kasus rusak.

**Usulan (perlu persetujuan tim).** Tambahkan aturan: gagal jika ada lead dengan |x| > 8 mV. Ambang ini berada di celah antara 5,2 mV (tampak sah) dan 9,8 mV (rusak) pada data yang ada. Alternatif: hanya menandai (`amplitude_flag`), atau menolkan lead yang rusak dan mempertahankan rekamannya.

**Catatan.** Dasar ambang hanya 6 rekaman. Validasi ulang pada rekaman tambahan dari fold 0-16 sebelum dikunci; jangan menyetel ambang memakai fold 19.


## 2026-10-23 — Clipping ±3 mV dipertahankan sementara (DISETUJUI; efektivitas BELUM divalidasi)

**Keputusan:** clipping ±3 mV tetap dipakai sesuai PROJECT_BRIEF 10.3 demi konsistensi dengan brief. Efektivitasnya terhadap performa dan stabilitas training belum divalidasi.

**Ketentuan:**
1. Clipping dilakukan dalam satuan mV (satuan header MIMIC-IV-ECG), setelah resample 500→100 Hz dan pemotongan 10 detik, sebelum standardisasi per-lead. Urutan ini sesuai brief 10.3.
2. Standardisasi tetap memakai statistik global yang dihitung dari data training saja. Clipping sebelum standardisasi juga melindungi statistik tersebut dari lead yang rusak. Tidak diubah secara sepihak.
3. `amplitude_flag` (> 20 mV) tetap penanda informatif. Aturan hard fail 8 mV per lead BELUM diadopsi (lihat entri usulan sebelumnya). QC mengikuti brief 10.2: gagal jika ada lead dengan > 20% NaN atau ≥ 3 lead datar.
4. Clipping tidak dimatikan dan standardisasi global tidak diubah sampai ablasi selesai dan keputusan baru dicatat di berkas ini.

**Rencana ablasi (setelah pipeline dan dataset stabil):**
- Bandingkan konfigurasi clipping (baseline brief ±3 mV, ambang lain, tanpa clipping dengan statistik robust, dan alternatif layak lain yang disepakati tim saat itu).
- Pemilihan HANYA memakai fold 0–16 (`internal_val_0_16`). Fold 17–19 tidak dipakai untuk memilih metode; fold 19 tidak disentuh sama sekali sebelum evaluasi akhir.
- Hasil dan keputusan akhir dicatat di decision_log.

**Catatan jejak fold pilot:** 20 rekaman pilot (`run_signal_qc_pilot`, tanpa filter fold) mencakup 2 rekaman di fold 18 dan 0 di fold 19. Pilot hanya QC sinyal tanpa label dan tanpa penyetelan threshold. Rekaman V6 rusak yang menjadi bukti usulan 8 mV berasal dari fold 4. Runner pilot tidak dipakai lagi untuk pemrosesan data.


## 2026-10-23 — Jalur QC tunggal sesuai brief, kontrak status signal_qc, validasi mV, pilot hanya fold 0-16 (DISETUJUI)

**Keputusan:**
1. Semua runner QC (`run_signal_qc_pilot`, `run_signal_qc_sample`, `run_signal_qc_brief`) memakai satu jalur di `src/dataset/signal_qc_core.py`: `read_ecg_record` -> `check_ecg_quality_brief` -> `prepare_ecg`, identik dengan `ECGDataset`. Fungsi lama (`load_ecg_record`, `check_ecg_quality`, `preprocess_ecg`) dipertahankan untuk tes lama, tetapi tidak dipakai pipeline (dijaga tes `test_runners_have_no_legacy_path`).
2. Kontrak status `signal_qc`: `pass` -> signal_ok=1; `qc_fail`, `preprocess_fail`, `load_fail` -> 0; `download_failed` dan `not_processed` -> kosong. Rekaman yang belum diperiksa tidak pernah dihitung lolos (`build_master` hanya menyimpan signal_ok == 1). `signal_qc.csv` produksi harus berisi SEMUA ed_stay_id master (`build_contract`); rekaman tanpa hasil = `not_processed`.
3. `read_ecg_record` memvalidasi satuan mV dari baris sinyal `.hea` mentah. wfdb mengisi satuan kosong menjadi mV secara diam-diam, sehingga `record.units` tidak cukup. Satuan kosong, tidak konsisten, atau bukan mV -> ValueError `unit_invalid` -> status `load_fail`. Alasan: clipping +-3 mV dan amplitude_flag hanya bermakna dalam mV.
4. Sampel pilot dibatasi fold 0-16. Fold 17-19 tidak boleh dipakai untuk menetapkan atau menyetel aturan QC.
5. Aturan QC TIDAK berubah: gagal jika ada lead > 20% NaN atau >= 3 lead datar. `amplitude_flag` (> 20 mV) tetap informatif; hard fail 8 mV tidak diadopsi. Clipping +-3 mV dipertahankan (lihat entri clipping); efektivitasnya belum divalidasi dan ablasinya direncanakan hanya pada fold 0-16.

**Keterbatasan bukti:**
- Validasi mV: 319 header di cache lokal semuanya menuliskan `mV` eksplisit (3.828 baris lead). Ini sampel dari fold 0-16 ditambah pilot, bukan seluruh dataset. Rekaman dengan satuan lain akan muncul sebagai `load_fail` dan perlu ditinjau, bukan diasumsikan mV.
- Inf tidak bisa ada di `.dat` 16-bit; penanganannya (diperlakukan seperti NaN) hanya diuji pada tingkat array.
- Hasil QC lama (`signal_qc_sample.csv`, aturan lama) bukan sumber QC dan tidak dipakai pipeline.
- Pilot lama (tanpa filter fold) memuat 2 rekaman fold 18 dan 0 fold 19; hanya QC sinyal tanpa label dan tanpa penyetelan. Berkas mentahnya masih ada di cache lokal dan tidak dipakai untuk keputusan apa pun.
- Cakupan QC pada seluruh dataset belum diketahui; baru sampel kecil (319 rekaman) yang diperiksa.

**Rencana ablasi clipping:** sama seperti entri clipping: bandingkan konfigurasi hanya pada fold 0-16 (`internal_val_0_16`), tidak pernah memakai fold 17-19 untuk memilih metode atau threshold.


## 2026-10-23 — Skala data M1: tahap 2a (positif penuh + negatif 3x, seed 42); fold 17-19 disiapkan terpisah

**Keputusan (usulan Orang 2; menunggu konfirmasi Orang 1 dan Orang 3):**
1. Training awal M1 (tahap 2a) memakai SEMUA positif fold 0-16 (4.134) + 3x negatif (12.402) acak, seed 42, total 16.536 rekaman unik. Sampel tahap 1 (1.000 rekaman) termuat di dalamnya. Komposisi setelah pembagian brief 7.2: training 14.835 rekaman (3.683 positif, 24,8%) dan validasi `internal_val_0_16 == 1` 1.701 rekaman (451 positif, 26,5%).
2. Negatif tambahan hanya bertahap, berdasarkan kebutuhan dan hasil validasi internal (fold 0-16), tidak berdasarkan fold 17-19.
3. Unduhan dan QC fold 17-19 (22.269 rekaman, 688 positif) disiapkan terpisah dengan aturan QC yang tetap. Tidak ada penyetelan threshold, clipping, atau preprocessing dari fold evaluasi. Fold 19 tetap disegel sampai evaluasi akhir (brief 11.5).

**Alasan:** laju unduh terukur sekitar 13,3 rekaman/menit (80 rekaman dalam 360 detik, satu jendela pengukuran); seluruh fold 0-16 yang belum di cache (125.639 rekaman) kira-kira 6,5 hari nonstop, 2a kira-kira 0,8 hari, fold 17-19 kira-kira 1,2 hari. Ekstrapolasi kasar dengan laju konstan, bukan jaminan.

**Perbedaan dari PROJECT_BRIEF (dicatat, bukan disembunyikan):**
- Brief 10.1 mengasumsikan seluruh 149 ribu EKG diunduh; 2a hanya sebagian fold 0-16.
- Brief 10.4 menyebut `pos_weight` sekitar 30 (prevalensi 3,2%). Dengan negatif disubsampel 3:1, `pos_weight = n_neg/n_pos` pada data training sekitar 3. Tujuannya sama (menyeimbangkan kelas), angkanya berbeda.
- Validasi internal ikut terdampak subsampling (sekitar 26% positif), sehingga AUPRC validasi tidak sebanding dengan prevalensi asli. Dipakai hanya untuk early stopping. Threshold dan kalibrasi tetap di fold 18 (prevalensi asli), sesuai brief.
- Penyimpanan praproses ke memmap/HDF5 (brief 10.3 langkah 7) belum dibuat; ECGDataset memproses dari cache WFDB ke RAM per run.

**Verifikasi pembagian (agregat, tabel master draft):** validasi internal 12.822 dari 126.734 rekaman fold 0-16 (10,1%); 0 pada fold >= 17; 0 pasien muncul di training dan validasi sekaligus.

**Verifikasi pipeline tahap 1 (999 rekaman lolos QC dari 1.000; 1 download_failed karena HTTP 500):** loss training turun; checkpoint dimuat ulang menghasilkan prediksi identik. Kontrol label diacak (3 seed, n_val=99 dengan 29 positif): AUROC validasi 0,466 / 0,458 / 0,570 versus label asli 0,664 / 0,701 / 0,686. Ini hanya uji pipeline pada subset diperkaya positif dan validasi kecil, BUKAN klaim performa model.

**Keterbatasan bukti:** laju unduh dari satu jendela pengukuran (kemungkinan ada unduhan lain yang berjalan bersamaan); subset tahap 1 berproporsi 30% positif; 1 dari 80 rekaman pada pengukuran laju gagal QC (alasan tidak disimpan).


## 2026-10-23 — Klarifikasi tahap 2a: urutan sampling vs pembagian, pos_weight, dan fungsi signal_qc_stage2a.csv

Mengoreksi/melengkapi entri "Skala data M1: tahap 2a" di atas. Tidak ada keputusan baru; mengikuti PROJECT_BRIEF.

1. **Urutan sampling dan pembagian.** Sampel 2a (semua positif + 3x negatif, seed 42) diambil dari seluruh pool fold 0-16 SEBELUM pembagian train/validasi, lalu dibagi dengan flag `internal_val_0_16` dari tabel master (berbasis pasien). Sampling TIDAK dilakukan terpisah di dalam tiap partisi. Hasil: training 14.835 rekaman (3.683 positif, 11.152 negatif, 3,03:1) dan validasi 1.701 rekaman (451 positif, 1.250 negatif, 2,77:1). Seluruh positif ikut, jadi semua 451 positif validasi ada. Tidak ada kesepakatan tim untuk sampling per partisi.
2. **pos_weight.** `train_m1` memakai `pos_weight = n_neg/n_pos` dari data training setelah sampling (11.152/3.683 = 3,03). Angka sekitar 30 di brief 10.4 adalah rasio pada data penuh (110.229 negatif / 3.683 positif = 29,9). Negatif yang diambil 10,1% dari 110.229, sehingga tiap negatif sampel mewakili sekitar 9,9 negatif; 29,9 / 9,9 = 3,03. Jadi 3,03 pada sampel setara secara efektif dengan sekitar 30 pada data penuh (bobot relatif positif terhadap negatif sama; hanya skala total yang berbeda). Memakai 30 pada sampel akan memberi bobot positif sekitar 10x terlalu besar. Probabilitas keluaran tetap tidak terkalibrasi ke prevalensi asli; kalibrasi di fold 18 (Orang 3). Transformasi monoton tidak mengubah AUROC/AUPRC.
3. **Validasi diperkaya.** Validasi internal berisi sekitar 26,5% positif; dipakai hanya untuk early stopping, bukan untuk melaporkan AUPRC. Validasi berprevalensi asli opsional (sekitar 11.100 rekaman tambahan, sekitar 17-18 jam unduhan) dan hanya dilakukan jika tim menginginkannya.
4. **signal_qc_stage2a.csv hanya untuk subset 2a** (16.536 baris), dibaca oleh `train_m1`. Bukan `signal_qc.csv` dan tidak dipakai `build_master`. Kontrak QC seluruh master dibuat nanti dengan `build_contract` atas semua `ed_stay_id` master, setelah tim memutuskan perlakuan baris tanpa sinyal di fold 0-16 (rekaman yang tidak diproses berstatus `not_processed` dan tidak dihitung lolos). Jangan menyalin berkas stage menjadi `signal_qc.csv`.


## 2026-10-29 — Hasil M1 (tahap 2a), QC fold 17-18, dan serah terima prediksi ke Orang 3

**Keputusan / status (Orang 2):**
1. M1 (ResNet-1D, dari nol) dilatih pada tahap 2a sesuai entri 2026-10-23 dan PROJECT_BRIEF 7.2 dan 10.4: seed 42, CPU, `train = fold <= 16 & internal_val_0_16 == 0`, `val = internal_val_0_16 == 1`, `pos_weight = n_neg/n_pos` = 3,03, early stopping pada AUPRC validasi (patience 5), AdamW lr 1e-3. Training 14.816 rekaman (3.680 positif), validasi 1.700 rekaman (450 positif); turun dari rencana 14.835/1.701 karena 20 rekaman tidak lolos (lihat QC di bawah). Memuat dan praproses 788 detik.
2. Checkpoint terbaik epoch 12: AUROC validasi 0,855 dan AUPRC validasi 0,715. Early stopping berhenti di epoch 17. Setelah epoch 12 loss training terus turun sementara metrik validasi memburuk (overfitting); checkpoint yang dipakai tetap epoch 12. Validasi berisi sekitar 26,5% positif, jadi AUPRC ini BUKAN performa pada prevalensi asli (3,2%) dan tidak boleh dibandingkan langsung dengan hasil akhir.
3. Prediksi M1 untuk fold 17 dan 18 ditulis ke `predictions/M1_fold17.parquet` (7.512 baris, 210 positif) dan `predictions/M1_fold18.parquet` (7.520 baris, 218 positif) dengan kontrak PROJECT_BRIEF 12.1. `y_prob` belum terkalibrasi; kalibrasi dan threshold di fold 18 dikerjakan Orang 3. Orang 2 tidak menghitung metrik di fold 17/18. Fold 19 belum disentuh (tidak diunduh, tidak diprediksi) dan tetap dikunci sampai rencana analisis disepakati (brief 11.5).
4. Paket serah terima untuk Orang 3 (kode, checkpoint, statistik standardisasi, catatan) dan paket prediksi dibuat terpisah di luar repo dan dikirim lewat jalur privat tim, bukan lewat Git.

**QC sinyal (aturan brief 10.2, tanpa perubahan aturan):**
- Tahap 2a (16.536 rekaman, fold 0-16): `pass` 16.516, `qc_fail` 19 (11 karena >=3 lead datar, 8 karena lead dengan >20% NaN), `download_failed` 1 (TimeoutError, fold 13, kegagalan jaringan sesaat). Lolos per label: negatif 12.386 dari 12.402, positif 4.130 dari 4.134. Kegagalan tersebar 0-3 per fold. `amplitude_flag` (>20 mV, informatif) pada 35 rekaman; clipping +-3 mV dan tidak adanya hard fail 8 mV tidak diubah.
- Fold 17 dan 18 (15.042 rekaman): `pass` 15.032 (fold 17: 7.512, fold 18: 7.520), `qc_fail` 9 (7 lead NaN, 2 lead datar; fold 17: 5, fold 18: 4), `download_failed` 1 (HTTPError, fold 17). Positif 428 dari 428 lolos; negatif 14.604 dari 14.614. `amplitude_flag` pada 46 rekaman. Aturan QC tidak memakai label; hasil QC tidak dipakai untuk menyetel apa pun.

**Koreksi perkiraan:** entri dan pesan sebelumnya memperkirakan fold 17-18 sekitar 21 ribu rekaman. Angka sebenarnya 15.042. Dengan total fold 17-19 sebesar 22.269 rekaman dan 688 positif, fold 19 sekitar 7.227 rekaman dan 260 positif (turunan aritmetika, belum dihitung dari tabel master).

**Perbedaan dari PROJECT_BRIEF (dicatat, bukan disembunyikan):**
- M1 dilatih pada subset fold 0-16 (positif penuh + negatif 3x), bukan seluruh data (brief 10.4). `pos_weight` 3,03 setara efektif dengan sekitar 30 pada data penuh (lihat klarifikasi 2026-10-23).
- Validasi internal diperkaya positif (sekitar 26,5%); dipakai hanya untuk early stopping.
- Training di CPU (torch +cpu); GPU belum dipakai. Penyimpanan praproses ke memmap/HDF5 (brief 10.3 langkah 7) belum dibuat.
- `signal_qc.csv` penuh (149.003 `ed_stay_id`) dan `master_table_v1` BELUM ada. Berkas `signal_qc_stage2a.csv` dan `signal_qc_eval.csv` hanya subset dan tidak dipakai `build_master`.

**Terbuka / butuh keputusan tim (tidak diputuskan sepihak oleh Orang 2):**
- D1: jalur unduhan untuk sisa kohort. Laju terukur sekitar 5-10 rekaman/menit (batas PhysioNet sekitar 2,5 KB/detik per koneksi, maksimal 8 koneksi); server tidak mengirim `.dat` terkompresi (tanpa `Content-Encoding`). Bucket AWS publik `s3://physionet-open/mimic-iv-ecg/` ternyata kosong; halaman dataset tidak menyebut bucket GCS. Opsi yang masih ada: uji kecepatan ZIP resmi 33,8 GB, kemungkinan akses cloud lewat pengaturan PhysioNet, atau penyesuaian lingkup. Tidak ada anggota tim yang memiliki salinan sinyal.
- D2: perlakuan rekaman yang benar-benar tidak tersedia di server (saat ini 1 rekaman di fold 17) apakah dikeluarkan dari v1 sebagai "sinyal tidak tersedia".
- D3: toleransi pembuatan `master_table_v1` (hanya jika tidak ada ID tanpa hasil QC, atau batas yang disepakati).
- D4: pemilik dan lokasi kode M4 (XGBoost dilatih ulang pada v1), versi Python tim, dan penambahan `xgboost` ke requirements.
- D5: QC fold 19 dengan aturan yang sama (tanpa tuning) agar semua model dinilai pada kunjungan yang sama; prediksi tetap dikunci.

**Keterbatasan bukti:** AUROC/AUPRC di atas dari validasi internal yang kecil dan diperkaya positif, satu kali jalan (seed 42), tanpa interval kepercayaan. Perkiraan waktu unduhan adalah ekstrapolasi kasar dari laju yang berubah-ubah. Konfirmasi Orang 1 dan Orang 3 atas skala 2a (entri 2026-10-23) belum tercatat di log ini.