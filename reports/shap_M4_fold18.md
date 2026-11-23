# SHAP M4 di fold 18

Dihasilkan otomatis oleh `src/model/shap_tabular.py`. Hanya angka agregat; tidak ada nilai per pasien.

- Kunjungan: 7,520 (218 positif) | tabel master `master_table_draft.parquet` | filter sinyal: signal_ok == 1 dari signal_qc.csv (148,861 baris tersisa)
- TreeSHAP bawaan XGBoost (pred_contribs), iterasi 0-84; skala log-odds model mentah (sebelum Platt).
- Cek aditivitas: selisih maksimum 1.8e-06 | cek terhadap berkas prediksi: 7.6e-08
- Nilai dasar (rata-rata bias): +0.0193 log-odds
- Dibuat: 2026-11-23T22:57:56+00:00

Arah pengaruh = rata-rata SHAP pada kelompok kunjungan (biner: nilai 1 vs 0; kontinu: kuartil bawah Q1 vs kuartil atas Q4 di fold ini; kategori: per kategori). Positif = mendorong prediksi ke arah IMA. Kelompok < 10 kunjungan tidak ditampilkan.

| peringkat | fitur | rata-rata abs SHAP | porsi | rata-rata SHAP | arah pengaruh |
|---|---|---|---|---|---|
| 1 | Usia (`age`) | 0.4847 | 13.8% | -0.2580 | Q1 (<= 48): -1.278; Q4 (>= 75): +0.279 |
| 2 | Sumbu T (`t_axis`) | 0.4375 | 12.5% | -0.1879 | Q1 (<= 15.5): -0.097; Q4 (>= 59): +0.226 |
| 3 | Tingkat kegawatan (ESI) (`acuity`) | 0.4077 | 11.6% | -0.1418 | Q1 (<= 2): +0.221; Q4 (>= 3): -0.680 |
| 4 | Keluhan nyeri dada (`cc_chest_pain`) | 0.4032 | 11.5% | -0.0908 | =1: +0.827; =0: -0.305 |
| 5 | Cara kedatangan (`arrival_transport`) | 0.3704 | 10.6% | -0.1410 | AMBULANCE: +0.201; HELICOPTER: +0.268; OTHER: +0.251; UNKNOWN: +0.661; WALK IN: -0.503 |
| 6 | Jenis kelamin laki-laki (`is_male`) | 0.1551 | 4.4% | -0.0289 | =1: +0.136; =0: -0.172 |
| 7 | Obat antiplatelet (`med_antiplatelet`) | 0.1507 | 4.3% | -0.0390 | =1: +0.163; =0: -0.184 |
| 8 | QTc Bazett (`qtc_bazett`) | 0.1258 | 3.6% | -0.0452 | Q1 (<= 421.7): -0.149; Q4 (>= 463.2): +0.160 |
| 9 | Skala nyeri (`pain`) | 0.1085 | 3.1% | -0.0288 | Q1 (<= 0): +0.075; Q4 (>= 7): -0.252 |
| 10 | Durasi QRS (`qrs_duration`) | 0.0948 | 2.7% | -0.0318 | Q1 (<= 84): -0.171; Q4 (>= 103): +0.090 |
| 11 | SpO2 (`o2sat`) | 0.0754 | 2.2% | -0.0258 | Q1 (<= 97): +0.045; Q4 (>= 100): -0.078 |
| 12 | Interval RR (`rr_interval`) | 0.0725 | 2.1% | -0.0126 | Q1 (<= 645): +0.097; Q4 (>= 895): -0.114 |
| 13 | Obat statin (`med_statin`) | 0.0636 | 1.8% | -0.0134 | =1: +0.076; =0: -0.074 |
| 14 | Obat nitrat (`med_nitrate`) | 0.0589 | 1.7% | -0.0168 | =1: +0.159; =0: -0.050 |
| 15 | Obat penyekat beta (`med_betablocker`) | 0.0518 | 1.5% | -0.0113 | =1: +0.057; =0: -0.062 |
| 16 | Keluhan pusing (`cc_dizziness`) | 0.0485 | 1.4% | -0.0167 | =1: -0.552; =0: +0.017 |
| 17 | Tekanan sistolik (`sbp`) | 0.0467 | 1.3% | -0.0095 | Q1 (<= 120): +0.035; Q4 (>= 151): -0.033 |
| 18 | Tekanan diastolik (`dbp`) | 0.0453 | 1.3% | -0.0172 | Q1 (<= 66): +0.017; Q4 (>= 87): -0.025 |
| 19 | Suhu (`temperature`) | 0.0424 | 1.2% | -0.0154 | Q1 (<= 97.5): +0.005; Q4 (>= 98.5): -0.060 |
| 20 | Jumlah obat rutin (`n_meds`) | 0.0386 | 1.1% | +0.0002 | Q1 (<= 4): +0.040; Q4 (>= 14): -0.078 |
| 21 | Laju napas (`resprate`) | 0.0341 | 1.0% | -0.0124 | Q1 (<= 16): -0.031; Q4 (>= 18): -0.014 |
| 22 | Interval PR (`pr_interval`) | 0.0301 | 0.9% | -0.0042 | Q1 (<= 141): +0.000; Q4 (>= 177): +0.037 |
| 23 | Sumbu QRS (`qrs_axis`) | 0.0286 | 0.8% | -0.0071 | Q1 (<= -10): +0.039; Q4 (>= 50): -0.022 |
| 24 | Keluhan sinkop (`cc_syncope`) | 0.0254 | 0.7% | -0.0070 | =1: -0.491; =0: +0.010 |
| 25 | Denyut nadi triase (`heartrate`) | 0.0244 | 0.7% | -0.0068 | Q1 (<= 72): -0.010; Q4 (>= 98): -0.043 |
| 26 | Sumbu P (`p_axis`) | 0.0235 | 0.7% | +0.0026 | Q1 (<= 30): +0.004; Q4 (>= 63): +0.010 |
| 27 | Insulin (`med_insulin`) | 0.0176 | 0.5% | -0.0019 | =1: +0.083; =0: -0.012 |
| 28 | Denyut jantung EKG (`hr_ecg`) | 0.0111 | 0.3% | -0.0023 | Q1 (<= 67.04): -0.015; Q4 (>= 93.02): +0.017 |
| 29 | Interval QT (`qt_interval`) | 0.0059 | 0.2% | -0.0001 | Q1 (<= 358): +0.004; Q4 (>= 418): -0.009 |
| 30 | Obat ACE-inhibitor/ARB (`med_acei_arb`) | 0.0048 | 0.1% | +0.0006 | =1: +0.010; =0: -0.003 |
| 31 | Keluhan sesak napas (`cc_dyspnea`) | 0.0046 | 0.1% | -0.0010 | =1: -0.022; =0: +0.002 |
| 32 | Gelombang P tidak terdeteksi (`p_wave_absent`) | 0.0043 | 0.1% | +0.0006 | =1: -0.019; =0: +0.003 |
| 33 | Antidiabetes oral (`med_oral_antidiabetic`) | 0.0041 | 0.1% | -0.0004 | =1: +0.002; =0: +0.001 |
| 34 | Obat antikoagulan (`med_anticoagulant`) | 0.0019 | 0.1% | +0.0001 | =1: -0.010; =0: +0.001 |
| 35 | Keluhan palpitasi (`cc_palpitations`) | 0.0017 | 0.0% | -0.0007 | =1: -0.049; =0: +0.001 |
| 36 | Keluhan nyeri lengan/rahang (`cc_arm_jaw_pain`) | 0.0016 | 0.0% | -0.0001 | =1: +0.087; =0: -0.001 |
| 37 | Penghambat kanal kalsium (`med_ccb`) | 0.0000 | 0.0% | +0.0000 | =1: +0.000; =0: +0.000 |
| 38 | Diuretik loop (`med_loop_diuretic`) | 0.0000 | 0.0% | +0.0000 | =1: +0.000; =0: +0.000 |
| 39 | Keluhan diaforesis (`cc_diaphoresis`) | 0.0000 | 0.0% | +0.0000 | =1: –; =0: +0.000 |
| 40 | Keluhan mual/muntah (`cc_nausea_vomit`) | 0.0000 | 0.0% | +0.0000 | =1: +0.000; =0: +0.000 |
| 41 | Keluhan nyeri epigastrium/perut (`cc_epigastric_abd`) | 0.0000 | 0.0% | +0.0000 | =1: +0.000; =0: +0.000 |
| 42 | Keluhan lemas (`cc_weakness`) | 0.0000 | 0.0% | +0.0000 | =1: +0.000; =0: +0.000 |
| 43 | Keluhan nyeri punggung (`cc_back_pain`) | 0.0000 | 0.0% | +0.0000 | =1: +0.000; =0: +0.000 |

Gambar: `reports/figures/shap_M4_fold18.png` (15 fitur teratas).
