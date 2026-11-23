# SHAP M2 di fold 18

Dihasilkan otomatis oleh `src/model/shap_tabular.py`. Hanya angka agregat; tidak ada nilai per pasien.

- Kunjungan: 7,520 (218 positif) | tabel master `master_table_draft.parquet` | filter sinyal: signal_ok == 1 dari signal_qc.csv (148,861 baris tersisa)
- TreeSHAP bawaan XGBoost (pred_contribs), iterasi 0-177; skala log-odds model mentah (sebelum Platt).
- Cek aditivitas: selisih maksimum 4.1e-06 | cek terhadap berkas prediksi: 7.7e-08
- Nilai dasar (rata-rata bias): +0.0487 log-odds
- Dibuat: 2026-11-23T22:57:53+00:00

Arah pengaruh = rata-rata SHAP pada kelompok kunjungan (biner: nilai 1 vs 0; kontinu: kuartil bawah Q1 vs kuartil atas Q4 di fold ini; kategori: per kategori). Positif = mendorong prediksi ke arah IMA. Kelompok < 10 kunjungan tidak ditampilkan.

| peringkat | fitur | rata-rata abs SHAP | porsi | rata-rata SHAP | arah pengaruh |
|---|---|---|---|---|---|
| 1 | Usia (`age`) | 0.6894 | 18.0% | -0.3906 | Q1 (<= 48): -1.783; Q4 (>= 75): +0.411 |
| 2 | Tingkat kegawatan (ESI) (`acuity`) | 0.4469 | 11.7% | -0.1663 | Q1 (<= 2): +0.237; Q4 (>= 3): -0.759 |
| 3 | Keluhan nyeri dada (`cc_chest_pain`) | 0.3951 | 10.3% | -0.0844 | =1: +0.822; =0: -0.296 |
| 4 | Cara kedatangan (`arrival_transport`) | 0.3808 | 9.9% | -0.1427 | AMBULANCE: +0.188; HELICOPTER: +0.390; OTHER: +0.337; UNKNOWN: +0.930; WALK IN: -0.515 |
| 5 | Obat antiplatelet (`med_antiplatelet`) | 0.2050 | 5.3% | -0.0576 | =1: +0.222; =0: -0.250 |
| 6 | Jenis kelamin laki-laki (`is_male`) | 0.1860 | 4.9% | -0.0302 | =1: +0.168; =0: -0.202 |
| 7 | Skala nyeri (`pain`) | 0.1769 | 4.6% | -0.0453 | Q1 (<= 0): +0.137; Q4 (>= 7): -0.380 |
| 8 | Obat penyekat beta (`med_betablocker`) | 0.1520 | 4.0% | -0.0408 | =1: +0.169; =0: -0.188 |
| 9 | Obat statin (`med_statin`) | 0.1316 | 3.4% | -0.0257 | =1: +0.170; =0: -0.141 |
| 10 | SpO2 (`o2sat`) | 0.1270 | 3.3% | -0.0429 | Q1 (<= 97): +0.083; Q4 (>= 100): -0.149 |
| 11 | Tekanan sistolik (`sbp`) | 0.1159 | 3.0% | -0.0296 | Q1 (<= 120): +0.075; Q4 (>= 151): -0.067 |
| 12 | Denyut nadi triase (`heartrate`) | 0.1050 | 2.7% | -0.0131 | Q1 (<= 72): -0.176; Q4 (>= 98): +0.056 |
| 13 | Suhu (`temperature`) | 0.0939 | 2.4% | -0.0262 | Q1 (<= 97.5): +0.056; Q4 (>= 98.5): -0.181 |
| 14 | Keluhan pusing (`cc_dizziness`) | 0.0881 | 2.3% | -0.0333 | =1: -1.028; =0: +0.029 |
| 15 | Tekanan diastolik (`dbp`) | 0.0823 | 2.1% | -0.0298 | Q1 (<= 66): +0.024; Q4 (>= 87): +0.033 |
| 16 | Jumlah obat rutin (`n_meds`) | 0.0814 | 2.1% | +0.0046 | Q1 (<= 4): +0.119; Q4 (>= 14): -0.161 |
| 17 | Obat nitrat (`med_nitrate`) | 0.0712 | 1.9% | -0.0224 | =1: +0.269; =0: -0.062 |
| 18 | Laju napas (`resprate`) | 0.0639 | 1.7% | -0.0168 | Q1 (<= 16): -0.044; Q4 (>= 18): -0.016 |
| 19 | Insulin (`med_insulin`) | 0.0546 | 1.4% | -0.0047 | =1: +0.251; =0: -0.040 |
| 20 | Diuretik loop (`med_loop_diuretic`) | 0.0382 | 1.0% | -0.0063 | =1: +0.094; =0: -0.032 |
| 21 | Keluhan sinkop (`cc_syncope`) | 0.0365 | 1.0% | -0.0099 | =1: -0.704; =0: +0.014 |
| 22 | Keluhan sesak napas (`cc_dyspnea`) | 0.0194 | 0.5% | -0.0034 | =1: +0.062; =0: -0.013 |
| 23 | Keluhan palpitasi (`cc_palpitations`) | 0.0191 | 0.5% | -0.0066 | =1: -0.504; =0: +0.006 |
| 24 | Antidiabetes oral (`med_oral_antidiabetic`) | 0.0184 | 0.5% | -0.0043 | =1: -0.063; =0: +0.008 |
| 25 | Obat antikoagulan (`med_anticoagulant`) | 0.0150 | 0.4% | -0.0016 | =1: -0.067; =0: +0.007 |
| 26 | Obat ACE-inhibitor/ARB (`med_acei_arb`) | 0.0141 | 0.4% | -0.0018 | =1: +0.014; =0: -0.007 |
| 27 | Keluhan nyeri epigastrium/perut (`cc_epigastric_abd`) | 0.0106 | 0.3% | -0.0051 | =1: -0.080; =0: +0.003 |
| 28 | Keluhan nyeri lengan/rahang (`cc_arm_jaw_pain`) | 0.0056 | 0.1% | -0.0006 | =1: +0.297; =0: -0.003 |
| 29 | Penghambat kanal kalsium (`med_ccb`) | 0.0033 | 0.1% | -0.0003 | =1: -0.006; =0: +0.002 |
| 30 | Keluhan mual/muntah (`cc_nausea_vomit`) | 0.0029 | 0.1% | -0.0002 | =1: +0.021; =0: -0.002 |
| 31 | Keluhan lemas (`cc_weakness`) | 0.0016 | 0.0% | -0.0008 | =1: -0.010; =0: -0.000 |
| 32 | Keluhan nyeri punggung (`cc_back_pain`) | 0.0011 | 0.0% | -0.0000 | =1: +0.019; =0: -0.000 |
| 33 | Keluhan diaforesis (`cc_diaphoresis`) | 0.0009 | 0.0% | -0.0005 | =1: –; =0: -0.001 |

Gambar: `reports/figures/shap_M2_fold18.png` (15 fitur teratas).
