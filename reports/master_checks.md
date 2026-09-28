# Pemeriksaan Tabel Master

Dihasilkan otomatis oleh `src/dataset/check_master.py` untuk `master_table_draft.parquet`. Hanya angka agregat.

| pemeriksaan | hasil | wajib | detail |
|---|---|---|---|
| ed_stay_id unik | LOLOS | ya |  |
| Setiap pasien hanya di satu fold | LOLOS | ya | 0 pasien melanggar |
| fold hanya 0-19 | LOLOS | ya |  |
| internal_val_0_16 tidak menyentuh fold >= 17 | LOLOS | ya | 0 baris |
| internal_val_0_16 berbasis pasien (tidak terbelah) | LOLOS | ya |  |
| internal_val_0_17 tidak menyentuh fold >= 18 | LOLOS | ya | 0 baris |
| internal_val_0_17 berbasis pasien (tidak terbelah) | LOLOS | ya |  |
| label_ami hanya 0/1 tanpa NaN | LOLOS | ya |  |
| Tidak ada kolom terlarang di daftar fitur | LOLOS | ya |  |
| Kolom analisis/label/split tidak ada di daftar fitur | LOLOS | ya |  |
| Tidak ada kolom teks bebas di tabel master | LOLOS | ya |  |
| Tidak ada teks keluhan/laporan mesin/onset mentah | LOLOS | ya |  |
| Tidak ada fitur tunggal dengan AUROC > 0.85 | LOLOS | tidak |  |

## AUROC satu fitur (fold 0-16), 15 teratas

AUROC mendekati 1 atau 0 pada satu fitur saja adalah tanda kebocoran label yang perlu diperiksa.

| fitur | auroc_fold0_16 | jarak_dari_0.5 |
|---|---|---|
| age | 0.665 | 0.165 |
| arrival_transport | 0.664 | 0.164 |
| acuity | 0.344 | 0.156 |
| qtc_bazett | 0.626 | 0.126 |
| qrs_duration | 0.625 | 0.125 |
| med_antiplatelet | 0.623 | 0.123 |
| med_betablocker | 0.607 | 0.107 |
| med_statin | 0.601 | 0.101 |
| o2sat | 0.402 | 0.098 |
| cc_chest_pain | 0.588 | 0.088 |
| t_axis | 0.585 | 0.085 |
| qrs_axis | 0.422 | 0.078 |
| n_meds | 0.577 | 0.077 |
| pain | 0.423 | 0.077 |
| is_male | 0.574 | 0.074 |
