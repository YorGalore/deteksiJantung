# Ringkasan Fitur Machine Measurements

Dihasilkan otomatis oleh `src/dataset/features_mm.py`. Hanya angka agregat.

- Jumlah kunjungan: 149,003
- Kunjungan tanpa data MM sama sekali: 0
- p_wave_absent = 1: 9.49%
- qrs_axis tepat 0: 1.62% (jika jauh lebih tinggi dari nilai 1 atau -1, curigai sentinel)
- qrs_axis tepat 1: 0.81% | tepat -1: 0.38%

## Fitur (kohort)

| fitur | missing_% | p1 | median | p99 | luar_rentang_->NaN |
|---|---|---|---|---|---|
| rr_interval | 0.1 | 416.0 | 769.0 | 1304.0 | 203 |
| hr_ecg | 0.1 | 46.0 | 78.0 | 144.2 | 0 |
| pr_interval | 7.64 | 86.0 | 158.0 | 263.0 | 2,497 |
| qrs_duration | 0.12 | 69.0 | 93.0 | 172.0 | 413 |
| qt_interval | 0.11 | 284.0 | 388.0 | 521.0 | 202 |
| qtc_bazett | 0.12 | 377.2 | 441.5 | 550.5 | 46 |
| p_axis | 9.64 | -78.0 | 50.0 | 99.0 | 457 |
| qrs_axis | 0.54 | -91.0 | 18.0 | 110.0 | 1,105 |
| t_axis | 0.94 | -147.0 | 38.0 | 161.0 | 710 |
| p_wave_absent | 0.0 | 0.0 | 0.0 | 1.0 | 0 |

## Sentinel (>= 29999) di seluruh machine_measurements

| kolom_mentah | n_sentinel_(semua_EKG) |
|---|---|
| rr_interval | 1,341 |
| p_onset | 123,434 |
| p_end | 230,363 |
| qrs_onset | 1,132 |
| qrs_end | 1,339 |
| t_end | 1,344 |
| p_axis | 130,555 |
| qrs_axis | 2,459 |
| t_axis | 3,898 |

Nilai onset/end mentah tidak dijadikan fitur karena titik acuannya bergantung jenis mesin.