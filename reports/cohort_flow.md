# Alur Kohort

Dihasilkan otomatis oleh `src/dataset/cohort.py`. Hanya angka agregat.

| langkah | n_ekg | n_kunjungan_ed | n_pasien |
|---|---|---|---|
| 0. Semua EKG di Ext-ICD | 800,035 | 149,021 | 161,352 |
| 1. EKG diambil di IGD | 184,720 | 149,021 | 83,744 |
| 2. Punya ed_stay_id | 184,720 | 149,021 | 83,744 |
| 3. Punya diagnosis (all_diag_all) | 184,700 | 149,003 | 83,738 |
| 4. EKG pertama per kunjungan (ecg_time terawal) | 149,003 | 149,003 | 83,738 |

EKG pertama (berdasarkan ecg_time) yang juga bernilai `ecg_no_within_stay == 0`: 99.99%

## Kunjungan per fold

| fold | n_kunjungan |
|---|---|
| 0 | 7,427 |
| 1 | 7,466 |
| 2 | 7,508 |
| 3 | 7,112 |
| 4 | 7,437 |
| 5 | 7,502 |
| 6 | 7,338 |
| 7 | 7,474 |
| 8 | 7,472 |
| 9 | 7,472 |
| 10 | 7,410 |
| 11 | 7,369 |
| 12 | 7,646 |
| 13 | 7,534 |
| 14 | 7,523 |
| 15 | 7,793 |
| 16 | 7,251 |
| 17 | 7,518 |
| 18 | 7,524 |
| 19 | 7,227 |

Filter 5 (kualitas sinyal) ditambahkan setelah Orang 2 menyerahkan hasil cek sinyal.