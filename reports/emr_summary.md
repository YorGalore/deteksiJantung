# Ringkasan Fitur EMR

Dihasilkan otomatis oleh `src/dataset/features_emr.py`. Hanya angka agregat.

- Jumlah kunjungan: 149,003

## Fitur numerik

| fitur | missing_% | median | dijadikan_NaN_(luar_rentang) |
|---|---|---|---|
| age | 0.0 | 63.0 | 0 |
| temperature | 7.65 | 98.0 | 622 |
| heartrate | 5.32 | 83.0 | 30 |
| resprate | 6.43 | 18.0 | 58 |
| o2sat | 6.43 | 99.0 | 139 |
| sbp | 5.61 | 135.0 | 207 |
| dbp | 5.92 | 76.0 | 593 |
| pain | 12.45 | 2.0 | 11,204 |
| acuity | 2.02 | 2.0 | 0 |
| n_meds | 18.53 | 9.0 | 0 |

## Flag (persentase kunjungan bernilai 1; flag obat dihitung dari kunjungan yang punya medrecon)

| flag | %_bernilai_1 | missing_% |
|---|---|---|
| cc_chest_pain | 18.11 | 0.0 |
| cc_dyspnea | 12.57 | 0.0 |
| cc_syncope | 3.47 | 0.0 |
| cc_weakness | 5.64 | 0.0 |
| cc_epigastric_abd | 9.51 | 0.0 |
| cc_palpitations | 2.58 | 0.0 |
| cc_dizziness | 5.76 | 0.0 |
| cc_nausea_vomit | 6.19 | 0.0 |
| cc_diaphoresis | 0.03 | 0.0 |
| cc_arm_jaw_pain | 0.93 | 0.0 |
| cc_back_pain | 2.1 | 0.0 |
| cc_explicit_mi | 0.42 | 0.0 |
| cc_missing | 0.0 | 0.0 |
| med_antiplatelet | 37.87 | 18.53 |
| med_anticoagulant | 9.81 | 18.53 |
| med_statin | 37.55 | 18.53 |
| med_betablocker | 37.59 | 18.53 |
| med_acei_arb | 32.56 | 18.53 |
| med_nitrate | 8.97 | 18.53 |
| med_insulin | 9.75 | 18.53 |
| med_oral_antidiabetic | 14.24 | 18.53 |
| med_loop_diuretic | 17.25 | 18.53 |
| med_ccb | 20.02 | 18.53 |
| medrecon_available | 81.47 | 0.0 |
| is_male | 46.01 | 0.0 |

## Cara kedatangan (kategori >= 10)

| arrival_transport | n |
|---|---|
| WALK IN | 74,021 |
| AMBULANCE | 69,095 |
| UNKNOWN | 5,152 |
| HELICOPTER | 404 |
| OTHER | 331 |

Catatan: `cc_explicit_mi`, `cc_missing`, dan `medrecon_available` adalah kolom analisis, bukan fitur model.