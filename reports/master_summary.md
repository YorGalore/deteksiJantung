# Ringkasan Tabel Master

Dihasilkan otomatis oleh `src/dataset/build_master.py`. Hanya angka agregat.

- Status: **DRAFT (menunggu hasil cek sinyal Orang 2)**
- Jumlah kunjungan: 149,003 | pasien: 83,738
- Positif: 4,822 (3.24%)
- Jumlah fitur: 33 EMR + 10 MM
- Validasi internal: 7,114 pasien (fold 0-16), 7,534 pasien (fold 0-17)

## Per fold

| fold | n | positif | internal_val_0_16 | internal_val_0_17 | peran |
|---|---|---|---|---|---|
| 0 | 7,427 | 235 | 693 | 766 | training |
| 1 | 7,466 | 237 | 752 | 661 | training |
| 2 | 7,508 | 234 | 732 | 778 | training |
| 3 | 7,112 | 279 | 699 | 729 | training |
| 4 | 7,437 | 247 | 750 | 854 | training |
| 5 | 7,502 | 239 | 811 | 810 | training |
| 6 | 7,338 | 258 | 762 | 688 | training |
| 7 | 7,474 | 201 | 808 | 703 | training |
| 8 | 7,472 | 238 | 751 | 745 | training |
| 9 | 7,472 | 232 | 833 | 692 | training |
| 10 | 7,410 | 281 | 754 | 744 | training |
| 11 | 7,369 | 282 | 743 | 762 | training |
| 12 | 7,646 | 227 | 748 | 764 | training |
| 13 | 7,534 | 250 | 845 | 820 | training |
| 14 | 7,523 | 230 | 772 | 820 | training |
| 15 | 7,793 | 233 | 691 | 751 | training |
| 16 | 7,251 | 231 | 678 | 705 | training |
| 17 | 7,518 | 210 | 0 | 777 | meta-learner |
| 18 | 7,524 | 218 | 0 | 0 | threshold |
| 19 | 7,227 | 260 | 0 | 0 | TEST (dikunci) |

## Fitur dengan nilai kosong

| fitur | missing_% |
|---|---|
| med_antiplatelet | 18.53 |
| med_anticoagulant | 18.53 |
| med_statin | 18.53 |
| med_betablocker | 18.53 |
| med_oral_antidiabetic | 18.53 |
| n_meds | 18.53 |
| med_ccb | 18.53 |
| med_loop_diuretic | 18.53 |
| med_nitrate | 18.53 |
| med_insulin | 18.53 |
| med_acei_arb | 18.53 |
| pain | 12.45 |
| p_axis | 9.64 |
| temperature | 7.65 |
| pr_interval | 7.64 |
| resprate | 6.43 |
| o2sat | 6.43 |
| dbp | 5.92 |
| sbp | 5.61 |
| heartrate | 5.32 |
| acuity | 2.02 |
| t_axis | 0.94 |
| qrs_axis | 0.54 |
| qrs_duration | 0.12 |
| qtc_bazett | 0.12 |
| qt_interval | 0.11 |
| rr_interval | 0.1 |
| hr_ecg | 0.1 |

## Kamus kolom (pakai daftar di `src/constants.py`, jangan memilih kolom sendiri)

| kelompok | kolom |
|---|---|
| ID | ed_stay_id, subject_id, study_id, file_name |
| Split | fold, internal_val_0_16, internal_val_0_17 |
| Label | label_ami, label_source, ami_stemi, ami_nstemi, ami_unspecified, ami_type2, ami_other, ami_subsequent, ami_type2_only |
| Analisis (BUKAN fitur) | cc_explicit_mi, cc_missing, medrecon_available, mm_missing_count, first_ecg_matches_no0, signal_ok |
| Fitur EMR | age, is_male, arrival_transport, temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity, cc_chest_pain, cc_dyspnea, cc_syncope, cc_weakness, cc_epigastric_abd, cc_palpitations, cc_dizziness, cc_nausea_vomit, cc_diaphoresis, cc_arm_jaw_pain, cc_back_pain, med_antiplatelet, med_anticoagulant, med_statin, med_betablocker, med_acei_arb, med_nitrate, med_insulin, med_oral_antidiabetic, med_loop_diuretic, med_ccb, n_meds |
| Fitur MM | rr_interval, hr_ecg, pr_interval, qrs_duration, qt_interval, qtc_bazett, p_axis, qrs_axis, t_axis, p_wave_absent |
