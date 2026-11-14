# Evaluasi pengembangan di fold 18

Dihasilkan otomatis oleh `src/evaluation/run_fold18.py`. Hanya angka agregat.

> Kalibrasi dan threshold dipilih DAN dinilai di fold 18. Metrik bergantung threshold di sini bersifat in-sample. Hasil akhir hanya dari fold 19 (sekali jalan) dengan parameter di `reports/locked_fold18.json`.

## M1

- Kunjungan: 7,520 | positif: 218 | prevalensi: 2.90% (garis dasar AUPRC = prevalensi)
- AUROC: 0.846 [0.822, 0.870] | AUPRC: 0.212 [0.160, 0.274]  (CI 95% bootstrap per subject_id)
- Probabilitas mentah: Brier 0.154, intercept kalibrasi -3.656, slope 0.727 (ideal 0 dan 1)
- Platt (fit di fold 18): a = -3.3995, b = 0.7273 | Brier setelah kalibrasi 0.025 (in-sample)

**Threshold (dipilih di fold 18; metrik di bawah in-sample, bukan hasil akhir)**

| threshold | y_prob_raw | sens | spes | PPV | NPV | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| Rule-out (sens >= 0,90) | 0.302827 | 0.904 | 0.613 | 0.065 | 0.995 | 0.122 | 197 | 2,827 | 21 | 4,475 |
| Rule-in (spes >= 0,90) | 0.711488 | 0.528 | 0.900 | 0.136 | 0.985 | 0.216 | 115 | 730 | 103 | 6,572 |

CI bootstrap @rule-out: sens 0.904 [0.863, 0.940], spes 0.613 [0.594, 0.633], NPV 0.995 [0.993, 0.997], PPV 0.065 [0.055, 0.076]

**Tabel kalibrasi setelah Platt (10 bin berisi jumlah kunjungan sama)**

| bin | n | mean_pred | observed |
|---|---|---|---|
| 1 | 752 | 0.0017 | 0.0013 |
| 2 | 752 | 0.0033 | 0.0013 |
| 3 | 752 | 0.0051 | 0.0027 |
| 4 | 752 | 0.0075 | 0.0053 |
| 5 | 752 | 0.0108 | 0.004 |
| 6 | 752 | 0.0153 | 0.0146 |
| 7 | 752 | 0.022 | 0.0226 |
| 8 | 752 | 0.0324 | 0.0399 |
| 9 | 752 | 0.0508 | 0.0505 |
| 10 | 752 | 0.1409 | 0.1476 |

## M2

- Kunjungan: 7,520 | positif: 218 | prevalensi: 2.90% (garis dasar AUPRC = prevalensi)
- AUROC: 0.831 [0.806, 0.857] | AUPRC: 0.144 [0.112, 0.192]  (CI 95% bootstrap per subject_id)
- Probabilitas mentah: Brier 0.165, intercept kalibrasi -3.406, slope 0.991 (ideal 0 dan 1)
- Platt (fit di fold 18): a = -3.4003, b = 0.9905 | Brier setelah kalibrasi 0.026 (in-sample)

**Threshold (dipilih di fold 18; metrik di bawah in-sample, bukan hasil akhir)**

| threshold | y_prob_raw | sens | spes | PPV | NPV | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| Rule-out (sens >= 0,90) | 0.316234 | 0.904 | 0.559 | 0.058 | 0.995 | 0.108 | 197 | 3,223 | 21 | 4,079 |
| Rule-in (spes >= 0,90) | 0.701304 | 0.482 | 0.900 | 0.126 | 0.983 | 0.199 | 105 | 730 | 113 | 6,572 |

CI bootstrap @rule-out: sens 0.904 [0.864, 0.942], spes 0.559 [0.542, 0.577], NPV 0.995 [0.993, 0.997], PPV 0.058 [0.049, 0.067]

**Tabel kalibrasi setelah Platt (10 bin berisi jumlah kunjungan sama)**

| bin | n | mean_pred | observed |
|---|---|---|---|
| 1 | 752 | 0.0004 | 0.0013 |
| 2 | 752 | 0.0014 | 0.0 |
| 3 | 752 | 0.0034 | 0.004 |
| 4 | 752 | 0.0064 | 0.0066 |
| 5 | 752 | 0.0103 | 0.0066 |
| 6 | 752 | 0.0158 | 0.0133 |
| 7 | 752 | 0.0241 | 0.0306 |
| 8 | 752 | 0.0368 | 0.0332 |
| 9 | 752 | 0.0591 | 0.0638 |
| 10 | 752 | 0.1323 | 0.1303 |

## M3

- Kunjungan: 7,520 | positif: 218 | prevalensi: 2.90% (garis dasar AUPRC = prevalensi)
- AUROC: 0.761 [0.730, 0.791] | AUPRC: 0.085 [0.067, 0.109]  (CI 95% bootstrap per subject_id)
- Probabilitas mentah: Brier 0.183, intercept kalibrasi -3.388, slope 1.019 (ideal 0 dan 1)
- Platt (fit di fold 18): a = -3.3934, b = 1.0193 | Brier setelah kalibrasi 0.027 (in-sample)

**Threshold (dipilih di fold 18; metrik di bawah in-sample, bukan hasil akhir)**

| threshold | y_prob_raw | sens | spes | PPV | NPV | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| Rule-out (sens >= 0,90) | 0.311988 | 0.904 | 0.454 | 0.047 | 0.994 | 0.090 | 197 | 3,985 | 21 | 3,317 |
| Rule-in (spes >= 0,90) | 0.696361 | 0.353 | 0.900 | 0.095 | 0.979 | 0.150 | 77 | 730 | 141 | 6,572 |

CI bootstrap @rule-out: sens 0.904 [0.862, 0.941], spes 0.454 [0.435, 0.474], NPV 0.994 [0.991, 0.996], PPV 0.047 [0.040, 0.055]

**Tabel kalibrasi setelah Platt (10 bin berisi jumlah kunjungan sama)**

| bin | n | mean_pred | observed |
|---|---|---|---|
| 1 | 752 | 0.0056 | 0.0053 |
| 2 | 752 | 0.0076 | 0.0013 |
| 3 | 752 | 0.0096 | 0.008 |
| 4 | 752 | 0.012 | 0.012 |
| 5 | 752 | 0.015 | 0.0146 |
| 6 | 752 | 0.0189 | 0.0213 |
| 7 | 752 | 0.0249 | 0.0399 |
| 8 | 752 | 0.036 | 0.0293 |
| 9 | 752 | 0.058 | 0.0598 |
| 10 | 752 | 0.1022 | 0.0984 |

## M4

- Kunjungan: 7,520 | positif: 218 | prevalensi: 2.90% (garis dasar AUPRC = prevalensi)
- AUROC: 0.855 [0.831, 0.879] | AUPRC: 0.180 [0.139, 0.237]  (CI 95% bootstrap per subject_id)
- Probabilitas mentah: Brier 0.153, intercept kalibrasi -3.327, slope 1.142 (ideal 0 dan 1)
- Platt (fit di fold 18): a = -3.4152, b = 1.1417 | Brier setelah kalibrasi 0.026 (in-sample)

**Threshold (dipilih di fold 18; metrik di bawah in-sample, bukan hasil akhir)**

| threshold | y_prob_raw | sens | spes | PPV | NPV | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| Rule-out (sens >= 0,90) | 0.314095 | 0.904 | 0.581 | 0.061 | 0.995 | 0.114 | 197 | 3,056 | 21 | 4,246 |
| Rule-in (spes >= 0,90) | 0.681518 | 0.550 | 0.900 | 0.141 | 0.985 | 0.225 | 120 | 730 | 98 | 6,572 |

CI bootstrap @rule-out: sens 0.904 [0.859, 0.943], spes 0.581 [0.564, 0.600], NPV 0.995 [0.993, 0.997], PPV 0.061 [0.051, 0.071]

**Tabel kalibrasi setelah Platt (10 bin berisi jumlah kunjungan sama)**

| bin | n | mean_pred | observed |
|---|---|---|---|
| 1 | 752 | 0.0004 | 0.0013 |
| 2 | 752 | 0.0012 | 0.0 |
| 3 | 752 | 0.0026 | 0.0013 |
| 4 | 752 | 0.0046 | 0.0066 |
| 5 | 752 | 0.0075 | 0.0053 |
| 6 | 752 | 0.0124 | 0.0186 |
| 7 | 752 | 0.0201 | 0.0146 |
| 8 | 752 | 0.0329 | 0.0359 |
| 9 | 752 | 0.0581 | 0.0505 |
| 10 | 752 | 0.1502 | 0.1556 |
