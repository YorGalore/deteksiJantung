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
