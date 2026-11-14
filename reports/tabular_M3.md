# M3: XGBoost machine measurements

Dihasilkan otomatis oleh `src/model/train_tabular.py`. Hanya angka agregat.

- Tabel master: `master_table_draft.parquet` (sha256 6ec97f8b873f...) | filter sinyal: signal_ok == 1 dari signal_qc.csv (148,861 baris tersisa)
- Fitur: 10 kolom asal -> 10 kolom matriks (arrival_transport one-hot)
- Training: 113,804 kunjungan (3,680 positif) | validasi internal: 12,807 (450 positif) | scale_pos_weight = 29.93
- Terpilih: trial 1 | iterasi terbaik 150 | AUPRC validasi 0.1122 | AUROC validasi 0.7585
- Metrik validasi internal dipakai untuk memilih model, BUKAN hasil akhir (sedikit optimistis karena dipakai memilih).

## Random search

| trial | max_depth | learning_rate | min_child_weight | subsample | colsample_bytree | best_iteration | val_auprc | val_auroc |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.05 | 1 | 0.8 | 0.8 | 150 | 0.1122 | 0.7585 |
| 2 | 3 | 0.1 | 5 | 0.8 | 0.7 | 78 | 0.1103 | 0.7603 |
| 3 | 8 | 0.02 | 5 | 0.7 | 0.5 | 12 | 0.0974 | 0.7389 |
| 4 | 5 | 0.1 | 5 | 0.9 | 0.8 | 36 | 0.1078 | 0.7539 |
| 5 | 6 | 0.05 | 1 | 1.0 | 0.7 | 116 | 0.1050 | 0.7425 |
| 6 | 5 | 0.03 | 1 | 1.0 | 0.8 | 10 | 0.1019 | 0.7502 |
| 7 | 6 | 0.03 | 10 | 0.8 | 0.7 | 54 | 0.1065 | 0.7528 |
| 8 | 5 | 0.02 | 1 | 0.8 | 1.0 | 35 | 0.1040 | 0.7540 |
| 9 | 3 | 0.1 | 10 | 0.7 | 0.8 | 99 | 0.1118 | 0.7586 |
| 10 | 3 | 0.1 | 5 | 0.7 | 0.5 | 65 | 0.1110 | 0.7580 |
| 11 | 8 | 0.03 | 10 | 0.9 | 0.8 | 49 | 0.1042 | 0.7431 |
| 12 | 6 | 0.02 | 3 | 0.8 | 0.7 | 6 | 0.1056 | 0.7481 |
| 13 | 3 | 0.05 | 1 | 0.9 | 0.8 | 222 | 0.1101 | 0.7586 |
| 14 | 8 | 0.05 | 3 | 1.0 | 0.7 | 24 | 0.1015 | 0.7394 |
| 15 | 4 | 0.1 | 3 | 0.6 | 0.7 | 23 | 0.1088 | 0.7596 |
| 16 | 6 | 0.02 | 3 | 0.6 | 0.8 | 9 | 0.1047 | 0.7544 |
| 17 | 5 | 0.03 | 1 | 0.8 | 0.8 | 48 | 0.1054 | 0.7581 |
| 18 | 8 | 0.03 | 1 | 1.0 | 0.8 | 42 | 0.1029 | 0.7384 |
| 19 | 6 | 0.02 | 3 | 0.9 | 1.0 | 35 | 0.1019 | 0.7477 |
| 20 | 5 | 0.1 | 10 | 0.7 | 1.0 | 42 | 0.1013 | 0.7515 |
