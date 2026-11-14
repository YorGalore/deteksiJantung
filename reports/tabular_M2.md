# M2: XGBoost EMR

Dihasilkan otomatis oleh `src/model/train_tabular.py`. Hanya angka agregat.

- Tabel master: `master_table_draft.parquet` (sha256 6ec97f8b873f...) | filter sinyal: signal_ok == 1 dari signal_qc.csv (148,861 baris tersisa)
- Fitur: 33 kolom asal -> 37 kolom matriks (arrival_transport one-hot)
- Training: 113,804 kunjungan (3,680 positif) | validasi internal: 12,807 (450 positif) | scale_pos_weight = 29.93
- Terpilih: trial 9 | iterasi terbaik 177 | AUPRC validasi 0.1981 | AUROC validasi 0.8460
- Metrik validasi internal dipakai untuk memilih model, BUKAN hasil akhir (sedikit optimistis karena dipakai memilih).

## Random search

| trial | max_depth | learning_rate | min_child_weight | subsample | colsample_bytree | best_iteration | val_auprc | val_auroc |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.05 | 1 | 0.8 | 0.8 | 327 | 0.1975 | 0.8433 |
| 2 | 3 | 0.1 | 5 | 0.8 | 0.7 | 222 | 0.1964 | 0.8436 |
| 3 | 8 | 0.02 | 5 | 0.7 | 0.5 | 225 | 0.1776 | 0.8393 |
| 4 | 5 | 0.1 | 5 | 0.9 | 0.8 | 167 | 0.1890 | 0.8409 |
| 5 | 6 | 0.05 | 1 | 1.0 | 0.7 | 191 | 0.1918 | 0.8378 |
| 6 | 5 | 0.03 | 1 | 1.0 | 0.8 | 342 | 0.1953 | 0.8424 |
| 7 | 6 | 0.03 | 10 | 0.8 | 0.7 | 324 | 0.1924 | 0.8425 |
| 8 | 5 | 0.02 | 1 | 0.8 | 1.0 | 704 | 0.1898 | 0.8428 |
| 9 | 3 | 0.1 | 10 | 0.7 | 0.8 | 177 | 0.1981 | 0.8460 |
| 10 | 3 | 0.1 | 5 | 0.7 | 0.5 | 305 | 0.1950 | 0.8474 |
| 11 | 8 | 0.03 | 10 | 0.9 | 0.8 | 191 | 0.1855 | 0.8342 |
| 12 | 6 | 0.02 | 3 | 0.8 | 0.7 | 328 | 0.1874 | 0.8419 |
| 13 | 3 | 0.05 | 1 | 0.9 | 0.8 | 454 | 0.1962 | 0.8432 |
| 14 | 8 | 0.05 | 3 | 1.0 | 0.7 | 131 | 0.1779 | 0.8293 |
| 15 | 4 | 0.1 | 3 | 0.6 | 0.7 | 166 | 0.1924 | 0.8442 |
| 16 | 6 | 0.02 | 3 | 0.6 | 0.8 | 310 | 0.1871 | 0.8408 |
| 17 | 5 | 0.03 | 1 | 0.8 | 0.8 | 453 | 0.1910 | 0.8429 |
| 18 | 8 | 0.03 | 1 | 1.0 | 0.8 | 170 | 0.1801 | 0.8316 |
| 19 | 6 | 0.02 | 3 | 0.9 | 1.0 | 497 | 0.1875 | 0.8394 |
| 20 | 5 | 0.1 | 10 | 0.7 | 1.0 | 51 | 0.1942 | 0.8408 |
