# M4: XGBoost concat EMR + MM (Jalur B)

Dihasilkan otomatis oleh `src/model/train_tabular.py`. Hanya angka agregat.

- Tabel master: `master_table_draft.parquet` (sha256 6ec97f8b873f...) | filter sinyal: signal_ok == 1 dari signal_qc.csv (148,861 baris tersisa)
- Fitur: 43 kolom asal -> 47 kolom matriks (arrival_transport one-hot)
- Training: 120,572 kunjungan (3,879 positif) | validasi internal: 13,552 (461 positif) | scale_pos_weight = 30.08
- Terpilih: trial 9 | iterasi terbaik 84 | AUPRC validasi 0.2236 | AUROC validasi 0.8485
- Metrik validasi internal dipakai untuk memilih model, BUKAN hasil akhir (sedikit optimistis karena dipakai memilih).

## Random search

| trial | max_depth | learning_rate | min_child_weight | subsample | colsample_bytree | best_iteration | val_auprc | val_auroc |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.05 | 1 | 0.8 | 0.8 | 302 | 0.2229 | 0.8504 |
| 2 | 3 | 0.1 | 5 | 0.8 | 0.7 | 153 | 0.2227 | 0.8518 |
| 3 | 8 | 0.02 | 5 | 0.7 | 0.5 | 275 | 0.2074 | 0.8449 |
| 4 | 5 | 0.1 | 5 | 0.9 | 0.8 | 69 | 0.2206 | 0.8494 |
| 5 | 6 | 0.05 | 1 | 1.0 | 0.7 | 137 | 0.2218 | 0.8483 |
| 6 | 5 | 0.03 | 1 | 1.0 | 0.8 | 275 | 0.2221 | 0.8480 |
| 7 | 6 | 0.03 | 10 | 0.8 | 0.7 | 235 | 0.2221 | 0.8502 |
| 8 | 5 | 0.02 | 1 | 0.8 | 1.0 | 388 | 0.2167 | 0.8509 |
| 9 | 3 | 0.1 | 10 | 0.7 | 0.8 | 84 | 0.2236 | 0.8485 |
| 10 | 3 | 0.1 | 5 | 0.7 | 0.5 | 129 | 0.2205 | 0.8513 |
| 11 | 8 | 0.03 | 10 | 0.9 | 0.8 | 120 | 0.2062 | 0.8436 |
| 12 | 6 | 0.02 | 3 | 0.8 | 0.7 | 541 | 0.2213 | 0.8491 |
| 13 | 3 | 0.05 | 1 | 0.9 | 0.8 | 143 | 0.2184 | 0.8498 |
| 14 | 8 | 0.05 | 3 | 1.0 | 0.7 | 86 | 0.1997 | 0.8407 |
| 15 | 4 | 0.1 | 3 | 0.6 | 0.7 | 115 | 0.2175 | 0.8480 |
| 16 | 6 | 0.02 | 3 | 0.6 | 0.8 | 163 | 0.2178 | 0.8492 |
| 17 | 5 | 0.03 | 1 | 0.8 | 0.8 | 293 | 0.2236 | 0.8500 |
| 18 | 8 | 0.03 | 1 | 1.0 | 0.8 | 165 | 0.2057 | 0.8418 |
| 19 | 6 | 0.02 | 3 | 0.9 | 1.0 | 347 | 0.2180 | 0.8490 |
| 20 | 5 | 0.1 | 10 | 0.7 | 1.0 | 87 | 0.2097 | 0.8448 |
