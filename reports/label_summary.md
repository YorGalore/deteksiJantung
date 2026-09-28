# Ringkasan Label

Dihasilkan otomatis oleh `src/dataset/labels.py`. Hanya angka agregat.

- Total kunjungan: 149,003
- Positif (I21/I22): 4,822 (3.24%)

## Per fold

| fold | n | positif | prevalensi_% |
|---|---|---|---|
| 0 | 7,427 | 235 | 3.16 |
| 1 | 7,466 | 237 | 3.17 |
| 2 | 7,508 | 234 | 3.12 |
| 3 | 7,112 | 279 | 3.92 |
| 4 | 7,437 | 247 | 3.32 |
| 5 | 7,502 | 239 | 3.19 |
| 6 | 7,338 | 258 | 3.52 |
| 7 | 7,474 | 201 | 2.69 |
| 8 | 7,472 | 238 | 3.19 |
| 9 | 7,472 | 232 | 3.1 |
| 10 | 7,410 | 281 | 3.79 |
| 11 | 7,369 | 282 | 3.83 |
| 12 | 7,646 | 227 | 2.97 |
| 13 | 7,534 | 250 | 3.32 |
| 14 | 7,523 | 230 | 3.06 |
| 15 | 7,793 | 233 | 2.99 |
| 16 | 7,251 | 231 | 3.19 |
| 17 | 7,518 | 210 | 2.79 |
| 18 | 7,524 | 218 | 2.9 |
| 19 | 7,227 | 260 | 3.6 |

## Per sumber label

| sumber | n | positif | prevalensi_% |
|---|---|---|---|
| ed | 50,879 | 32 | 0.06 |
| hosp | 98,124 | 4,790 | 4.88 |

## Subtipe di antara kasus positif (satu kasus bisa punya >1 subtipe)

| subtipe | n |
|---|---|
| ami_stemi | 906 |
| ami_nstemi | 3362 |
| ami_unspecified | <10 |
| ami_type2 | 562 |
| ami_other | <10 |
| ami_subsequent | 59 |
| ami_type2_only | 550 |

## Gerbang jumlah positif (fold 17 & 18 >= 100)

- Lolos.