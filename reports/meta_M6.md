# M6: Late fusion tabular: machine measurements (M3) + EMR (M2)

Dihasilkan otomatis oleh `src/model/train_meta.py`. Hanya angka agregat.

- Dilatih di fold 17: 7,513 kunjungan (210 positif); input: logit p_M3, logit p_M2 (distandarkan)
- Regresi logistik L2, C = 1.0; bootstrap per subject_id 2000x (fit ulang di setiap sampel)
- Korelasi antar-input (logit) di fold 17: 0.419
- AUROC/AUPRC in-sample fold 17 (hanya cek kewajaran, bukan hasil): 0.864 / 0.211

| term | koefisien (per 1 SD) | CI 95% | odds ratio per 1 SD |
|---|---|---|---|
| intercept | -4.955 | [-5.337, -4.653] |  |
| logit_p_M3 | 0.511 | [0.338, 0.681] | 1.667 |
| logit_p_M2 | 1.745 | [1.509, 1.996] | 5.729 |

Porsi |koefisien terstandar| (gambaran bobot relatif): logit_p_M3: 0.227, logit_p_M2: 0.773

Catatan tafsir: input saling berkorelasi, jadi porsi bobot bukan dekomposisi kausal. Koefisien menjawab: seberapa besar log-odds AMI berubah per 1 SD logit model dasar, dengan input lain tetap.
