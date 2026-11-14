# M5: Late fusion utama: sinyal EKG (M1) + EMR (M2)

Dihasilkan otomatis oleh `src/model/train_meta.py`. Hanya angka agregat.

- Dilatih di fold 17: 7,513 kunjungan (210 positif); input: logit p_M1, logit p_M2 (distandarkan)
- Regresi logistik L2, C = 1.0; bootstrap per subject_id 2000x (fit ulang di setiap sampel)
- Korelasi antar-input (logit) di fold 17: 0.500
- AUROC/AUPRC in-sample fold 17 (hanya cek kewajaran, bukan hasil): 0.888 / 0.317

| term | koefisien (per 1 SD) | CI 95% | odds ratio per 1 SD |
|---|---|---|---|
| intercept | -5.162 | [-5.583, -4.839] |  |
| logit_p_M1 | 0.950 | [0.794, 1.124] | 2.585 |
| logit_p_M2 | 1.484 | [1.243, 1.739] | 4.410 |

Porsi |koefisien terstandar| (gambaran bobot relatif): logit_p_M1: 0.39, logit_p_M2: 0.61

Catatan tafsir: input saling berkorelasi, jadi porsi bobot bukan dekomposisi kausal. Koefisien menjawab: seberapa besar log-odds AMI berubah per 1 SD logit model dasar, dengan input lain tetap.
