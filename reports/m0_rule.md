# M0: aturan kata kunci interpretasi mesin EKG

Dihasilkan otomatis oleh `src/model/m0_rule.py`. Hanya angka agregat; tidak ada metrik kinerja.

- Versi aturan: **M0-v2** | sidik jari aturan (sha256): `f515ab4f4d4d8d85d2e249ecacbad94fe67fc4eaf93f5648b8e6d452df19589d`
- Tabel master: `master_table_draft.parquet` | filter sinyal: signal_ok == 1 dari signal_qc.csv (148,861 baris tersisa)
- mm_reports sha256: `c4cca5b2c97348b0...`

## Aturan

Satuan = satu pernyataan `report_*` (huruf kecil). Pernyataan positif bila cocok salah satu pola definisi dan tidak cocok pola negasi. Kunjungan positif bila minimal satu pernyataannya positif.

| nama | pola (regex) | M0 (primer) | M0S1 (sensitivitas) |
|---|---|---|---|
| stemi | `\bstemi\b` | ya | ya |
| acute_lalu_infark | `\bacute\b[^;]*\b(?:mi\|infarct\w*)\b` | ya | ya |
| infark_lalu_acute | `\b(?:mi\|infarct\w*)\b[^;]*\bacute\b` | ya | ya |
| infark_recent | `^(?!.*\b(?:compared\|comparison\|previous\|prior\|since\|tracing\|unchanged\|old\|age\s+(?:un\|in)determin\w*)\b).*?(?:\b(?:mi\|infarct\w*)\b,?\s+(?:(?:possibly\|probably\|probable\|possible\|likely\|is)\s+){0,2}(?:recent\|evolving)\b\|\b(?:recent\|evolving)\s+(?:(?!(?:or\|and\|pericardi\w*\|repolari\w*)\b)\w+\s+){0,2}?(?:mi\|infarct\w*)\b)` | ya | ya |
| injury | `\binjury\b` | ya | hanya injury_tunggal |
| acute_iskemia | `\bacute\s+(?:(?!(?:or\|and\|pericardi\w*\|repolari\w*)\b)\w+\s+){0,2}?\bisch(?:a)?emi\w*\|\bisch(?:a)?emi\w*,?\s+(?:(?:possibly\|probably\|probable\|possible\|likely\|is)\s+){0,2}acute\b` | ya | - |
| injury_tunggal | injury dan tidak cocok `\bisch(?:a)?emi\w*\|\bpericardi\w*\|\brepolari[sz]\w*` | - | ya |
| negasi (dikecualikan) | `\b(?:no\|without)\b[^,;]*\b(?:acute\|stemi\|infarct\w*\|injury\|mi)\b` | ya | ya |

- **M0**: primer: mesin menandai kemungkinan infark, injury, atau iskemia akut.
- **M0S1**: sensitivitas S1: mesin menyatakan infark akut/STEMI. Setiap kunjungan M0S1 = 1 juga M0 = 1.
- "injury" = istilah EKG (current of injury), bukan "myocardial injury" biokimia (UDMI 2018).

## Jumlah agregat

| fold | kunjungan | tanpa baris di mm_reports | tanpa teks report | M0 = 1 | M0S1 = 1 | pindah M0 -> M0S1 |
|---|---|---|---|---|---|---|
| 18 | 7,520 | <10 | <10 | 186 | 162 | 24 |

M0 = M0S1 + pindah. Bila salah satu < 10, yang lain disembunyikan agar sel kecil tidak bisa dihitung.

Dibuat: 2026-11-23T22:54:11+00:00
