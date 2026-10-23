## 2026-10-23 — Usulan aturan QC sinyal (Orang 2) — BELUM DISETUJUI

**Konteks.** Aturan PROJECT_BRIEF 10.2 (gagal jika ada lead dengan > 20% NaN, atau >= 3 lead datar) diterapkan pada 300 rekaman sampel (fold 0-16, acak: 100 positif + 200 negatif) dan 20 rekaman pilot. Hasil: 299 dari 300 sampel lolos.

**Temuan.** 4 dari 319 rekaman (1,3%) memiliki satu lead dengan artefak besar (maksimum 9,8-22,5 mV; p99 5,5-18,7 mV) sementara 11 lead lain normal (maksimum <= 2,4 mV). Aturan brief meloloskan keempatnya; setelah clip +-3 mV, lead itu menjadi derau jenuh. Ambang amplitudo 20 mV hanya menangkap 2 dari 4. Dua rekaman bervoltase tinggi yang tampak sah (maksimum 3,8 dan 5,2 mV, beberapa lead tinggi sekaligus) tidak termasuk kasus rusak.

**Usulan (perlu persetujuan tim).** Tambahkan aturan: gagal jika ada lead dengan |x| > 8 mV. Ambang ini berada di celah antara 5,2 mV (tampak sah) dan 9,8 mV (rusak) pada data yang ada. Alternatif: hanya menandai (`amplitude_flag`), atau menolkan lead yang rusak dan mempertahankan rekamannya.

**Catatan.** Dasar ambang hanya 6 rekaman. Validasi ulang pada rekaman tambahan dari fold 0-16 sebelum dikunci; jangan menyetel ambang memakai fold 19.
