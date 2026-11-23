"""Baseline klinis M0: aturan kata kunci pada interpretasi otomatis mesin EKG (PROJECT_BRIEF 11.2).

Jalankan dari root repo:
    python -m src.model.m0_rule --spec                       # spesifikasi aturan + sidik jari (decision log)
    python -m src.model.m0_rule --audit-angka                # hitungan pola di fold 0-16, TANPA label (aman dibagikan)
    python -m src.model.m0_rule --audit                      # + daftar teks mesin (lokal saja, JANGAN dibagikan)
    python -m src.model.m0_rule --folds 18                   # tulis predictions/M0_fold18 dan M0S1_fold18
    # HANYA setelah rencana analisis FINAL dan aturan ini di-commit:
    python -m src.model.m0_rule --folds 19 --include-fold-19

ATURAN M0-v2 (dikunci sebelum M0 dinilai pada fold mana pun; M0-v1 = draf yang tidak pernah di-commit,
diperbaiki setelah audit pola di fold 0-16 TANPA label):
  Satuan = satu pernyataan report_0 ... report_17 dari EKG pertama kunjungan (huruf kecil).
  Pernyataan positif bila cocok salah satu pola definisi dan TIDAK cocok NEGATION_PATTERN.
  Kunjungan positif (y_prob = 1) bila minimal satu pernyataannya positif; selain itu 0, termasuk
  kunjungan tanpa teks report.
  - M0   (definisi PRIMER, "mesin menandai kemungkinan infark, injury, atau iskemia akut"):
           stemi, acute + infark (dua urutan), infark + recent/evolving, injury, acute + iskemia.
  - M0S1 (analisis sensitivitas S1, "mesin menyatakan infark akut/STEMI"):
           stemi, acute + infark, infark + recent/evolving, dan injury HANYA bila pernyataan yang sama
           tidak menyebut iskemia, perikarditis, atau repolarisasi. acute + iskemia tidak dihitung.
  Infark lama / "age undetermined/indeterminate" tanpa acute/recent/evolving tidak dihitung di keduanya.
  "injury" = istilah EKG (current of injury), bukan "myocardial injury" biokimia (UDMI 2018).
M0 tidak dilatih dan tidak punya threshold. Skrip ini TIDAK menghitung kinerja (tidak membaca label
untuk menilai aturan); kinerja hanya dihitung oleh src/evaluation/run_fold19.py setelah aturan dikunci.

Masukan : data/interim/mm_reports.parquet (ed_stay_id + report_*, KHUSUS M0) dan tabel master
          (lewat src.model.train_tabular.load_master, jadi filter signal_ok sama dengan model lain).
Keluaran: predictions/M0_fold{N}.parquet dan predictions/M0S1_fold{N}.parquet (kontrak 12.1; berisi ID,
          jangan di-commit); reports/m0_rule.md (spesifikasi aturan + jumlah agregat, aman di-commit)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import FOLD_COL, FOLD_TEST, FOLD_THRESHOLD, FOLDS_BASE_TRAIN
from src.evaluation import lock
from src.evaluation import predictions as P
from src.evaluation.predictions import CONTRACT, validate_predictions
from src.model import train_tabular as T

RULE_VERSION = "M0-v2"
REPORT_COLS = [f"report_{i}" for i in range(18)]
_INF = r"\b(?:mi|infarct\w*)\b"
_ISCH = r"\bisch(?:a)?emi\w*"
_GAP = r"(?:(?!(?:or|and|pericardi\w*|repolari\w*)\b)\w+\s+){0,2}?"          # <= 2 kata, bukan or/and/perikarditis
_HEDGE = r"(?:(?:possibly|probably|probable|possible|likely|is)\s+){0,2}"           # <= 2 kata keraguan
_NOT_OLD_OR_COMPARE = (r"^(?!.*\b(?:compared|comparison|previous|prior|since|tracing|unchanged|old"
                       r"|age\s+(?:un|in)determin\w*)\b)")
PATTERNS = {
    "stemi": r"\bstemi\b",
    "acute_lalu_infark": r"\bacute\b[^;]*" + _INF,
    "infark_lalu_acute": _INF + r"[^;]*\bacute\b",
    "infark_recent": _NOT_OLD_OR_COMPARE + r".*?(?:" + _INF + r",?\s+" + _HEDGE + r"(?:recent|evolving)\b"
                     r"|\b(?:recent|evolving)\s+" + _GAP + r"(?:mi|infarct\w*)\b)",
    "injury": r"\binjury\b",
    "acute_iskemia": r"\bacute\s+" + _GAP + _ISCH + r"|" + _ISCH + r",?\s+" + _HEDGE + r"acute\b",
}
INJURY_ALTERNATIVES = _ISCH + r"|\bpericardi\w*|\brepolari[sz]\w*"
NEGATION_PATTERN = r"\b(?:no|without)\b[^,;]*\b(?:acute|stemi|infarct\w*|injury|mi)\b"
VARIANTS = {
    "M0": {"label": "primer: mesin menandai kemungkinan infark, injury, atau iskemia akut",
           "pola": ["stemi", "acute_lalu_infark", "infark_lalu_acute", "infark_recent", "injury", "acute_iskemia"]},
    "M0S1": {"label": "sensitivitas S1: mesin menyatakan infark akut/STEMI",
             "pola": ["stemi", "acute_lalu_infark", "infark_lalu_acute", "infark_recent", "injury_tunggal"]},
}
ACUTE_PATTERNS = PATTERNS                     # nama lama, dipakai laporan audit
AUDIT_STEMS = r"\b(?:acute|infarct\w*|injury|stemi|mi|recent|evolving)\b|" + _ISCH
MIN_COUNT = 10


_RX: dict[str, re.Pattern] = {}


def contains(series: pd.Series, pattern: str) -> pd.Series:
    """Pencocokan regex dengan modul `re` Python, apa pun backend string pandas (pyarrow/RE2 tidak mendukung
    lookahead dan bisa berbeda untuk \\w). Hasil aturan tidak bergantung pada versi pandas."""
    rx = _RX.get(pattern) or _RX.setdefault(pattern, re.compile(pattern))
    return pd.Series([isinstance(v, str) and rx.search(v) is not None for v in series.astype(object)],
                     index=series.index, dtype=bool)


def rule_spec() -> dict:
    return {"version": RULE_VERSION, "patterns": PATTERNS, "injury_tunggal": "injury dan TIDAK cocok "
            "injury_alternatives", "injury_alternatives": INJURY_ALTERNATIVES, "negation_pattern": NEGATION_PATTERN,
            "variants": {k: v["pola"] for k, v in VARIANTS.items()}, "primary": "M0",
            "unit": "satu pernyataan report_i (huruf kecil)",
            "positif_bila": "minimal satu pernyataan cocok salah satu pola varian dan tidak cocok pola negasi"}


def rule_sha256() -> str:
    """Sidik jari aturan (pola + versi). Dicatat di decision log saat aturan dikunci."""
    return hashlib.sha256(json.dumps(rule_spec(), sort_keys=True).encode("utf-8")).hexdigest()


def cnt(n: int) -> str:
    return f"{n:,}" if n >= MIN_COUNT else f"<{MIN_COUNT}"


def statements(reports: pd.DataFrame) -> pd.DataFrame:
    """Tabel panjang: satu baris per pernyataan tidak kosong (ed_stay_id, text huruf kecil)."""
    cols = [c for c in REPORT_COLS if c in reports.columns]
    if not cols:
        raise ValueError("mm_reports tidak punya kolom report_*.")
    long = reports[["ed_stay_id"] + cols].melt(id_vars="ed_stay_id", value_name="text")["ed_stay_id text".split()]
    long["text"] = long["text"].astype("string").str.strip().str.lower()
    return long[long["text"].notna() & (long["text"] != "")].reset_index(drop=True)


def statement_flags(long: pd.DataFrame) -> pd.DataFrame:
    """Kolom boolean per pola, negasi, dan hasil per varian (pos_M0, pos_M0S1) per pernyataan.
    Kolom `positive` = definisi primer (M0)."""
    out = long.copy()
    for name, pat in PATTERNS.items():
        out[name] = contains(out["text"], pat)
    out["injury_tunggal"] = out["injury"] & ~contains(out["text"], INJURY_ALTERNATIVES)
    out["negated"] = contains(out["text"], NEGATION_PATTERN)
    for code, v in VARIANTS.items():
        out[f"pos_{code}"] = out[v["pola"]].any(axis=1) & ~out["negated"]
    out["acute_any"] = out[VARIANTS["M0"]["pola"]].any(axis=1)
    out["positive"] = out["pos_M0"]
    return out


def m0_per_visit(reports: pd.DataFrame) -> pd.DataFrame:
    """ed_stay_id, m0 (primer), m0s1 (sensitivitas S1), n_statements. Kunjungan tanpa teks tetap ada (0)."""
    if reports["ed_stay_id"].duplicated().any():
        raise ValueError("ed_stay_id duplikat di mm_reports.")
    flags = statement_flags(statements(reports))
    agg = flags.groupby("ed_stay_id").agg(m0=("pos_M0", "any"), m0s1=("pos_M0S1", "any"),
                                          n_statements=("text", "size"))
    out = reports[["ed_stay_id"]].merge(agg, left_on="ed_stay_id", right_index=True, how="left")
    for c in ("m0", "m0s1"):
        out[c] = out[c].fillna(False).astype("int8")
    out["n_statements"] = out["n_statements"].fillna(0).astype("int64")
    return out.reset_index(drop=True)


def load_reports() -> tuple[pd.DataFrame, str]:
    path = T.ROOT / "data/interim/mm_reports.parquet"
    if not path.exists():
        raise SystemExit("Tidak ada data/interim/mm_reports.parquet (dibuat oleh src.dataset.features_mm, Orang 1).")
    return pd.read_parquet(path), lock.sha256(path)


def write_predictions(folds: list[int], include_19: bool) -> None:
    folds = sorted(set(int(f) for f in folds))
    bad = [f for f in folds if f not in (FOLD_THRESHOLD, FOLD_TEST)]
    if bad:
        raise SystemExit(f"M0 hanya ditulis untuk fold evaluasi 18 dan 19, bukan {bad}.")
    lock_info = {}
    if FOLD_TEST in folds:
        if not include_19:
            raise SystemExit("Fold 19 dikunci (brief 11.5). Tambahkan --include-fold-19 setelah rencana analisis "
                             "FINAL dan aturan M0 di-commit.")
        lock_info = lock.require_fold19_unlocked(T.ROOT)

    master, info = T.load_master()
    reports, reports_sha = load_reports()
    visits = m0_per_visit(reports)
    rows = []
    for f in folds:
        d = master[master[FOLD_COL] == f][["ed_stay_id", "subject_id", "study_id", FOLD_COL, "label_ami"]]
        if d.empty:
            raise SystemExit(f"Tidak ada baris fold {f} di tabel master.")
        d = d.merge(visits, on="ed_stay_id", how="left", validate="one_to_one")
        no_report_row = int(d["m0"].isna().sum())
        for c in ("m0", "m0s1"):
            d[c] = d[c].fillna(0).astype("int8")
        d["n_statements"] = d["n_statements"].fillna(0)
        P.PRED_DIR.mkdir(parents=True, exist_ok=True)
        r = {"fold": f, "n": len(d), "tanpa_baris_report": no_report_row, "tanpa_teks": int((d["n_statements"] == 0).sum()),
             "M0": int(d["m0"].sum()), "M0S1": int(d["m0s1"].sum()),
             "pindah": int(((d["m0"] == 1) & (d["m0s1"] == 0)).sum())}
        if ((d["m0s1"] == 1) & (d["m0"] == 0)).any():
            raise SystemExit("M0S1 = 1 tetapi M0 = 0: S1 seharusnya bagian dari definisi primer. Periksa aturan.")
        for code, col in (("M0", "m0"), ("M0S1", "m0s1")):
            out = pd.DataFrame({"ed_stay_id": d["ed_stay_id"], "subject_id": d["subject_id"], "study_id": d["study_id"],
                                "fold": int(f), "y_true": d["label_ami"].astype("int8"),
                                "y_prob": d[col].astype("float64")})
            out = validate_predictions(out[CONTRACT], expected_fold=f, name=f"{code}_fold{f}")
            out.to_parquet(P.PRED_DIR / f"{code}_fold{f}.parquet", index=False)
        rows.append(r)
        s1_txt = cnt(r["M0S1"]) if r["pindah"] >= MIN_COUNT else "disembunyikan"
        print(f"[M0] fold {f}: {r['n']:,} kunjungan | tanpa teks report {cnt(r['tanpa_teks'])} | "
              f"M0 = 1: {cnt(r['M0'])} | M0S1 = 1: {s1_txt} | pindah M0 -> M0S1: {cnt(r['pindah'])} "
              f"-> M0_fold{f}.parquet, M0S1_fold{f}.parquet")

    def esc(v: str) -> str:
        return v.replace("|", chr(92) + "|")

    lines = [
        "# M0: aturan kata kunci interpretasi mesin EKG", "",
        "Dihasilkan otomatis oleh `src/model/m0_rule.py`. Hanya angka agregat; tidak ada metrik kinerja.", "",
        f"- Versi aturan: **{RULE_VERSION}** | sidik jari aturan (sha256): `{rule_sha256()}`",
        f"- Tabel master: `{info['master_file']}` | filter sinyal: {info['signal_filter']}",
        f"- mm_reports sha256: `{reports_sha[:16]}...`", "",
        "## Aturan", "",
        "Satuan = satu pernyataan `report_*` (huruf kecil). Pernyataan positif bila cocok salah satu pola definisi dan "
        "tidak cocok pola negasi. Kunjungan positif bila minimal satu pernyataannya positif.", "",
        "| nama | pola (regex) | M0 (primer) | M0S1 (sensitivitas) |", "|---|---|---|---|",
        *[f"| {k} | `{esc(v)}` | {'ya' if k in VARIANTS['M0']['pola'] else '-'} | "
          f"{'ya' if k in VARIANTS['M0S1']['pola'] else ('hanya injury_tunggal' if k == 'injury' else '-')} |"
          for k, v in PATTERNS.items()],
        f"| injury_tunggal | injury dan tidak cocok `{esc(INJURY_ALTERNATIVES)}` | - | ya |",
        f"| negasi (dikecualikan) | `{esc(NEGATION_PATTERN)}` | ya | ya |", "",
        f"- **M0**: {VARIANTS['M0']['label']}.",
        f"- **M0S1**: {VARIANTS['M0S1']['label']}. Setiap kunjungan M0S1 = 1 juga M0 = 1.",
        "- \"injury\" = istilah EKG (current of injury), bukan \"myocardial injury\" biokimia (UDMI 2018).", "",
        "## Jumlah agregat", "",
        "| fold | kunjungan | tanpa baris di mm_reports | tanpa teks report | M0 = 1 | M0S1 = 1 | pindah M0 -> M0S1 |",
        "|---|---|---|---|---|---|---|",
        *[f"| {r['fold']} | {r['n']:,} | {cnt(r['tanpa_baris_report'])} | {cnt(r['tanpa_teks'])} | {cnt(r['M0'])} | "
          f"{cnt(r['M0S1']) if r['pindah'] >= MIN_COUNT else 'disembunyikan'} | {cnt(r['pindah'])} |" for r in rows], "",
        "M0 = M0S1 + pindah. Bila salah satu < 10, yang lain disembunyikan agar sel kecil tidak bisa dihitung.", "",
        f"Dibuat: {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
        + (f" | rencana analisis {lock_info['plan_commit']}, aturan {lock_info['m0_rule_commit']}" if lock_info else ""),
        "",
    ]
    T.REPORTS.mkdir(parents=True, exist_ok=True)
    (T.REPORTS / "m0_rule.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[M0] laporan: reports/m0_rule.md | aturan {RULE_VERSION}, sha256 {rule_sha256()[:16]}...")


# Pola bantu KHUSUS audit (tidak memengaruhi aturan dan sidik jarinya)
_A_INF = r"\b(?:mi|infarct\w*)\b"
_A_ACUTE = r"\bacute\b"
_A_INJ = r"\binjury\b"
_A_STEMI = r"\bstemi\b"
_A_ISCH = r"\bisch(?:a)?emi\w*"
_A_DIFF = r"\b(?:pericardi\w*|repolari[sz]\w*)"
_A_ALT = r"\bor\b|/"
_A_AGE = r"\bage\s+(?:un|in)determin\w*|\bold\b"
_A_ST_ELEV = r"\bst\W*elev\w*"
_A_ABBR = r"\b(?:ami|asmi|imi|lmi|almi|ipmi|infarc)\b"
_A_RECENT = r"\b(?:recent\w*|evolv\w*)\b"
_A_HANG = r"(?:,|\b(?:or|and|possibly|possible|probable|consider|cannot|can not|rule out))\s*$"
_A_COMPARE = r"\b(?:compared|comparison|previous|prior|since|tracing|last)\b"


def statements_ordered(reports: pd.DataFrame) -> pd.DataFrame:
    """Seperti statements(), ditambah nomor kolom report (idx) untuk mendeteksi pernyataan terpotong."""
    cols = [c for c in REPORT_COLS if c in reports.columns]
    if not cols:
        raise ValueError("mm_reports tidak punya kolom report_*.")
    long = reports[["ed_stay_id"] + cols].melt(id_vars="ed_stay_id", var_name="kolom", value_name="text")
    long["idx"] = long["kolom"].str.removeprefix("report_").astype(int)
    long["text"] = long["text"].astype("string").str.strip().str.lower()
    long = long[long["text"].notna() & (long["text"] != "")]
    return long[["ed_stay_id", "idx", "text"]].sort_values(["ed_stay_id", "idx"]).reset_index(drop=True)


def audit_counts(reports: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    """Hitungan agregat untuk menilai aturan (tanpa label). Mengembalikan (hitungan, tabel flag per pernyataan)."""
    f = statement_flags(statements_ordered(reports))
    t = f["text"]

    def has(pat):
        return contains(t, pat)

    inf, acute, inj, stemi = has(_A_INF), has(_A_ACUTE), has(_A_INJ), has(_A_STEMI)
    isch, diff, alt, age = has(_A_ISCH), has(_A_DIFF), has(_A_ALT), has(_A_AGE)
    pos, notpos = f["positive"], ~f["positive"]
    other = f[[k for k in VARIANTS["M0"]["pola"] if k != "injury"]].any(axis=1)
    inj_only = pos & f["injury"] & ~other
    visit_pos = f.groupby("ed_stay_id")["positive"].any()
    visit_s1 = f.groupby("ed_stay_id")["pos_M0S1"].any()

    g = pd.DataFrame({"ed_stay_id": f["ed_stay_id"], "a": inj_only, "b": inj_only & isch, "c": inj_only & diff})[pos]
    per = g.groupby("ed_stay_id")[["a", "b", "c"]].all()

    same_next = (f["ed_stay_id"].shift(-1) == f["ed_stay_id"]) & (f["idx"].shift(-1) == f["idx"] + 1)
    inf_n, acute_n = inf.shift(-1, fill_value=False), acute.shift(-1, fill_value=False)
    fwd = inf & ~acute & acute_n & ~inf_n                 # baris i: infark, baris i+1: acute
    bwd = acute & ~inf & inf_n & ~acute_n                 # baris i: acute, baris i+1: infark
    split = same_next & (fwd | bwd)
    split_visits = set(f.loc[split, "ed_stay_id"])
    neg_visits = set(visit_pos.index[~visit_pos])
    # Rincian pasangan pada kunjungan M0 = 0: baris "acute" pendek, baris pertama menggantung, perikarditis
    in_neg = f["ed_stay_id"].isin(neg_visits)
    pair_i = split & in_neg
    text_n = t.shift(-1)
    acute_line = t.where(bwd, text_n)                     # teks baris yang memuat acute
    first_line = t                                        # baris pertama pasangan (urutan asli)
    n_words = acute_line.str.split().str.len()
    peri_acute = contains(acute_line, _A_DIFF)
    hang = contains(first_line, _A_HANG)
    b4 = notpos & acute & ~inf & ~inj & ~stemi
    b3 = notpos & inf & ~acute & ~age
    recent = f["infark_recent"] & ~f["negated"]
    akis = f["acute_iskemia"] & ~f["negated"]

    c = {
        "pernyataan": len(f), "kunjungan_m0": int(visit_pos.sum()), "kunjungan_m0s1": int(visit_s1.sum()),
        "kunjungan_pindah": int((visit_pos & ~visit_s1).sum()),
        **{f"pola_{k}": int(f[k].sum()) for k in ACUTE_PATTERNS},
        "pola_injury_tunggal": int(f["injury_tunggal"].sum()),
        "d_recent_saja": int((recent & ~f[["stemi", "acute_lalu_infark", "infark_lalu_acute", "injury"]].any(axis=1)).sum()),
        "d_recent_usia": int((recent & age).sum()),
        "d_recent_pembanding": int((recent & has(_A_COMPARE)).sum()),
        "d_akis_saja": int((akis & ~f[["stemi", "acute_lalu_infark", "infark_lalu_acute", "infark_recent", "injury"]]
                                   .any(axis=1)).sum()),
        "d_akis_perikarditis": int((akis & diff).sum()),
        "dinegasi": int((f["acute_any"] & f["negated"]).sum()),
        "a_pos": int(pos.sum()), "a_inj_saja": int(inj_only.sum()),
        "a_inj_iskemia": int((inj_only & isch).sum()), "a_inj_peri_repol": int((inj_only & diff).sum()),
        "a_inj_alternatif": int((inj_only & alt).sum()),
        "a_kunj_hanya_inj": int(per["a"].sum()), "a_kunj_hanya_inj_iskemia": int(per["b"].sum()),
        "a_kunj_hanya_inj_peri_repol": int(per["c"].sum()),
        "b_acute_dan_infark": int((notpos & acute & inf).sum()),
        "b_infark_usia": int((notpos & inf & ~acute & age).sum()),
        "b_infark_lain": int((notpos & inf & ~acute & ~age).sum()),
        "b3_recent_evolving": int((b3 & has(_A_RECENT)).sum()),
        "b_acute_saja": int(b4.sum()),
        "b4_perikarditis": int((b4 & diff).sum()),
        "b4_iskemia": int((b4 & isch).sum()),
        "b4_st_elevasi": int((b4 & has(_A_ST_ELEV)).sum()),
        "b4_singkatan_infark": int((b4 & has(_A_ABBR)).sum()),
        "b4_lain": int((b4 & ~diff & ~isch & ~has(_A_ST_ELEV) & ~has(_A_ABBR)).sum()),
        "b_stemi": int((notpos & stemi).sum()),
        "c_terpotong_kunj_m0_0": len(split_visits & neg_visits),
        "c_terpotong_kunj_m0_1": len(split_visits - neg_visits),
        "c1_pasangan": int(pair_i.sum()),
        "c1_acute_pendek": int((pair_i & (n_words <= 3) & ~peri_acute).sum()),
        "c1_baris_pertama_menggantung": int((pair_i & hang).sum()),
        "c1_acute_perikarditis": int((pair_i & peri_acute).sum()),
    }
    return c, f


AUDIT_LINES = [
    ("pernyataan", "pernyataan tidak kosong"),
    ("kunjungan_m0", "kunjungan M0 = 1 (primer)"),
    ("kunjungan_m0s1", "kunjungan M0S1 = 1 (sensitivitas S1)"),
    ("kunjungan_pindah", "kunjungan M0 = 1 tetapi M0S1 = 0"),
    *[(f"pola_{k}", f"pola {k}") for k in ACUTE_PATTERNS],
    ("pola_injury_tunggal", "pola injury_tunggal (S1)"),
    ("dinegasi", "pernyataan cocok pola tapi dinegasi (dikecualikan)"),
    (None, "D. Cek pola baru M0-v2"),
    ("d_recent_saja", "D1 positif HANYA karena infark + recent/evolving"),
    ("d_recent_usia", "D1a  ... pernyataan infark_recent yang juga menyebut age un/indetermined atau old"),
    ("d_recent_pembanding", "D1b  ... pernyataan infark_recent yang juga memuat kata pembanding (compared/previous/prior/since/tracing/last)"),
    ("d_akis_saja", "D2 positif HANYA karena acute + iskemia"),
    ("d_akis_perikarditis", "D2a  ... pernyataan acute_iskemia yang juga menyebut perikarditis/repolarisasi"),
    (None, "A. Pernyataan yang membuat M0 = 1"),
    ("a_pos", "A0 semua pernyataan positif"),
    ("a_inj_saja", "A1 positif HANYA karena kata injury"),
    ("a_inj_iskemia", "A1a  ... dan juga menyebut iskemia (ischemia/ischaemia)"),
    ("a_inj_peri_repol", "A1b  ... dan juga menyebut perikarditis atau repolarisasi"),
    ("a_inj_alternatif", "A1c  ... dan memuat 'or' atau '/' (injury salah satu kemungkinan)"),
    ("a_kunj_hanya_inj", "A2 kunjungan M0 = 1 yang SEMUA alasannya kategori A1"),
    ("a_kunj_hanya_inj_iskemia", "A2a  ... dan semuanya juga menyebut iskemia"),
    ("a_kunj_hanya_inj_peri_repol", "A2b  ... dan semuanya juga menyebut perikarditis/repolarisasi"),
    (None, "B. Pernyataan terkait yang TIDAK membuat M0 = 1"),
    ("b_acute_dan_infark", "B1 memuat acute DAN mi/infarct (harus = jumlah dinegasi)"),
    ("b_infark_usia", "B2 mi/infarct tanpa acute, dengan 'age un/indetermined' atau 'old'"),
    ("b_infark_lain", "B3 mi/infarct tanpa acute dan tanpa keterangan usia"),
    ("b3_recent_evolving", "B3a  ... dan menyebut recent/evolving (sisa yang tidak tertangkap pola infark_recent)"),
    ("b_acute_saja", "B4 acute tanpa mi/infarct/injury/stemi"),
    ("b4_perikarditis", "B4a  ... menyebut perikarditis atau repolarisasi"),
    ("b4_iskemia", "B4b  ... menyebut iskemia (sisa yang tidak tertangkap pola acute_iskemia)"),
    ("b4_st_elevasi", "B4c  ... menyebut ST elevation"),
    ("b4_singkatan_infark", "B4d  ... memuat singkatan infark (ami/asmi/imi/lmi/almi/ipmi/infarc)"),
    ("b4_lain", "B4e  ... tidak termasuk B4a-B4d"),
    ("b_stemi", "B5 memuat stemi tapi tidak positif"),
    (None, "C. Kemungkinan pernyataan terpotong (report_i dan report_i+1 berurutan)"),
    ("c_terpotong_kunj_m0_0", "C1 kunjungan M0 = 0 dengan mi/infarct di satu baris dan acute di baris sebelah"),
    ("c_terpotong_kunj_m0_1", "C2 sama, tapi kunjungannya sudah M0 = 1"),
    ("c1_pasangan", "C3 jumlah pasangan baris di kunjungan C1"),
    ("c1_acute_pendek", "C3a  ... baris acute pendek (<= 3 kata) dan bukan perikarditis"),
    ("c1_baris_pertama_menggantung", "C3b  ... baris pertama berakhir koma/or/and/possibly/consider/..."),
    ("c1_acute_perikarditis", "C3c  ... baris acute menyebut perikarditis/repolarisasi"),
]


def audit(top: int, show_text: bool) -> None:
    """Cek pola pada fold 0-16 tanpa label. Bagian ANGKA aman dibagikan. Bagian TEKS (hanya bila
    show_text) mencetak teks mesin: JANGAN dibagikan ke layanan AI."""
    master, _ = T.load_master()
    keep = set(master.loc[master[FOLD_COL].isin(FOLDS_BASE_TRAIN), "ed_stay_id"])
    reports, _ = load_reports()
    c, flags = audit_counts(reports[reports["ed_stay_id"].isin(keep)])
    print("=" * 78)
    print(f"AUDIT ANGKA {RULE_VERSION} (fold 0-16, tanpa label). Bagian ini AMAN dibagikan.")
    print("=" * 78)
    print(f"kunjungan fold 0-16: {len(keep):,}")
    shown = {k: cnt(v) for k, v in c.items()}
    if c["kunjungan_pindah"] < MIN_COUNT:                        # M0 = M0S1 + pindah: supresi komplementer
        shown["kunjungan_m0s1"] = "disembunyikan"
    for key, label in AUDIT_LINES:
        if key is None:
            print(f"\n{label}")
        else:
            print(f"  {label}: {shown[key]}")
    print("=== AKHIR BAGIAN ANGKA ===")
    if not show_text:
        return

    def show(title: str, sel: pd.Series) -> None:
        vc = flags.loc[sel, "text"].value_counts()
        vc = vc[vc >= MIN_COUNT].head(top)
        print(f"\n--- TEKS (lokal saja, JANGAN dibagikan): {title} [{len(vc)} teratas, hanya yang >= {MIN_COUNT}x]")
        for text, n in vc.items():
            print(f"  {n:7,}  {text}")

    show("pernyataan yang MEMBUAT M0 = 1", flags["positive"])
    related = contains(flags["text"], AUDIT_STEMS)
    show("pernyataan terkait infark/akut yang TIDAK membuat M0 = 1", related & ~flags["positive"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Baseline M0: aturan kata kunci interpretasi mesin EKG")
    ap.add_argument("--folds", type=int, nargs="+", help="fold evaluasi yang ditulis (18; 19 dengan flag)")
    ap.add_argument("--include-fold-19", action="store_true")
    ap.add_argument("--audit-angka", action="store_true",
                    help="hitungan agregat aturan di fold 0-16 tanpa label (aman dibagikan)")
    ap.add_argument("--audit", action="store_true",
                    help="hitungan agregat + daftar TEKS mesin (lokal saja, jangan dibagikan)")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--spec", action="store_true", help="cetak spesifikasi aturan dan sidik jarinya")
    a = ap.parse_args(argv)
    if a.spec:
        print(json.dumps(rule_spec(), indent=2, ensure_ascii=False))
        print(f"sha256 aturan: {rule_sha256()}")
    if a.audit or a.audit_angka:
        audit(a.top, show_text=a.audit)
    if a.folds:
        write_predictions(a.folds, a.include_fold_19)
    if not (a.spec or a.audit or a.audit_angka or a.folds):
        ap.error("pilih minimal satu: --spec, --audit-angka, --audit, atau --folds")


if __name__ == "__main__":
    main()
