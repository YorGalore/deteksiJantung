"""SHAP global untuk model tabular M2 dan M4 (Orang 3; rencana proyek: XAI SHAP XGBoost).

Jalankan dari root repo, di proses terpisah dari kode M1/torch (macOS, libomp):
    python -m src.model.shap_tabular --model M2 --fold 18
    python -m src.model.shap_tabular --model M4 --fold 18
    python -m src.model.shap_tabular --model M2 --fold 18 --cek-shap     # cocokkan dengan pustaka shap
    python -m src.model.shap_tabular --model M2 --stabilitas             # fold 18 vs 19 (setelah keduanya ada)

Nilai SHAP dihitung dengan algoritma TreeSHAP (path-dependent) bawaan XGBoost
(Booster.predict(pred_contribs=True)), pada iterasi terbaik yang sama dengan prediksi (iteration_range).
Ini definisi yang sama dengan shap.TreeExplainer untuk model pohon; --cek-shap membandingkan keduanya.
Skala: log-odds model MENTAH (sebelum Platt). Platt hanya mengalikan logit dengan b > 0 dan menambah a,
jadi urutan kepentingan fitur tidak berubah.

Pemeriksaan otomatis (gagal keras bila tidak lolos):
  - jumlah kontribusi + bias = margin model (aditivitas TreeSHAP);
  - sigmoid(margin) = y_prob di predictions/{model}_fold{N}.parquet (model dan baris yang sama).
Keluaran HANYA agregat (tidak ada nilai per pasien; kelompok < 10 kunjungan ditulis "<10"):
  reports/shap_{model}_fold{N}.md, reports/shap_{model}_fold{N}.csv, reports/figures/shap_{model}_fold{N}.png
Fold 19 hanya dengan --include-fold-19 setelah rencana analisis FINAL di-commit dan bila rencana memuatnya.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import expit

from src.constants import FOLD_COL, FOLD_TEST, FOLD_THRESHOLD
from src.evaluation import lock
from src.evaluation import predictions as P
from src.model import train_tabular as T
from src.model.tabular import ARRIVAL_CATEGORIES, ARRIVAL_COL, MODEL_SPECS, arrival_dummy_name, build_matrix

SHAP_MODELS = ("M2", "M4")
MIN_COUNT = 10
TOP_PLOT = 15
LABELS = {
    "age": "Usia", "is_male": "Jenis kelamin laki-laki", "arrival_transport": "Cara kedatangan",
    "temperature": "Suhu", "heartrate": "Denyut nadi triase", "resprate": "Laju napas", "o2sat": "SpO2",
    "sbp": "Tekanan sistolik", "dbp": "Tekanan diastolik", "pain": "Skala nyeri", "acuity": "Tingkat kegawatan (ESI)",
    "cc_chest_pain": "Keluhan nyeri dada", "cc_dyspnea": "Keluhan sesak napas", "cc_syncope": "Keluhan sinkop",
    "cc_weakness": "Keluhan lemas", "cc_epigastric_abd": "Keluhan nyeri epigastrium/perut",
    "cc_palpitations": "Keluhan palpitasi", "cc_dizziness": "Keluhan pusing", "cc_nausea_vomit": "Keluhan mual/muntah",
    "cc_diaphoresis": "Keluhan diaforesis", "cc_arm_jaw_pain": "Keluhan nyeri lengan/rahang",
    "cc_back_pain": "Keluhan nyeri punggung", "med_antiplatelet": "Obat antiplatelet",
    "med_anticoagulant": "Obat antikoagulan", "med_statin": "Obat statin", "med_betablocker": "Obat penyekat beta",
    "med_acei_arb": "Obat ACE-inhibitor/ARB", "med_nitrate": "Obat nitrat", "med_insulin": "Insulin",
    "med_oral_antidiabetic": "Antidiabetes oral", "med_loop_diuretic": "Diuretik loop",
    "med_ccb": "Penghambat kanal kalsium", "n_meds": "Jumlah obat rutin", "rr_interval": "Interval RR",
    "hr_ecg": "Denyut jantung EKG", "pr_interval": "Interval PR", "qrs_duration": "Durasi QRS",
    "qt_interval": "Interval QT", "qtc_bazett": "QTc Bazett", "p_axis": "Sumbu P", "qrs_axis": "Sumbu QRS",
    "t_axis": "Sumbu T", "p_wave_absent": "Gelombang P tidak terdeteksi",
}


def cnt(n: int) -> str:
    return f"{n:,}" if n >= MIN_COUNT else f"<{MIN_COUNT}"


def group_mean(phi: np.ndarray, mask: np.ndarray):
    """(n, rata-rata SHAP) kelompok; rata-rata disembunyikan bila n < 10."""
    n = int(mask.sum())
    return n, (float(phi[mask].mean()) if n >= MIN_COUNT else np.nan)


def contributions(model: str, fold: int):
    """Kontribusi TreeSHAP per kolom matriks + bias, dengan dua pemeriksaan konsistensi."""
    import xgboost as xgb
    meta_path, model_path = T.OUT_DIR / f"{model}_meta.json", T.OUT_DIR / f"{model}_model.json"
    if not (meta_path.exists() and model_path.exists()):
        raise SystemExit(f"Model {model} belum dilatih (tidak ada {model_path.name}).")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    master, info = T.load_master()
    if info["master_sha256"] != meta["data"]["master_sha256"]:
        raise SystemExit("Tabel master berbeda dari yang dipakai saat training model ini.")
    d = master[master[FOLD_COL] == fold].reset_index(drop=True)
    if d.empty:
        raise SystemExit(f"Tidak ada baris fold {fold} di tabel master.")
    spec = MODEL_SPECS[model]
    X = build_matrix(d, spec["features"])
    if list(X.columns) != list(meta["features_matrix"]):
        raise SystemExit("Kolom matriks fitur berbeda dari saat training.")
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    booster = clf.get_booster()
    best = int(meta["best_iteration"])
    dm = xgb.DMatrix(X, missing=np.nan)
    contrib = booster.predict(dm, pred_contribs=True, iteration_range=(0, best + 1))
    margin = booster.predict(dm, output_margin=True, iteration_range=(0, best + 1))
    phi, bias = contrib[:, :-1].astype(np.float64), contrib[:, -1].astype(np.float64)
    add_err = float(np.max(np.abs(phi.sum(axis=1) + bias - margin)))
    if add_err > 1e-3:
        raise SystemExit(f"Aditivitas SHAP gagal (selisih maksimum {add_err:.2e}).")

    pred = P.load_predictions(model, fold, pred_dir=P.PRED_DIR, allow_test=(fold == FOLD_TEST))
    chk = d[["ed_stay_id"]].assign(p_shap=expit(margin.astype(np.float64))).merge(
        pred[["ed_stay_id", "y_prob"]], on="ed_stay_id", how="inner", validate="one_to_one")
    if len(chk) != len(d):
        raise SystemExit(f"Baris fold {fold} ({len(d):,}) tidak sama dengan berkas prediksi ({len(chk):,}).")
    pred_err = float(np.max(np.abs(chk["p_shap"] - chk["y_prob"])))
    if pred_err > 1e-5:
        raise SystemExit(f"Prediksi model tersimpan berbeda dari predictions/{model}_fold{fold}.parquet "
                         f"(selisih maksimum {pred_err:.2e}).")
    return {"X": X, "phi": phi, "bias": bias, "booster": booster, "best": best, "d": d, "info": info,
            "add_err": add_err, "pred_err": pred_err}


def summarize_groups(X: pd.DataFrame, phi: np.ndarray, raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Satu baris per fitur asal (one-hot cara kedatangan dijumlahkan). Hanya agregat."""
    col = {c: i for i, c in enumerate(X.columns)}
    rows = []
    for f in features:
        if f == ARRIVAL_COL:
            idx = [col[arrival_dummy_name(c)] for c in ARRIVAL_CATEGORIES]
            v = phi[:, idx].sum(axis=1)
            s = raw[f].astype("string").str.upper().str.strip()
            groups = {f"kategori {c}": (s == c).fillna(False).to_numpy() for c in ARRIVAL_CATEGORIES}
            groups["kosong"] = s.isna().to_numpy()
            kind = "kategori"
        else:
            v = phi[:, col[f]]
            x = X[f].to_numpy(dtype=np.float64)
            miss = np.isnan(x)
            vals = np.unique(x[~miss])
            if len(vals) <= 2 and set(vals.tolist()) <= {0.0, 1.0}:
                kind = "biner"
                groups = {"= 1": x == 1, "= 0": x == 0, "kosong": miss}
            else:
                kind = "kontinu"
                q1, q3 = np.nanquantile(x, [0.25, 0.75]) if (~miss).any() else (np.nan, np.nan)
                groups = {"Q1": (x <= q1) & ~miss, "Q4": (x >= q3) & ~miss, "kosong": miss}
        row = {"fitur": f, "label": LABELS.get(f, f), "jenis": kind, "mean_abs_shap": float(np.mean(np.abs(v))),
               "mean_shap": float(np.mean(v))}
        if kind == "kontinu":
            row["batas Q1 (<=)"], row["batas Q4 (>=)"] = float(q1), float(q3)
        ns = {}
        for g, mask in groups.items():
            n, mean = group_mean(v, mask)
            ns[g] = n
            row[f"SHAP {g}"] = mean
        shown = {g: cnt(n) for g, n in ns.items()}
        if kind in ("biner", "kategori"):                  # kelompok membagi habis kunjungan: supresi komplementer
            small = [g for g, n in ns.items() if n < MIN_COUNT and n > 0]
            rest = sorted((n, g) for g, n in ns.items() if n >= MIN_COUNT)
            if len(small) == 1 and rest:
                shown[rest[0][1]] = "disembunyikan"
        for g in ns:
            row[f"n {g}"] = shown[g]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out.insert(0, "peringkat", np.arange(1, len(out) + 1))
    out["porsi_%"] = 100 * out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    return out


def direction(r: pd.Series) -> str:
    """Kalimat arah pengaruh dari rata-rata SHAP kelompok (agregat)."""
    def f(x):
        return "–" if pd.isna(x) else f"{x:+.3f}"
    if r["jenis"] == "biner":
        return f"=1: {f(r.get('SHAP = 1'))}; =0: {f(r.get('SHAP = 0'))}"
    if r["jenis"] == "kategori":
        parts = [f"{c}: {f(r.get(f'SHAP kategori {c}'))}" for c in ARRIVAL_CATEGORIES]
        return "; ".join(parts)
    return (f"Q1 (<= {r['batas Q1 (<=)']:.4g}): {f(r.get('SHAP Q1'))}; "
            f"Q4 (>= {r['batas Q4 (>=)']:.4g}): {f(r.get('SHAP Q4'))}")


def plot(summary: pd.DataFrame, model: str, fold: int, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    top = summary.head(TOP_PLOT).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 0.32 * len(top) + 1.0))
    ax.barh(top["label"], top["mean_abs_shap"], color="#2f6fd1", height=0.6)
    for y, v in enumerate(top["mean_abs_shap"]):
        ax.text(v, y, " " + f"{v:.3f}".replace(".", ","), va="center", fontsize=8)
    ax.set_xlabel("Rata-rata |SHAP| (log-odds model mentah)", fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    ax.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, top["mean_abs_shap"].max() * 1.18)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def cek_shap(r: dict, n: int = 500) -> float:
    """Bandingkan dengan shap.TreeExplainer pada booster yang dipotong ke iterasi terbaik."""
    try:
        import shap
    except ImportError:
        raise SystemExit("Pustaka shap belum terpasang. Jalankan tanpa --cek-shap (hasil SHAP tidak berubah), "
                         "atau pasang dulu: python3 -m pip install shap") from None
    sub = r["X"].iloc[:n]
    explainer = shap.TreeExplainer(r["booster"][: r["best"] + 1])
    sv = np.asarray(explainer.shap_values(sub), dtype=np.float64)
    return float(np.max(np.abs(sv - r["phi"][:n])))


def stability(model: str, top: int = 10) -> dict:
    """Cek stabilitas SHAP global fold 18 vs fold 19 dari dua CSV agregat (rencana analisis, Bagian SHAP):
    korelasi Spearman peringkat rata-rata |SHAP| semua fitur asal, dan irisan 10 fitur teratas.
    Deskriptif saja (tanpa ambang lulus/gagal). Tidak membaca data pasien."""
    from scipy.stats import spearmanr
    paths = {f: T.REPORTS / f"shap_{model}_fold{f}.csv" for f in (FOLD_THRESHOLD, FOLD_TEST)}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"Belum ada: {missing}. Jalankan SHAP fold 18 dan fold 19 dulu.")
    a, b = (pd.read_csv(paths[f]).set_index("fitur") for f in (FOLD_THRESHOLD, FOLD_TEST))
    if set(a.index) != set(b.index):
        raise SystemExit("Daftar fitur fold 18 dan fold 19 berbeda.")
    b = b.loc[a.index]
    rho = float(spearmanr(a["mean_abs_shap"], b["mean_abs_shap"]).statistic)
    top18 = list(a.sort_values("mean_abs_shap", ascending=False).index[:top])
    top19 = list(b.sort_values("mean_abs_shap", ascending=False).index[:top])
    overlap = len(set(top18) & set(top19))
    lines = [f"# Stabilitas SHAP {model}: fold 18 vs fold 19", "",
             "Dihasilkan otomatis oleh `src/model/shap_tabular.py --stabilitas`. Deskriptif, tanpa ambang lulus/gagal.", "",
             f"- Korelasi Spearman peringkat rata-rata |SHAP| ({len(a)} fitur asal): {rho:.3f}",
             f"- Irisan {top} fitur teratas: {overlap} dari {top}", "",
             f"| peringkat | fold 18 | fold 19 (hasil utama) |", "|---|---|---|",
             *[f"| {i + 1} | {a.loc[x, 'label']} | {b.loc[y, 'label']} |" for i, (x, y) in enumerate(zip(top18, top19))], ""]
    (T.REPORTS / f"shap_{model}_stabilitas.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[{model}] stabilitas SHAP fold 18 vs 19: Spearman {rho:.3f} | irisan {top} teratas {overlap}/{top} "
          f"-> reports/shap_{model}_stabilitas.md")
    return {"spearman": rho, "overlap": overlap}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SHAP global M2/M4 (TreeSHAP XGBoost)")
    ap.add_argument("--model", required=True, choices=SHAP_MODELS)
    ap.add_argument("--fold", type=int, default=FOLD_THRESHOLD, choices=[FOLD_THRESHOLD, FOLD_TEST])
    ap.add_argument("--include-fold-19", action="store_true")
    ap.add_argument("--cek-shap", action="store_true", help="cocokkan dengan pustaka shap (500 baris pertama)")
    ap.add_argument("--stabilitas", action="store_true",
                    help="bandingkan CSV SHAP fold 18 dan fold 19 yang sudah ada (tanpa menghitung ulang)")
    a = ap.parse_args(argv)
    if a.stabilitas:
        stability(a.model)
        return
    T.warn_if_torch_loaded()
    lock_info = {}
    if a.fold == FOLD_TEST:
        if not a.include_fold_19:
            raise SystemExit("Fold 19 dikunci (brief 11.5). SHAP fold 19 hanya bila rencana analisis memuatnya.")
        lock_info = lock.require_fold19_unlocked(T.ROOT, need_m0_rule=False)

    r = contributions(a.model, a.fold)
    features = MODEL_SPECS[a.model]["features"]
    summary = summarize_groups(r["X"], r["phi"], r["d"], features)
    cek = cek_shap(r) if a.cek_shap else None
    if a.cek_shap and "torch" in sys.modules:
        print("PERINGATAN: pustaka shap memuat torch; jalankan SHAP terpisah dari kode M1 di macOS.")

    stem = f"shap_{a.model}_fold{a.fold}"
    T.REPORTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(T.REPORTS / f"{stem}.csv", index=False)
    fig_path = T.REPORTS / "figures" / f"{stem}.png"
    plot(summary, a.model, a.fold, fig_path)
    n_pos = int(r["d"]["label_ami"].sum())
    rows = [f"| {s.peringkat} | {s.label} (`{s.fitur}`) | {s.mean_abs_shap:.4f} | {s['porsi_%']:.1f}% | "
            f"{s.mean_shap:+.4f} | {direction(s)} |" for _, s in summary.iterrows()]
    lines = [
        f"# SHAP {a.model} di fold {a.fold}", "",
        "Dihasilkan otomatis oleh `src/model/shap_tabular.py`. Hanya angka agregat; tidak ada nilai per pasien.", "",
        f"- Kunjungan: {len(r['d']):,} ({cnt(n_pos)} positif) | tabel master `{r['info']['master_file']}` "
        f"| filter sinyal: {r['info']['signal_filter']}",
        f"- TreeSHAP bawaan XGBoost (pred_contribs), iterasi 0-{r['best']}; skala log-odds model mentah (sebelum Platt).",
        f"- Cek aditivitas: selisih maksimum {r['add_err']:.1e} | cek terhadap berkas prediksi: {r['pred_err']:.1e}"
        + (f" | cek pustaka shap (500 baris): {cek:.1e}" if cek is not None else ""),
        f"- Nilai dasar (rata-rata bias): {float(np.mean(r['bias'])):+.4f} log-odds",
        f"- Dibuat: {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
        + (f" | rencana analisis {lock_info['plan_commit']}" if lock_info else ""), "",
        "Arah pengaruh = rata-rata SHAP pada kelompok kunjungan (biner: nilai 1 vs 0; kontinu: kuartil bawah Q1 vs "
        "kuartil atas Q4 di fold ini; kategori: per kategori). Positif = mendorong prediksi ke arah IMA. "
        "Kelompok < 10 kunjungan tidak ditampilkan.", "",
        "| peringkat | fitur | rata-rata abs SHAP | porsi | rata-rata SHAP | arah pengaruh |",
        "|---|---|---|---|---|---|", *rows, "",
        f"Gambar: `reports/figures/{stem}.png` ({TOP_PLOT} fitur teratas).", "",
    ]
    (T.REPORTS / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[{a.model}] fold {a.fold}: {len(r['d']):,} kunjungan | aditivitas {r['add_err']:.1e} | "
          f"cocok dengan berkas prediksi {r['pred_err']:.1e}" + (f" | cek shap {cek:.1e}" if cek is not None else ""))
    print(f"[{a.model}] 10 fitur teratas (rata-rata |SHAP|):")
    for _, s in summary.head(10).iterrows():
        print(f"  {s.peringkat:2d}. {s.label:38s} {s.mean_abs_shap:.4f}  ({s['porsi_%']:.1f}%)  {direction(s)}")
    print(f"[{a.model}] laporan: reports/{stem}.md | tabel: reports/{stem}.csv | gambar: reports/figures/{stem}.png")


if __name__ == "__main__":
    main()
