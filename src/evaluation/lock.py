"""Penjaga penguncian fold 19 (PROJECT_BRIEF 11.5; rencana analisis fold 19, Bagian 10).

Fold 19 hanya boleh dipakai (prediksi M0, SHAP, atau evaluasi akhir) bila semua berkas yang mengunci
analisis sudah di-commit dan tidak diubah sesudahnya:
  - reports/rencana_analisis_fold19.md   memuat baris "Status: FINAL"
  - reports/locked_fold18.json           kalibrasi + threshold dari fold 18
  - src/model/m0_rule.py                 aturan kata kunci M0
Penjaga ini membaca status git saja; tidak membaca data apa pun.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN_REL = Path("reports/rencana_analisis_fold19.md")
LOCKED_REL = Path("reports/locked_fold18.json")
M0_RULE_REL = Path("src/model/m0_rule.py")
FINAL_PATTERN = re.compile(r"^\W*status\W*final\b", re.IGNORECASE | re.MULTILINE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str], root: Path) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    except FileNotFoundError as e:                      # git tidak terpasang
        raise SystemExit("git tidak ditemukan; penguncian fold 19 tidak bisa diperiksa.") from e


def committed_unchanged(rel: Path, root: Path | None = None) -> str:
    """Hash commit terakhir yang menyentuh `rel`. Berhenti bila berkas tidak ada, tidak dilacak, atau berubah."""
    root = Path(root or ROOT)
    path = root / rel
    if not path.exists():
        raise SystemExit(f"Fold 19 dikunci: {rel} tidak ada.")
    if _git(["rev-parse", "--is-inside-work-tree"], root).returncode != 0:
        raise SystemExit(f"Fold 19 dikunci: {root} bukan repositori git.")
    if _git(["ls-files", "--error-unmatch", str(rel)], root).returncode != 0:
        raise SystemExit(f"Fold 19 dikunci: {rel} belum di-commit.")
    if _git(["diff", "--quiet", "HEAD", "--", str(rel)], root).returncode != 0:
        raise SystemExit(f"Fold 19 dikunci: {rel} berubah sejak commit terakhir. Commit atau kembalikan dulu.")
    return _git(["log", "-1", "--format=%h", "--", str(rel)], root).stdout.strip()


def require_fold19_unlocked(root: Path | None = None, need_m0_rule: bool = True) -> dict:
    """Kembalikan hash commit berkas pengunci. Berhenti dengan pesan jelas bila fold 19 belum boleh dipakai."""
    root = Path(root or ROOT)
    plan_commit = committed_unchanged(PLAN_REL, root)
    if not FINAL_PATTERN.search((root / PLAN_REL).read_text(encoding="utf-8")):
        raise SystemExit(f"Fold 19 dikunci: {PLAN_REL} belum berstatus FINAL (tulis 'Status: FINAL' setelah "
                         "disetujui tim, lalu commit).")
    out = {"plan_commit": plan_commit, "locked_fold18_commit": committed_unchanged(LOCKED_REL, root)}
    if need_m0_rule:
        out["m0_rule_commit"] = committed_unchanged(M0_RULE_REL, root)
    return out


def checksums(pred_dir: Path, fold: int, models: list[str]) -> dict[str, str | None]:
    """SHA-256 berkas prediksi (hanya byte berkas; isi tidak dibaca sebagai data)."""
    out = {}
    for m in models:
        p = Path(pred_dir) / f"{m}_fold{fold}.parquet"
        out[m] = sha256(p) if p.exists() else None
    return out
