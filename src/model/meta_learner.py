"""Meta-learner late fusion (Orang 3): regresi logistik L2 pada logit prediksi model dasar.

M5 (late fusion utama, Jalur A) = LR(logit p_M1, logit p_M2)   -> sinyal EKG + EMR
M6 (late fusion tabular)          = LR(logit p_M3, logit p_M2)   -> machine measurements + EMR

Aturan (PROJECT_BRIEF 7.2, rencana proyek 6.3):
- Dilatih HANYA di fold 17. Model dasar M1/M2/M3 tidak pernah melihat fold 17, jadi prediksinya sah.
- Input = logit probabilitas mentah, distandarkan dengan rata-rata/SD fold 17. Kalibrasi model dasar tidak
  perlu: Platt adalah transformasi linear pada skala logit, dan regresi logistik menyerapnya.
- Regularisasi L2 ringan (C = 1, formulasi sama dengan scikit-learn: C * sum(logloss) + 0,5 * ||w||^2;
  intercept tidak dipenalti).
- Koefisien terstandar + CI bootstrap per subject_id dilaporkan sebagai XAI fusion.
Hanya NumPy/SciPy (tanpa scikit-learn) supaya aman dijalankan di proses yang memuat torch di macOS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit

from src.evaluation.bootstrap import cluster_resample_indices
from src.evaluation.core import safe_logit

META_SPECS = {
    "M5": {"inputs": ["M1", "M2"], "desc": "Late fusion utama: sinyal EKG (M1) + EMR (M2)"},
    "M6": {"inputs": ["M3", "M2"], "desc": "Late fusion tabular: machine measurements (M3) + EMR (M2)"},
}
C_DEFAULT = 1.0


def fit_l2_logistic(Z: np.ndarray, y: np.ndarray, C: float = C_DEFAULT,
                    max_iter: int = 100) -> tuple[float, np.ndarray]:
    """Minimalkan C * sum(logloss) + 0,5 * ||w||^2 (intercept tidak dipenalti). Kembalikan (intercept, w).

    Metode Newton dengan Hessian eksak + backtracking (Armijo). Fungsi objektif strictly convex (penalti L2
    pada w, dan intercept punya kelengkungan dari log-loss), jadi Newton konvergen kuadratik ke optimum unik.
    Berhenti bila Newton decrement lambda^2/2 <= 1e-14 * max(1, |f|): jauh di atas derau pembulatan f
    (sekitar 1e-16 * |f|), sehingga tidak bergantung pada heuristik line search atau BLAS platform.
    (v3 memakai L-BFGS-B dengan toleransi terlalu ketat -> status ABNORMAL di macOS; diganti di v3.1.)
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, k = Z.shape
    if not (0 < y.sum() < n):
        raise ValueError("Meta-learner butuh kedua kelas.")
    X = np.column_stack([np.ones(n), Z])
    pen = np.r_[0.0, np.ones(k)]                       # intercept tidak dipenalti

    def f(theta):
        z = X @ theta
        return C * float(np.sum(np.logaddexp(0.0, z) - y * z)) + 0.5 * float(np.sum(pen * theta ** 2))

    p0 = y.mean()
    theta = np.r_[np.log(p0 / (1 - p0)), np.zeros(k)]
    fx = f(theta)
    for _ in range(max_iter):
        p = expit(X @ theta)
        g = C * (X.T @ (p - y)) + pen * theta
        H = C * (X.T @ (X * (p * (1 - p))[:, None])) + np.diag(pen)
        step = np.linalg.solve(H, g)
        decrement = float(g @ step)                    # lambda^2 >= 0 karena H definit positif
        if decrement / 2 <= 1e-14 * max(1.0, abs(fx)):
            return float(theta[0]), theta[1:].copy()
        t = 1.0
        while True:
            cand = theta - t * step
            fc = f(cand)
            if fc <= fx - 1e-4 * t * decrement:
                break
            t *= 0.5
            if t < 1e-12:
                raise RuntimeError("Fit meta-learner gagal: line search Newton tidak menemukan penurunan.")
        theta, fx = cand, fc
    raise RuntimeError(f"Fit meta-learner gagal: belum konvergen setelah {max_iter} iterasi Newton.")


@dataclass
class MetaModel:
    inputs: list[str]
    mean: np.ndarray
    sd: np.ndarray
    intercept: float
    coef: np.ndarray                  # per 1 SD logit input (terstandar)
    C: float = C_DEFAULT
    extra: dict = field(default_factory=dict)

    @staticmethod
    def design(wide: pd.DataFrame, inputs: list[str]) -> np.ndarray:
        return np.column_stack([safe_logit(wide[f"p_{m}"].to_numpy()) for m in inputs])

    @classmethod
    def fit(cls, wide: pd.DataFrame, inputs: list[str], C: float = C_DEFAULT) -> "MetaModel":
        X = cls.design(wide, inputs)
        mean, sd = X.mean(axis=0), X.std(axis=0)
        if (sd < 1e-12).any():
            raise ValueError("Salah satu input meta-learner konstan; periksa berkas prediksi.")
        b, w = fit_l2_logistic((X - mean) / sd, wide["y_true"].to_numpy(), C)
        return cls(list(inputs), mean, sd, b, w, C)

    def predict(self, wide: pd.DataFrame) -> np.ndarray:
        Z = (self.design(wide, self.inputs) - self.mean) / self.sd
        return expit(self.intercept + Z @ self.coef)

    def to_dict(self) -> dict:
        return {"inputs": self.inputs, "mean": self.mean.tolist(), "sd": self.sd.tolist(),
                "intercept": self.intercept, "coef_standardized": self.coef.tolist(),
                "coef_per_logit_unit": (self.coef / self.sd).tolist(), "C": self.C, **self.extra}

    @classmethod
    def from_dict(cls, d: dict) -> "MetaModel":
        return cls(list(d["inputs"]), np.asarray(d["mean"], float), np.asarray(d["sd"], float),
                   float(d["intercept"]), np.asarray(d["coef_standardized"], float), float(d["C"]))


def bootstrap_coefficients(wide: pd.DataFrame, inputs: list[str], n_boot: int = 2000, seed: int = 42,
                           C: float = C_DEFAULT) -> pd.DataFrame:
    """CI 95% persentil untuk intercept dan koefisien terstandar (resample per subject_id, fit ulang).

    Standarisasi dihitung ulang di setiap sampel agar CI mencerminkan seluruh prosedur.
    """
    names = ["intercept"] + [f"logit_p_{m}" for m in inputs]
    draws = []
    for idx in cluster_resample_indices(wide["subject_id"].to_numpy(), n_boot, seed):
        w = wide.iloc[idx]
        if 0 < w["y_true"].sum() < len(w):
            m = MetaModel.fit(w, inputs, C)
            draws.append(np.r_[m.intercept, m.coef])
    draws = np.asarray(draws)
    point = MetaModel.fit(wide, inputs, C)
    est = np.r_[point.intercept, point.coef]
    out = pd.DataFrame({"term": names, "estimate": est,
                        "ci_low": np.quantile(draws, 0.025, axis=0), "ci_high": np.quantile(draws, 0.975, axis=0)})
    out["odds_ratio_per_sd"] = np.where(out["term"] == "intercept", np.nan, np.exp(out["estimate"]))
    out["n_valid"] = len(draws)
    return out
