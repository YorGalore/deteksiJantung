import numpy as np
import pytest

from src.model.metrics import confusion_matrix, evaluate, pr_auc, roc_auc


def test_confusion_and_basic_rates():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.2, 0.7, 0.3, 0.1, 0.1, 0.4]
    assert confusion_matrix(y, p, 0.5) == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}
    r = evaluate(y, p, 0.5)
    assert r["sensitivity"] == pytest.approx(2 / 3)
    assert r["specificity"] == pytest.approx(4 / 5)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)
    assert r["n"] == 8 and r["n_pos"] == 3


def test_perfect_and_inverted_ranking():
    y = [0, 0, 0, 1, 1]
    assert roc_auc(y, [0.1, 0.2, 0.3, 0.8, 0.9]) == 1.0
    assert pr_auc(y, [0.1, 0.2, 0.3, 0.8, 0.9]) == 1.0
    assert roc_auc(y, [0.9, 0.8, 0.7, 0.2, 0.1]) == 0.0


def test_ties_give_chance_level_auc():
    y = [0, 1, 0, 1, 0, 1]
    assert roc_auc(y, [0.5] * 6) == 0.5


def test_matches_pairwise_bruteforce_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.round(rng.random(200), 1)  # banyak ties
    pos, neg = p[y == 1], p[y == 0]
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert roc_auc(y, p) == pytest.approx(brute)


def test_pr_auc_known_value():
    # Urutan skor menurun: 1, 0, 1 -> AP = (1/1 + 2/3) / 2
    assert pr_auc([1, 0, 1], [0.9, 0.8, 0.7]) == pytest.approx((1 + 2 / 3) / 2)


def test_undefined_cases_are_nan_not_errors():
    r = evaluate([0, 0, 0], [0.1, 0.2, 0.3])
    assert np.isnan(r["roc_auc"]) and np.isnan(r["pr_auc"]) and np.isnan(r["sensitivity"])
    r = evaluate([1, 1, 0], [0.1, 0.2, 0.3])  # tidak ada prediksi positif
    assert np.isnan(r["precision"]) and np.isnan(r["f1"])


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        evaluate([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        evaluate([0, 1], [0.1, float("nan")])
    with pytest.raises(ValueError):
        evaluate([0, 1], [0.1])
