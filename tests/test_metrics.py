"""Юнит-тесты системы метрик на игрушечных примерах."""
import numpy as np
import pytest

from src.metrics import evaluate, recall_at_precision


def test_perfect_separation():
    # Идеально разделимые scores: все метрики = 1.0.
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    m = evaluate(y_true, y_score)
    assert m["auc_pr"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["recall_at_precision_90"] == pytest.approx(1.0)
    assert m["n_pos"] == 2 and m["n_neg"] == 3


def test_counts_and_keys():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_score = np.array([0.2, 0.8, 0.6, 0.4, 0.1, 0.9])
    m = evaluate(y_true, y_score)
    expected_keys = {
        "auc_pr", "roc_auc", "f1", "precision", "recall",
        "recall_at_precision_90", "n_pos", "n_neg", "threshold",
    }
    assert expected_keys <= set(m)
    assert m["n_pos"] == 3 and m["n_neg"] == 3
    for k in ("auc_pr", "roc_auc", "f1", "precision", "recall"):
        assert 0.0 <= m[k] <= 1.0


def test_random_scores_auc_pr_near_baseline():
    # При случайных scores AUC-PR ≈ доле позитивов (здесь 0.5).
    rng = np.random.default_rng(0)
    y_true = np.array([0, 1] * 500)
    y_score = rng.random(len(y_true))
    m = evaluate(y_true, y_score)
    assert m["auc_pr"] == pytest.approx(0.5, abs=0.1)
    assert m["roc_auc"] == pytest.approx(0.5, abs=0.1)


def test_recall_at_precision_unreachable_returns_zero():
    # Полностью перепутанные scores: precision>=0.9 недостижим → 0.0.
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert recall_at_precision(y_true, y_score, 0.9) == 0.0


def test_fixed_threshold_is_respected():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.8])
    # При пороге 0.5 предсказания = [0,0,1,1] → идеально.
    m = evaluate(y_true, y_score, threshold=0.5)
    assert m["threshold"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(1.0)


def test_pr_curve_figure_saved(tmp_path):
    from src.metrics import pr_curve_figure

    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.8])
    out = tmp_path / "pr.png"
    pr_curve_figure(y_true, y_score, str(out))
    assert out.exists() and out.stat().st_size > 0
