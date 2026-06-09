"""Система метрик для несбалансированной задачи (positive = illicit, ~2%).

Главные метрики — AUC-PR (average precision) и F1 по позитивному классу;
практическая — recall при фиксированной точности. Accuracy намеренно не
выносится в основной вывод.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(
    y_true,
    y_score,
    threshold: Optional[float] = None,
) -> dict:
    """Посчитать метрики качества для бинарной задачи.

    y_true: бинарные метки (1 = positive/illicit).
    y_score: вероятности (score) позитивного класса.
    threshold: порог бинаризации. Если None — подбирается по максимуму F1
        на ПЕРЕДАННЫХ данных. ВАЖНО: для теста порог фиксировать по валидации
        (передавать threshold, найденный на val), а не подбирать по тесту.

    Возвращает dict: auc_pr, roc_auc, f1, precision, recall,
    recall_at_precision_90, n_pos, n_neg, threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    auc_pr = float(average_precision_score(y_true, y_score))
    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        roc_auc = float("nan")  # один класс в y_true

    if threshold is None:
        threshold = _best_f1_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)

    return {
        "auc_pr": auc_pr,
        "roc_auc": roc_auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "recall_at_precision_90": recall_at_precision(y_true, y_score, 0.90),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "threshold": float(threshold),
    }


def _best_f1_threshold(y_true, y_score) -> float:
    """Подобрать порог, максимизирующий F1, по сетке precision_recall_curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall длиннее thresholds на 1; берём совпадающие позиции.
    p, r = precision[:-1], recall[:-1]
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def recall_at_precision(y_true, y_score, min_precision: float = 0.9) -> float:
    """Максимальный recall среди порогов, где precision >= min_precision.

    Если такой точности достичь нельзя — вернуть 0.0.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= min_precision
    if not mask.any():
        return 0.0
    return float(recall[mask].max())


def evaluate_per_group(
    y_true,
    y_score,
    group_labels,
    threshold: float,
    groups: Optional[list] = None,
) -> dict:
    """Метрики по типам паттернов: для каждого типа цепочки — насколько модель её ловит.

    y_true: бинарные метки (1 = laundering/illicit).
    y_score: вероятности позитивного класса.
    group_labels: для каждого объекта — строковый тип паттерна позитива
        (fan_out, fan_in, ..., stack) либо 'none' для негативов.
    threshold: единый порог (тот же, что для общей метрики — обычно фиксируется по val).
    groups: список интересующих паттернов; по умолчанию — все встретившиеся у позитивов.

    Логика: для каждого паттерна берём его позитивы vs ВСЕ негативы и считаем
    f1/precision/recall + сколько позитивов этого типа поймано. Главный сигнал —
    recall и n_detected/n_pos: какие цепочки модель видит, а какие пропускает.
    Возвращает {pattern: {f1, precision, recall, n_pos, n_detected}}.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    group_labels = np.asarray(group_labels, dtype=object)
    y_pred = (y_score >= threshold).astype(int)

    neg_mask = y_true == 0
    if groups is None:
        groups = sorted({g for g, t in zip(group_labels, y_true) if t == 1})

    result: dict = {}
    for g in groups:
        pos_mask = (y_true == 1) & (group_labels == g)
        n_pos = int(pos_mask.sum())
        if n_pos == 0:
            continue
        sub = pos_mask | neg_mask  # позитивы этого типа против всех негативов
        yt, yp = y_true[sub], y_pred[sub]
        result[g] = {
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "n_pos": n_pos,
            "n_detected": int(y_pred[pos_mask].sum()),
        }
    return result


def pr_curve_figure(y_true, y_score, path: str) -> None:
    """Построить и сохранить PR-кривую (с отметкой AUC-PR) в файл path."""
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AUC-PR = {auc_pr:.3f}")
    baseline = float((y_true == 1).mean())
    ax.axhline(baseline, ls="--", color="grey", label=f"baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
