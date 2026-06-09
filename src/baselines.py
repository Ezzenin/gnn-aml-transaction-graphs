"""Неграфовые бейзлайны на признаках узлов Elliptic: XGBoost и LogReg.

Воспроизведение табличного подхода Weber et al. (2019): деревья/линейные
модели на признаках узла без использования структуры графа.
"""
from __future__ import annotations

import numpy as np


def _scale_pos_weight(y_train) -> float:
    """scale_pos_weight ≈ n_neg / n_pos для борьбы с дисбалансом."""
    y = np.asarray(y_train)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return (n_neg / n_pos) if n_pos else 1.0


def train_xgboost(X_train, y_train, X_val, y_val, params: dict):
    """Обучить XGBClassifier с учётом дисбаланса и ранней остановкой по val.

    params — гиперпараметры (n_estimators, max_depth, learning_rate, ...).
    Возвращает обученную модель (predict_proba(X)[:, 1] = score позитива).
    """
    from xgboost import XGBClassifier

    params = dict(params or {})
    params.setdefault("scale_pos_weight", _scale_pos_weight(y_train))
    early_stopping_rounds = params.pop("early_stopping_rounds", 30)
    # n_jobs=1 по умолчанию (детерминизм на Elliptic); на больших IBM можно >1.
    n_jobs = params.pop("n_jobs", 1)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=params.pop("random_state", 42),
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_logreg(X_train, y_train, params: dict):
    """Обучить логистическую регрессию (со стандартизацией) на дисбалансе.

    class_weight='balanced' компенсирует редкий позитивный класс.
    Возвращает Pipeline (StandardScaler → LogisticRegression).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    params = dict(params or {})
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=params.pop("max_iter", 1000),
        C=params.pop("C", 1.0),
        n_jobs=params.pop("n_jobs", None),
        random_state=params.pop("random_state", 42),
        **params,
    )
    model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    model.fit(X_train, y_train)
    return model


def predict_scores(model, X) -> "np.ndarray":
    """Вернуть вероятности позитивного класса для XGBoost/LogReg-пайплайна."""
    return model.predict_proba(X)[:, 1]
