"""Загрузчик датасета Elliptic (Bitcoin) через PyTorch Geometric.

Elliptic: node-classification, размеченные узлы illicit/licit + большой пласт
unknown. Сплит train/test — встроенный временно́й (ранние шаги → train, поздние
→ test). positive_class (illicit) определяется программно как миноритарный
размеченный класс — не хардкодим целое значение метки.
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import torch
from torch_geometric.datasets import EllipticBitcoinDataset

# Ожидаемые размеры PyG-версии Elliptic (для предупреждений, не для падения).
EXPECTED = {"num_nodes": 203769, "num_edges": 234355, "num_features": 165}


def load_elliptic(root: str = "data/elliptic") -> Tuple["torch.Tensor", dict]:
    """Загрузить Elliptic и вернуть (data, meta).

    data — единый граф PyG: data.x, data.edge_index, data.y,
           data.train_mask, data.test_mask.
    meta — dict со статистикой: num_nodes, num_edges, num_features,
           label_counts (по всем узлам), positive_class (illicit, миноритарный),
           размеры и баланс классов в train/test.
    """
    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]

    # Размеченные узлы = объединение train/test масок (unknown исключён масками).
    labeled_mask = data.train_mask | data.test_mask
    vals, counts = torch.unique(data.y[labeled_mask], return_counts=True)
    # positive = illicit = размеченный класс с наименьшим числом узлов.
    positive_class = int(vals[torch.argmin(counts)].item())

    all_vals, all_counts = torch.unique(data.y, return_counts=True)
    label_counts = {int(v): int(c) for v, c in zip(all_vals.tolist(), all_counts.tolist())}

    meta = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(data.num_features),
        "label_counts": label_counts,
        "positive_class": positive_class,
        "n_labeled": int(labeled_mask.sum()),
        "splits": {
            "train": _split_stats(data, data.train_mask, positive_class),
            "test": _split_stats(data, data.test_mask, positive_class),
        },
    }

    _warn_on_mismatch(meta)
    return data, meta


def _split_stats(data, mask, positive_class: int) -> dict:
    """Статистика по сплиту: размер, число pos/neg, доля позитивного класса."""
    y = data.y[mask]
    n = int(mask.sum())
    n_pos = int((y == positive_class).sum())
    n_neg = n - n_pos
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": (n_pos / n) if n else 0.0,
    }


def _warn_on_mismatch(meta: dict) -> None:
    """Предупредить, если размеры отличаются от ожидаемых (другая версия PyG)."""
    for key, expected in EXPECTED.items():
        actual = meta[key]
        if actual != expected:
            warnings.warn(
                f"Elliptic {key}={actual}, ожидалось {expected}. "
                f"Возможно, иная версия PyG/датасета — проверьте на рантайме.",
                stacklevel=2,
            )


def get_xy(data, mask, positive_class: int):
    """Признаки и бинарные метки (1 = positive/illicit) для узлов под маской."""
    X = data.x[mask].cpu().numpy()
    y = (data.y[mask] == positive_class).long().cpu().numpy()
    return X, y


def make_val_split(y_binary, val_fraction: float = 0.2, seed: int = 42):
    """Стратифицированно разбить позиции [0..n) на train/val по меткам y_binary.

    Возвращает (train_pos, val_pos) — массивы позиций внутри переданной выборки.
    Случайный сплит делается ТОЛЬКО внутри train-узлов (временна́я утечка
    исключена: train-узлы — это ранние временны́е шаги).
    """
    from sklearn.model_selection import train_test_split

    positions = np.arange(len(y_binary))
    train_pos, val_pos = train_test_split(
        positions,
        test_size=val_fraction,
        stratify=y_binary,
        random_state=seed,
    )
    return train_pos, val_pos


if __name__ == "__main__":
    # Быстрая ручная проверка структуры датасета.
    warnings.filterwarnings("ignore")
    data, meta = load_elliptic()
    print(data)
    import json

    print(json.dumps(meta, indent=2, ensure_ascii=False))
