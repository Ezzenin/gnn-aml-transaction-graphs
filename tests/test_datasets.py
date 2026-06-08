"""Тесты загрузчика Elliptic.

Загрузка требует скачанного датасета (PyG). Если данных нет и сети нет —
тест скипается, чтобы pytest не падал и не лез в сеть в CI.
"""
import os

import pytest

pytest.importorskip("torch_geometric")

ROOT = os.environ.get("ELLIPTIC_ROOT", "data/elliptic")
_HAS_DATA = os.path.exists(os.path.join(ROOT, "processed", "data.pt")) or os.path.exists(
    os.path.join(ROOT, "elliptic_bitcoin_dataset", "elliptic_txs_features.csv")
)


@pytest.mark.skipif(not _HAS_DATA, reason="Elliptic не скачан (нет data/elliptic)")
def test_load_elliptic_structure():
    from src.datasets import load_elliptic

    data, meta = load_elliptic(root=ROOT)

    # Базовая структура графа.
    assert data.x.shape[0] == data.num_nodes
    assert data.edge_index.shape[0] == 2
    assert hasattr(data, "train_mask") and hasattr(data, "test_mask")

    # positive_class — миноритарный размеченный класс (illicit).
    pos = meta["positive_class"]
    tr = meta["splits"]["train"]
    assert tr["n_pos"] < tr["n_neg"], "позитив должен быть миноритарным в train"
    assert 0.0 < tr["pos_rate"] < 0.5

    # Число признаков фиксируется на рантайме (PyG-версия = 165).
    assert meta["num_features"] == data.num_features

    # Маски не пересекаются.
    overlap = int((data.train_mask & data.test_mask).sum())
    assert overlap == 0


@pytest.mark.skipif(not _HAS_DATA, reason="Elliptic не скачан (нет data/elliptic)")
def test_make_val_split_is_stratified():
    import numpy as np

    from src.datasets import make_val_split

    y = np.array([0] * 80 + [1] * 20)
    tr_pos, va_pos = make_val_split(y, val_fraction=0.2, seed=42)
    assert len(tr_pos) + len(va_pos) == len(y)
    assert len(set(tr_pos) & set(va_pos)) == 0
    # Стратификация: доля позитивов в val близка к общей (0.2).
    val_pos_rate = y[va_pos].mean()
    assert val_pos_rate == pytest.approx(0.2, abs=0.05)
