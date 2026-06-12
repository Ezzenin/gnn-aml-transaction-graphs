"""Тесты строгого temporal-контекста edge-GNN (P0.1).

_context_data строит message-passing граф ТОЛЬКО из переданных рёбер. Проверяем,
что train-контекст не содержит val/test рёбер (антиутечка) и что reverse_mp
удваивает рёбра контекста.
"""
import numpy as np
import torch
from torch_geometric.data import Data

from src.train_edge import _context_data


def _fake_data():
    # 6 рёбер: первые 3 — train, 4-е — val, 5-6 — test (по edge_attr[:,0] видно индекс).
    ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    ea = torch.arange(6).float().view(-1, 1).repeat(1, 6)  # 6 фич, первая = индекс ребра
    d = Data(x=torch.randn(6, 5), edge_index=ei, edge_attr=ea)
    d.num_nodes = 6
    return d


def test_context_excludes_future_edges():
    d = _fake_data()
    train_mask = np.array([True, True, True, False, False, False])
    ctx = _context_data(d, train_mask, reverse_mp=False)
    # Контекст содержит ровно train-рёбра (индексы 0,1,2), без val/test (3,4,5).
    assert ctx.edge_index.shape[1] == 3
    assert set(ctx.edge_attr[:, 0].tolist()) == {0.0, 1.0, 2.0}


def test_context_trainval():
    d = _fake_data()
    mask = np.array([True, True, True, True, False, False])  # train+val
    ctx = _context_data(d, mask, reverse_mp=False)
    assert ctx.edge_index.shape[1] == 4
    assert 5.0 not in set(ctx.edge_attr[:, 0].tolist())  # test-ребро (idx 5) исключено


def test_context_reverse_doubles():
    d = _fake_data()
    train_mask = np.array([True, True, True, False, False, False])
    ctx = _context_data(d, train_mask, reverse_mp=True)
    assert ctx.edge_index.shape[1] == 6        # 3 train × 2 (прямые+обратные)
    assert ctx.edge_attr.shape[1] == 7         # +флаг направления
