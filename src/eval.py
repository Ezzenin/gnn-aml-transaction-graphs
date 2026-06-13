"""Загрузка чекпоинта edge-GNN и инференс — общий слой для Streamlit-продукта (G1).

Чекпоинт-агностичен: архитектура восстанавливается из метаданных чекпоинта
(in_node/in_edge/in_edge_label + флаги адаптаций из config), поэтому одинаково
грузит base GINe и Multi-GNN. score_edges повторяет протокол оценки из train_edge
(строгий контекст train+val для test-рёбер, признаки сид-ребра через input_id).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader

from src.models import build_edge_model
from src.train_edge import _context_data
from src.utils import resolve_device


def load_edge_model(checkpoint_path: str, device=None):
    """Загрузить чекпоинт → (model в eval-режиме, ckpt-dict с метаданными).

    ckpt содержит: state_dict, config, in_node, in_edge, in_edge_label, threshold.
    """
    device = device or resolve_device("auto")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mp = (ckpt.get("config", {}).get("model", {}) or {}).get("params", {})
    model = build_edge_model(
        ckpt.get("config", {}).get("model", {}).get("type", "gine"),
        in_node=ckpt["in_node"], in_edge=ckpt["in_edge"],
        in_edge_label=ckpt.get("in_edge_label"),
        hidden=int(mp.get("hidden", 64)), num_layers=int(mp.get("num_layers", 2)),
        dropout=float(mp.get("dropout", 0.5)),
        reverse_mp=bool(mp.get("reverse_mp", False)),
        ports=bool(mp.get("ports", False)),
        ego_ids=bool(mp.get("ego_ids", False)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    ckpt["device"] = device
    ckpt["reverse_mp"] = bool(mp.get("reverse_mp", False))
    return model, ckpt


def score_edges(model, ckpt, data, edge_idx, num_neighbors=(10, 10), batch_size=2048):
    """Вернуть вероятности laundering для рёбер edge_idx (порядок сохранён).

    Контекст message passing = train+val рёбра (тот же протокол, что для test в
    train_edge). reverse_mp берётся из чекпоинта (контекст делается двунаправленным).
    """
    device = ckpt["device"]
    tr = data.train_mask.numpy()
    va = data.val_mask.numpy()
    ctx = _context_data(data, tr | va, ckpt["reverse_mp"])
    seed_attr = data.edge_attr[edge_idx].float()
    loader = LinkNeighborLoader(
        ctx, num_neighbors=list(num_neighbors),
        edge_label_index=data.edge_index[:, edge_idx],
        edge_label=data.edge_label[edge_idx],
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ela = seed_attr[batch.input_id.cpu()].to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, ela)
            scores.append(F.softmax(out, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores) if scores else np.array([])


def sample_test_edges(data, n_neg=2000, seed=42):
    """Сэмпл test-рёбер для демо: все illicit + n_neg случайных легитимных.

    Илличит-рёбер мало (~0.1%), поэтому берём все позитивы + ограниченный фон
    негативов для отзывчивого UI. Возвращает индексы в исходном edge_index.
    """
    rng = np.random.default_rng(seed)
    te = np.flatnonzero(data.test_mask.numpy())
    y = data.edge_label.numpy()
    pos = te[y[te] == 1]
    neg = te[y[te] == 0]
    neg = rng.choice(neg, size=min(len(neg), n_neg), replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return idx
