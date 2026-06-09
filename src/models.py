"""Модели GNN для node-classification на Elliptic.

Линейка базовых архитектур: GCN, GraphSAGE, GAT, GIN, PNA. Все — 2-слойные
message-passing сети с бинарным выходом (out_channels=2: licit/illicit), обучаются
только на размеченных узлах через маску в функции потерь. Сборка через build_model().

Следующий этап: Multi-GNN адаптации (ego-IDs, port numbering, reverse MP) поверх
этих архитектур.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv, PNAConv, SAGEConv


class GCN(torch.nn.Module):
    """2-слойный GCN."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGE(torch.nn.Module):
    """2-слойный GraphSAGE (агрегация по соседям)."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GAT(torch.nn.Module):
    """2-слойный GAT (multi-head attention в первом слое)."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GIN(torch.nn.Module):
    """2-слойный GIN (MLP-агрегатор + BatchNorm)."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5):
        super().__init__()
        self.conv1 = GINConv(_mlp(in_channels, hidden_channels), train_eps=True)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = GINConv(_mlp(hidden_channels, hidden_channels), train_eps=True)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


class PNA(torch.nn.Module):
    """2-слойный PNA (несколько агрегаторов и скейлеров; требует deg-гистограмму)."""

    AGGREGATORS = ["mean", "min", "max", "std"]
    SCALERS = ["identity", "amplification", "attenuation"]

    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5, deg=None):
        super().__init__()
        if deg is None:
            raise ValueError("PNA требует deg (гистограмму степеней обучающего графа)")
        common = dict(aggregators=self.AGGREGATORS, scalers=self.SCALERS, deg=deg, towers=1)
        self.conv1 = PNAConv(in_channels, hidden_channels, **common)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = PNAConv(hidden_channels, hidden_channels, **common)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


def _mlp(in_dim: int, out_dim: int) -> Sequential:
    return Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))


# ─────────────────────── Edge-classification (IBM AML) ───────────────────────
class EdgeGNN(torch.nn.Module):
    """Message-passing с учётом edge_attr + голова edge-классификации (GINe-база).

    Кодировщики узлов/рёбер → стек GINEConv (учитывает edge_attr) → голова
    MLP([h_u || h_v]) -> 2 логита для каждого классифицируемого ребра
    (edge_label_index). Флаги адаптаций reverse_mp/ports/ego_ids зарезервированы
    для Фазы D (в C2 выключены): ego-IDs помечает конечные узлы сид-ребра.
    """

    def __init__(self, in_node, in_edge, hidden=64, num_layers=2, dropout=0.5,
                 reverse_mp=False, ports=False, ego_ids=False):
        super().__init__()
        self.reverse_mp = reverse_mp
        self.ports = ports
        self.ego_ids = ego_ids
        self.dropout = dropout

        node_in = in_node + (1 if ego_ids else 0)
        edge_in = in_edge + (1 if reverse_mp else 0) + (1 if ports else 0)
        self.node_enc = Linear(node_in, hidden)
        self.edge_enc = Linear(edge_in, hidden)

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINEConv(_mlp(hidden, hidden), edge_dim=hidden, train_eps=True))
            self.bns.append(BatchNorm1d(hidden))

        self.head = Sequential(
            Linear(2 * hidden, hidden), ReLU(),
            torch.nn.Dropout(dropout), Linear(hidden, 2),
        )

    def forward(self, x, edge_index, edge_attr, edge_label_index):
        h = self.node_enc(x)
        e = self.edge_enc(edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, edge_index, e)))
            h = F.dropout(h, p=self.dropout, training=self.training)
        h_u = h[edge_label_index[0]]
        h_v = h[edge_label_index[1]]
        return self.head(torch.cat([h_u, h_v], dim=-1))


def build_edge_model(name: str, in_node: int, in_edge: int, hidden: int = 64,
                     num_layers: int = 2, dropout: float = 0.5,
                     reverse_mp: bool = False, ports: bool = False,
                     ego_ids: bool = False) -> torch.nn.Module:
    """Фабрика edge-моделей. name='gine' (C2); адаптации — флагами (Фаза D)."""
    name = name.lower()
    if name in ("gine", "gin", "edgegnn"):
        return EdgeGNN(in_node, in_edge, hidden, num_layers, dropout,
                       reverse_mp=reverse_mp, ports=ports, ego_ids=ego_ids)
    raise ValueError(f"Неизвестная edge-архитектура: {name!r} (gine)")


def build_model(
    name: str,
    in_channels: int,
    hidden_channels: int = 64,
    out_channels: int = 2,
    dropout: float = 0.5,
    heads: int = 8,
    deg: Optional["torch.Tensor"] = None,
) -> torch.nn.Module:
    """Фабрика моделей по имени архитектуры (gcn|sage|gat|gin|pna)."""
    name = name.lower()
    if name == "gcn":
        return GCN(in_channels, hidden_channels, out_channels, dropout)
    if name in ("sage", "graphsage"):
        return GraphSAGE(in_channels, hidden_channels, out_channels, dropout)
    if name == "gat":
        return GAT(in_channels, hidden_channels, out_channels, dropout, heads=heads)
    if name == "gin":
        return GIN(in_channels, hidden_channels, out_channels, dropout)
    if name == "pna":
        return PNA(in_channels, hidden_channels, out_channels, dropout, deg=deg)
    raise ValueError(f"Неизвестная архитектура: {name!r} (gcn|sage|gat|gin|pna)")


def compute_degree_histogram(edge_index, num_nodes: int) -> "torch.Tensor":
    """Гистограмма входных степеней — нужна PNA-агрегаторам."""
    from torch_geometric.utils import degree

    d = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
    return torch.bincount(d)
