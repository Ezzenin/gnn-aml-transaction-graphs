"""Модели GNN. Этап 1: минимальный 2-слойный GCN для sanity-проверки PyG-конвейера.

На следующих этапах сюда добавятся GraphSAGE/GAT/GIN/PNA и Multi-GNN адаптации
(ego-IDs, port numbering, reverse message passing).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    """Простой 2-слойный GCN для node-classification.

    Бинарная задача → out_channels=2 (логиты licit/illicit); обучаем только
    на размеченных узлах через маску в функции потерь.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
