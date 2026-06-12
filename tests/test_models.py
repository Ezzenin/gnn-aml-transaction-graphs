"""Тесты edge-моделей и Multi-GNN адаптаций (Фаза D).

compute_ports — различение кратных рёбер мультиграфа; EdgeGNN.forward с флагами
reverse/port/ego должен отрабатывать без ошибок формы и давать 2 логита на ребро.
"""
import torch

from src.models import EdgeGNN, build_edge_model, compute_ports


def test_compute_ports_parallel_edges():
    # 3 параллельных ребра в узел 2 + одно в узел 0. Порты локальны для dst:
    # рёбра в узел 2 → 0,1,2 (в каком-то порядке, но различные); в узел 0 → 0.
    edge_index = torch.tensor([[0, 1, 3, 4],
                               [2, 2, 2, 0]])
    ports = compute_ports(edge_index, num_nodes=5)
    # Узел 2: три входящих ребра (индексы 0,1,2) должны получить {0,1,2}.
    assert sorted(ports[:3].tolist()) == [0, 1, 2]
    # Узел 0: единственное входящее ребро → порт 0.
    assert ports[3].item() == 0


def test_compute_ports_empty():
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    assert compute_ports(edge_index, num_nodes=0).numel() == 0


def test_compute_ports_single_per_node():
    # Каждый узел-получатель уникален → все порты 0.
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    assert compute_ports(edge_index, num_nodes=4).tolist() == [0, 0, 0]


def _toy_batch(in_node=5, in_edge=6, n_nodes=8, n_edges=12):
    torch.manual_seed(0)
    x = torch.randn(n_nodes, in_node)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, in_edge)
    edge_label_index = torch.randint(0, n_nodes, (2, 4))  # 4 классифицируемых ребра
    return x, edge_index, edge_attr, edge_label_index


def test_edge_gnn_base_forward_shape():
    x, ei, ea, eli = _toy_batch()
    model = build_edge_model("gine", in_node=5, in_edge=6, hidden=16)
    out = model(x, ei, ea, eli)
    assert out.shape == (4, 2)  # 4 ребра × 2 логита


def test_edge_gnn_all_adaptations_forward_shape():
    # Все три адаптации включены (полный Multi-GNN путь) — самый нагруженный.
    x, ei, ea, eli = _toy_batch()
    model = build_edge_model("gine", in_node=5, in_edge=6, hidden=16,
                             reverse_mp=True, ports=True, ego_ids=True)
    out = model(x, ei, ea, eli)
    assert out.shape == (4, 2)


def test_edge_gnn_each_adaptation_independently():
    x, ei, ea, eli = _toy_batch()
    for flags in [dict(reverse_mp=True), dict(ports=True), dict(ego_ids=True)]:
        model = EdgeGNN(in_node=5, in_edge=6, hidden=16, **flags)
        out = model(x, ei, ea, eli)
        assert out.shape == (4, 2), f"сломалось на {flags}"
