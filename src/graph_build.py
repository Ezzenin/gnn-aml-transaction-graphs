"""Построение ego-подграфов вокруг подозрительного ребра для визуализации (G1).

ego_subgraph извлекает k-hop окрестность вокруг классифицируемой транзакции
(узлы = счета, рёбра = транзакции) для подсветки цепочки в Streamlit. detect_chain
помечает структурные признаки (реципрокность/короткий цикл, fan-in/out) — простая
интерпретируемая «почему подозрительно», без GNNExplainer (E+/stretch).

(line-graph «узел = транзакция» для RQ1 — отдельная stretch-задача F1, не здесь.)
"""
from __future__ import annotations

import numpy as np


def ego_subgraph(data, center_edge_idx: int, scores=None, score_idx=None,
                 num_hops: int = 1, max_edges: int = 150) -> dict:
    """k-hop ego-подграф вокруг ребра center_edge_idx (по обоим направлениям).

    Возвращает dict для визуализации:
      nodes: список id счетов (int);
      edges: [{u, v, amount, label, pattern, score, is_center}];
      center: (u, v).
    scores/score_idx — необязательная карта score по индексам рёбер (для подсветки).
    Расширение ограничено max_edges (отзывчивость UI на полном графе ~5M рёбер).
    """
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    u, v = int(src[center_edge_idx]), int(dst[center_edge_idx])

    nodes = {u, v}
    frontier = {u, v}
    for _ in range(num_hops):
        fr = np.fromiter(frontier, dtype=np.int64)
        mask = np.isin(src, fr) | np.isin(dst, fr)
        e = np.flatnonzero(mask)
        nbr = set(src[e].tolist()) | set(dst[e].tolist())
        frontier = nbr - nodes
        nodes |= nbr
        if not frontier:
            break

    nodes_arr = np.fromiter(nodes, dtype=np.int64)
    in_sub = np.isin(src, nodes_arr) & np.isin(dst, nodes_arr)
    eidx = np.flatnonzero(in_sub)
    # Центр всегда включаем; остальное обрезаем до max_edges.
    eidx = np.concatenate([[center_edge_idx], eidx[eidx != center_edge_idx]])[:max_edges]

    amt = np.expm1(data.edge_attr[:, 0].numpy())  # log1p(Amount Paid) → сумма
    label = data.edge_label.numpy()
    pattern = data.edge_pattern
    score_map = {}
    if scores is not None and score_idx is not None:
        score_map = {int(i): float(s) for i, s in zip(score_idx, scores)}

    edges = []
    for i in eidx.tolist():
        edges.append({
            "u": int(src[i]), "v": int(dst[i]),
            "amount": round(float(amt[i]), 2),
            "label": int(label[i]),
            "pattern": str(pattern[i]),
            "score": score_map.get(int(i)),
            "is_center": (i == center_edge_idx),
        })
    used_nodes = sorted({e["u"] for e in edges} | {e["v"] for e in edges})
    return {"nodes": used_nodes, "edges": edges, "center": (u, v)}


def detect_chain(data, center_edge_idx: int) -> dict:
    """Простые структурные признаки ребра (интерпретируемое «почему подозрительно»).

    reciprocal: есть ли обратное ребро v->u (2-cycle/транзит);
    fan_out: out-степень отправителя; fan_in: in-степень получателя;
    pattern: истинный тип паттерна (если размечен).
    """
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    u, v = int(src[center_edge_idx]), int(dst[center_edge_idx])
    reciprocal = bool(np.any((src == v) & (dst == u)))
    fan_out = int(np.count_nonzero(src == u))
    fan_in = int(np.count_nonzero(dst == v))
    return {
        "reciprocal": reciprocal,
        "fan_out": fan_out,
        "fan_in": fan_in,
        "pattern": str(data.edge_pattern[center_edge_idx]),
        "label": int(data.edge_label.numpy()[center_edge_idx]),
    }
