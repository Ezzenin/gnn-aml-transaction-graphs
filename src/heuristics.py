"""Классические графовые эвристики для AML — интерпретируемый baseline (Фаза E, RQ3).

Неообучаемые структурные детекторы поверх транзакционного мультиграфа. Дают
per-edge «оценку подозрительности», порог фиксируется по val (как у моделей),
метрики (общая + per-pattern) считаются на test — чтобы эвристики честно встали
четвёртым семейством в сводку рядом с XGBoost / base GNN / Multi-GNN.

Детекторы:
  - fan-out: высокая out-степень счёта-отправителя (звезда расходящихся переводов);
  - fan-in:  высокая in-степень счёта-получателя (звезда сходящихся переводов);
  - reciprocity / 2-cycle: существует обратное ребро dst->src (простейший цикл,
    транзит «туда-обратно»), вклад в cycle/stack-типы.

Масштаб: глобальное перечисление простых циклов (nx.simple_cycles) на ~5M рёбер
неосуществимо, поэтому на полном графе детекторы векторные (numpy). NetworkX
используется в enumerate_simple_cycles — ограниченной утилите для малых графов
(демо/тесты/подграф в Streamlit), где simple_cycles до длины k уместен.

Запуск:
    python -m src.heuristics                 # полный IBM HI-Small
    python -m src.heuristics --max-rows 300000
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.datasets import CANONICAL_PATTERNS, load_ibm_aml
from src.metrics import evaluate, evaluate_per_group, pr_curve_figure
from src.utils import save_json


def degree_arrays(edge_index: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """in/out-степени узлов по всем рёбрам (numpy bincount)."""
    src, dst = edge_index[0], edge_index[1]
    out_deg = np.bincount(src, minlength=num_nodes).astype(np.float64)
    in_deg = np.bincount(dst, minlength=num_nodes).astype(np.float64)
    return in_deg, out_deg


def reciprocity_flag(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Для каждого ребра src->dst: 1.0, если существует обратное dst->src (2-cycle).

    Векторно через int64-ключи (src*N+dst) и np.isin — масштабируется на миллионы
    рёбер (в отличие от глобального перечисления циклов).
    """
    src, dst = edge_index[0].astype(np.int64), edge_index[1].astype(np.int64)
    keys = src * num_nodes + dst
    rev_keys = dst * num_nodes + src
    return np.isin(rev_keys, keys).astype(np.float64)


def heuristic_scores(edge_index: np.ndarray, num_nodes: int,
                     context_edge_index: "np.ndarray | None" = None) -> np.ndarray:
    """Комбинированная оценка подозрительности на ребро — степенные детекторы.

    Сумма нормированных log-степеней отправителя (fan-out) и получателя (fan-in):
    классический AML-сигнал «счёт-хаб». Реципрокность (2-cycle) НЕ включена — в
    этих данных она почти константна (~70% рёбер имеют обратное), т.е. не
    дискриминативна; оставлена отдельной утилитой reciprocity_flag.

    P1.5 (антиутечка): степени считаются по context_edge_index (по умолчанию —
    train-рёбра), а применяются к src/dst ВСЕХ оцениваемых рёбер — как и узловые
    признаки в load_ibm_aml (train-only). Абсолютный масштаб неважен — порог
    подбирается по val. Возвращает score [E].
    """
    ctx = edge_index if context_edge_index is None else context_edge_index
    in_deg, out_deg = degree_arrays(ctx, num_nodes)
    src, dst = edge_index[0], edge_index[1]
    fan_out = np.log1p(out_deg[src])   # отправитель «вещает» многим
    fan_in = np.log1p(in_deg[dst])     # получатель «собирает» со многих

    def _norm(a):
        rng = a.max() - a.min()
        return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)

    return _norm(fan_out) + _norm(fan_in)


def enumerate_simple_cycles(edge_index: np.ndarray, max_len: int = 6) -> list[list[int]]:
    """Простые циклы до длины max_len через NetworkX (для МАЛЫХ графов).

    Для полного IBM (~5M рёбер) неприменимо — используется в Streamlit-демо и
    тестах на подграфах, где цепочки-циклы локальны и коротки.
    """
    import networkx as nx

    g = nx.DiGraph()
    g.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    return [c for c in nx.simple_cycles(g, length_bound=max_len)]


def run(root: str = "data/ibm_aml", variant: str = "HI-Small", max_rows=None,
        output_dir: str = "results", output_name: str = "ibm_heuristics") -> dict:
    data, meta = load_ibm_aml(root=root, variant=variant, max_rows=max_rows)
    edge_index = data.edge_index.numpy()
    num_nodes = int(data.num_nodes)
    y = data.edge_label.numpy()
    patt = data.edge_pattern
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    print(f"[heuristics] IBM {variant}: {edge_index.shape[1]:,} рёбер, "
          f"illicit={meta['n_illicit']} ({meta['illicit_rate']*100:.3f}%)")
    # Степени — по train-контексту (антиутечка, P1.5).
    train_ctx_ei = edge_index[:, data.train_mask.numpy()]
    score = heuristic_scores(edge_index, num_nodes, context_edge_index=train_ctx_ei)
    recip_rate = reciprocity_flag(edge_index, num_nodes).mean()
    print(f"[heuristics] доля рёбер с обратным (2-cycle): {recip_rate*100:.2f}%")

    # Порог — по val (как у моделей), метрики — на test.
    val_metrics = evaluate(y[val_mask], score[val_mask], threshold=None)
    threshold = val_metrics["threshold"]
    test_metrics = evaluate(y[test_mask], score[test_mask], threshold=threshold)
    groups = list(CANONICAL_PATTERNS) + ["unknown"]
    per_pattern = evaluate_per_group(y[test_mask], score[test_mask], patt[test_mask], threshold, groups=groups)

    print(f"[VAL ] AUC-PR={val_metrics['auc_pr']:.4f} F1={val_metrics['f1']:.4f}")
    print(f"[TEST] AUC-PR={test_metrics['auc_pr']:.4f} F1={test_metrics['f1']:.4f} "
          f"R@P90={test_metrics['recall_at_precision_90']:.4f}")
    print("[per-pattern recall] " + ", ".join(
        f"{k}:{v['recall']:.2f}" for k, v in per_pattern.items()))

    result = {
        "model_type": "heuristics", "dataset_meta": meta,
        "fixed_threshold_from_val": threshold,
        "val_metrics": val_metrics, "test_metrics": test_metrics, "per_pattern": per_pattern,
    }
    metrics_path = os.path.join(output_dir, f"{output_name}_metrics.json")
    save_json(result, metrics_path)
    pr_curve_figure(y[test_mask], score[test_mask], os.path.join(output_dir, f"{output_name}_pr_curve.png"))
    print(f"[saved] {metrics_path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Графовые эвристики на IBM AML (интерпретируемый baseline)")
    p.add_argument("--root", default="data/ibm_aml")
    p.add_argument("--variant", default="HI-Small")
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()
    run(root=args.root, variant=args.variant, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
