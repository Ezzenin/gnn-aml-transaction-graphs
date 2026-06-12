"""Тесты графовых эвристик (Фаза E): степени, реципрокность, циклы."""
import numpy as np

from src.heuristics import (
    degree_arrays,
    enumerate_simple_cycles,
    heuristic_scores,
    reciprocity_flag,
)


def test_degree_arrays():
    # 0->1, 0->2, 1->2 : out(0)=2,out(1)=1; in(1)=1,in(2)=2.
    ei = np.array([[0, 0, 1], [1, 2, 2]])
    in_deg, out_deg = degree_arrays(ei, num_nodes=3)
    assert out_deg.tolist() == [2, 1, 0]
    assert in_deg.tolist() == [0, 1, 2]


def test_reciprocity_flag():
    # Рёбра: 0->1 (есть обратное 1->0), 1->0, 2->3 (нет обратного).
    ei = np.array([[0, 1, 2], [1, 0, 3]])
    recip = reciprocity_flag(ei, num_nodes=4)
    assert recip.tolist() == [1.0, 1.0, 0.0]


def test_heuristic_scores_high_for_hub():
    # Узел 0 — отправитель-хаб (3 исходящих); ребро 0->x должно набрать больше,
    # чем периферийное ребро 4->5.
    ei = np.array([[0, 0, 0, 4], [1, 2, 3, 5]])
    s = heuristic_scores(ei, num_nodes=6)
    assert s[0] > s[3]  # ребро из хаба подозрительнее периферийного


def test_enumerate_simple_cycles_small():
    # Цикл 0->1->2->0 + хвост 2->3.
    ei = np.array([[0, 1, 2, 2], [1, 2, 0, 3]])
    cycles = enumerate_simple_cycles(ei, max_len=6)
    assert any(set(c) == {0, 1, 2} for c in cycles)
