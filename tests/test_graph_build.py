"""Smoke-тесты построения графа IBM AML (Фаза B).

Главная проверка — байт-в-байт согласованность ключа ребра между
parse_ibm_patterns и load_ibm_aml: если бы ключи расходились, привязка паттернов
терялась бы молча. Тест строит мини-CSV из реальных строк Patterns.txt, поэтому
все его рёбра illicit и ДОЛЖНЫ получить канонический паттерн (coverage 100%).
Полный датасет не грузим. Если данных нет — тест скипается (CI-дружелюбно).
"""
import os

import pytest

from src.datasets import (
    CANONICAL_PATTERNS,
    IBM_COLS,
    load_ibm_aml,
    parse_ibm_patterns,
    _ibm_edge_key,
    _normalize_pattern,
)

ROOT = "data/ibm_aml"
TRANS = os.path.join(ROOT, "HI-Small_Trans.csv")
PATTERNS = os.path.join(ROOT, "HI-Small_Patterns.txt")
HAVE_DATA = os.path.exists(TRANS) and os.path.exists(PATTERNS)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="IBM AML данные не скачаны")


def test_normalize_pattern_canonical():
    assert _normalize_pattern("FAN-OUT:  Max 16-degree Fan-Out") == "fan_out"
    assert _normalize_pattern("SCATTER-GATHER") == "scatter_gather"
    assert _normalize_pattern("GATHER-SCATTER:  Max 2-degree Fan-In") == "gather_scatter"
    assert _normalize_pattern("CYCLE:  Max 10 hops") == "cycle"
    assert _normalize_pattern("STACK") == "stack"


def test_edge_key_uses_first_10_fields():
    fields = "2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1".split(",")
    key = _ibm_edge_key(fields)
    parts = key.split(",")
    assert len(parts) == 10          # Is Laundering (11-е поле) исключён
    assert parts[-1] == "ACH"        # последнее поле ключа = Payment Format
    assert parts[0] == "2022/09/01 00:06"


@needs_data
def test_patterns_parse_nonempty():
    m = parse_ibm_patterns(PATTERNS)
    assert len(m) > 0
    assert set(m.values()) <= set(CANONICAL_PATTERNS)


@needs_data
def test_key_consistency_on_pattern_rows(tmp_path):
    # Берём первые ~80 транзакционных строк из Patterns.txt → мини-CSV (все illicit).
    rows = []
    with open(PATTERNS, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(("BEGIN", "END")) or not line:
                continue
            rows.append(line)
            if len(rows) >= 80:
                break

    mini = tmp_path / "HI-Small_Trans.csv"
    mini.write_text(",".join(IBM_COLS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    # Patterns.txt должен лежать рядом, чтобы load_ibm_aml его нашёл.
    import shutil

    shutil.copy(PATTERNS, tmp_path / "HI-Small_Patterns.txt")

    data, meta = load_ibm_aml(root=str(tmp_path), variant="HI-Small")

    # Базовая структура.
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] == len(rows)
    assert data.edge_attr.shape == (len(rows), meta["num_edge_features"])
    assert data.x.shape[1] == meta["num_node_features"]
    assert data.train_mask.numel() == len(rows)

    # Все строки illicit → у всех должен быть КАНОНИЧЕСКИЙ паттерн (ключи совпали).
    assert int(data.edge_label.sum()) == len(rows)
    assert meta["patterns_matched"] == len(rows), "ключи ребра и паттерна разошлись!"
    assert all(p in CANONICAL_PATTERNS for p in data.edge_pattern)
