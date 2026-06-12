"""Построение под-графов для продукта/анализа — ЗАГЛУШКА (Фаза G1; line-graph — stretch F1).

TODO(G1): извлечение ego-подграфа вокруг подозрительного ребра для визуализации
  (узлы-счета + рёбра-транзакции, k-hop окрестность).
TODO(F1, stretch): line-graph «узел = транзакция» для RQ1 (node-classification
  альтернатива edge-classification из load_ibm_aml).
Account-граф «узел=счёт, ребро=транзакция» сейчас строится в src/datasets.load_ibm_aml.
"""
