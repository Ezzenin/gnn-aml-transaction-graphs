"""Загрузка чекпоинта и инференс edge-GNN — ЗАГЛУШКА (Фаза G1).

TODO(G1): вынести сюда общий инференс для Streamlit-продукта:
  - load_edge_model(checkpoint_path) -> (model, threshold, meta) с учётом
    in_node/in_edge/in_edge_label из чекпоинта (см. train_edge.py сохранение);
  - score_edges(model, data, edge_idx, context) -> вероятности (переиспользовать
    _eval_edges из train_edge или вынести общий код сюда).
Пока не реализовано — обучение/оценка делаются в src/train_edge.py.
"""
