# GNN для выявления цепочек финансовых операций

Курсовая работа. НИУ ВШЭ, ФКН, магистерская программа «Финансовые технологии и анализ данных» (01.04.02).

## Тема
Исследование методов построения графовых представлений транзакционных данных
и применения графовых нейронных сетей (GNN) для выявления цепочек финансовых
операций (AML / антифрод).

«Цепочки» формализуются как типовые **паттерны отмывания** (8 паттернов AMLSim:
fan-out, fan-in, gather-scatter, scatter-gather, simple cycle, random, bipartite, stack).

## Исследовательские вопросы
1. Что лучше для детекции цепочек: представление «узел = счёт, ребро = транзакция»
   (edge-classification) или «узел = транзакция» (node-classification)?
2. Сколько дают мультиграфовые адаптации (направленность, reverse message passing,
   временной порядок рёбер) поверх базовых GNN?
3. Где граница: на каких паттернах GNN уверенно бьёт XGBoost и классические эвристики?

## Датасеты
- [Elliptic (Bitcoin)](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) —
  node-classification, встроен в PyTorch Geometric (`EllipticBitcoinDataset`).
- [IBM AML HI-Small](https://github.com/IBM/AMLSim) — edge-classification, разметка по 8 паттернам.

## Структура
```
data/          # датасеты (в .gitignore; PyG скачивает Elliptic сюда)
src/           # модели, граф-билдеры, обучение, метрики, бейзлайны
app/           # Streamlit-демо (два режима: AML + personal finance)
notebooks/     # исследовательские ноутбуки, EDA, графики
configs/       # YAML-конфиги экспериментов
tests/         # юнит-тесты (граф-билдеры, метрики)
docs/          # постановка задачи, обзор литературы
results/       # метрики (json) + графики (png)
```

## Метрики
Из-за дисбаланса (~2% illicit) основные метрики — **AUC-PR** и **F1 по позитивному
классу**, практическая — **recall @ fixed precision**. Accuracy не используется как
отчётная метрика.

## Стек
Python · PyTorch · PyTorch Geometric · XGBoost / LightGBM · scikit-learn ·
NetworkX · Streamlit · Weights & Biases

## Запуск (Этап 1 — Elliptic + неграфовый бейзлайн)
```bash
pip install -r requirements.txt
pytest -q
python -m src.train_baseline --config configs/elliptic_xgb.yaml
```

## Автор
Ezzenin · eseninaleksandr@gmail.com
