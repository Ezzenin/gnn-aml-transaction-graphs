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

## GNN-линейка на Elliptic
Архитектуры (единый фактори-реестр `src/models.py`): **GCN, GraphSAGE, GAT, GIN, PNA**.
Тот же временно́й сплит, та же система метрик и фиксация порога по val, что у бейзлайна.

```bash
python -m src.train --config configs/elliptic_gat.yaml   # одна модель
python -m src.compare --run                              # все модели + сводная таблица/график
```
Сводка: `results/comparison.md` и `results/comparison_models.png`.

Результаты (test, AUC-PR): сильнейший — табличный **XGBoost (0.80)**; среди GNN лучший
**GraphSAGE (0.66)**. Все статические GNN заметно теряют на test относительно val —
известный эффект распределенческого сдвига Elliptic (поздние временны́е шаги). Это
мотивирует следующий этап: темпоральные модели (EvolveGCN) и Multi-GNN адаптации.

Строгий temporal-val (порог фиксируется на хронологически поздних train-узлах):
`configs/elliptic_gcn_temporal.yaml` (`temporal_val: true`).

## Логирование (Weights & Biases)
Единая точка — `src.utils.init_wandb`. По умолчанию **выключено**. Включение:
```bash
wandb login                                   # один раз (ключ с https://wandb.ai/authorize)
WANDB_MODE=online python -m src.train --config configs/elliptic_gcn.yaml
```
или через конфиг (`wandb.enabled: true`). Логируются `config`, per-epoch
`loss`/`val/auc_pr`/`val/f1`, финальные `test/*` и PR-кривая как `wandb.Image`.
Выключить: `wandb.enabled: false` (дефолт) или `WANDB_MODE=offline` (локальный дамп
в `wandb/`, без аккаунта).

## Автор
Ezzenin · eseninaleksandr@gmail.com
