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
app/           # Streamlit-демо: AML / антифрод (personal finance — в перспективах)
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

## Данные IBM AML (HI-Small) — для Этапа 2
Датасет: Kaggle [`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
Скачать **только** вариант **HI-Small** (НЕ Medium/Large) — два файла:
```
data/ibm_aml/HI-Small_Trans.csv      # ~5.08M транзакций, метка Is Laundering
data/ibm_aml/HI-Small_Patterns.txt   # инстансы 8 паттернов отмывания
```
Данные не коммитятся (`data/` в `.gitignore`). Загрузчик — `src.datasets.load_ibm_aml`:
направленный мультиграф «узел = счёт, ребро = транзакция», edge-classification,
строгий **temporal split** (train=ранние t, val=поздняя доля train, test=поздние t),
узловые признаки (degree + log-суммы) считаются только по train-рёбрам (антиутечка).
Привязка позитивов к типу паттерна — `src.datasets.parse_ibm_patterns`.
Реальная статистика: 515k счетов, 5.08M рёбер, illicit-доля ~0.10%.

## Воспроизведение Этапа 2 (IBM AML)

Полный пайплайн. Обучение GNN тяжёлое — рекомендуется **CUDA**; на CPU/MPS работает,
но медленно (любой `device: cuda` сам откатывается на cpu, см. `resolve_device`).
XGBoost, эвристики, сборка сводок и Streamlit — на CPU. Для быстрой отладки —
`dataset.max_rows` в конфиге.

```bash
# 0. Данные — см. раздел выше (data/ibm_aml/HI-Small_*).

# 1. Бейзлайны (XGBoost на рёбрах). CPU.
python -m src.train_baseline --config configs/ibm_xgb.yaml          # с norm_time
python -m src.train_baseline --config configs/ibm_xgb_notime.yaml   # без времени (лучший)
python -m src.train_baseline --config configs/ibm_xgb_fan.yaml      # + fan GF-фичи

# 2. Ablation Multi-GNN адаптаций (RQ2). CUDA. Три режима признаков/обучения:
for c in ibm_gine ibm_gine_rev ibm_gine_port ibm_gine_ego ibm_multignn; do
  python -m src.train_edge --config configs/$c.yaml                 # с временем
  python -m src.train_edge --config configs/${c}_notime.yaml        # без времени
done
python -m src.train_edge --config configs/ibm_gine_fulldata.yaml    # обучение на всех train-рёбрах
python -m src.train_edge --config configs/ibm_multignn_fulldata.yaml

# 3. Классические эвристики (RQ3, интерпретируемый baseline). CPU.
python -m src.heuristics

# 4. Сводки и графики (CPU): ablation×3 режима + per-pattern + reference-блок литературы.
python -m src.compare --ibm        # → results/ibm_comparison*.md, ablation*.png, per_pattern.*
#   (--run-ibm дополнительно прогоняет всю сетку перед сборкой)

# 5. Продукт (режим антифрод). CPU.
streamlit run app/streamlit_app.py
```

Результаты — в `results/` (`ibm_comparison*.md`/`.png`, `ablation*.png`,
`per_pattern.*`). Числа и выводы (RQ2/RQ3) и сравнение с литературой —
`docs/lit_benchmarks.md`, `docs/lit_review.md`. Данные и чекпоинты в `.gitignore`;
исключение — два демо-чекпоинта `checkpoints/ibm_*_fulldata.pt` (force-add, ~150KB)
для запуска продукта «из коробки» (какой нужен — `CLAUDE.md` / `docs/checkpoint_transfer_note.md`).

**Главные выводы Этапа 2 (кратко):** XGBoost (test AUC-PR 0.24 без времени)
доминирует над всеми edge-GNN; мультиграфовые адаптации Egressy в нашем —
ослабленном — режиме прироста над базовой GINe не дают (см. `docs/lit_review.md`,
честное очерчивание границ переноса); `norm_time` при temporal split вреден.

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
