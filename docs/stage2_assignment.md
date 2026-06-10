# Этап 2 — задание для Claude Code

**Проект:** `gnn-aml-transaction-graphs`
**Тема КР:** построение графовых представлений транзакционных данных и применение GNN для выявления цепочек (паттернов) финансовых операций.
**Исполнитель:** Claude Code (один, без Codex).
**Контекст:** Этап 1 завершён и работает — см. раздел 0. Этот документ — критический путь Этапа 2, отсортированный по вкладу в оценку.

---

## 0. Что уже готово (не трогать без причины)

Полностью реализовано и прогнано на **Elliptic** (node-classification):

- `src/datasets.py` — загрузчик Elliptic (PyG), программное определение `positive_class`, рантайм-проверка признаков, статистика сплитов.
- `src/baselines.py` — XGBoost + LogReg с обработкой дисбаланса.
- `src/models.py` — GCN / GraphSAGE / GAT / GIN / PNA + фабрика `build_model`.
- `src/train.py`, `src/train_baseline.py`, `src/compare.py` — обучение, сравнение, сводка.
- `src/metrics.py` — AUC-PR, F1-minority, recall@precision, PR-кривая.
- `src/utils.py` — сиды, конфиги, сохранение JSON.
- `configs/elliptic_*.yaml` (7 шт.), `tests/test_metrics.py`, `tests/test_datasets.py`.
- `results/*_metrics.json` + `comparison.md`; `notebooks/01_elliptic_eda.ipynb` (выполнен).

**Главный итог Этапа 1:** XGBoost (test AUC-PR 0.80) ≥ GNN (лучший SAGE 0.66); у GNN сильный разрыв val→test (распределенческий сдвиг Elliptic). Это валидированный фундамент, но **сам по себе он не отвечает на исследовательские вопросы КР** — Elliptic не содержит 8 паттернов. Вся новизна в Этапе 2.

---

## 1. Что максимизирует оценку (привязка к Прил. 5/6 правил)

| Критерий отзыва | Чем закрываем в Этапе 2 |
|---|---|
| Сложность/полнота исследования | Воспроизведение SOTA-адаптаций Egressy (reverse MP / port / ego-ID) **с ablation** — прямая проверка их вывода. Это ядро. |
| Полнота источников + связь с темой | Количественная разбивка F1 **по 8 паттернам** (RQ3) — то, ради чего брали IBM AML. |
| Программная реализация | Рабочий Streamlit-продукт с **визуализацией найденной цепочки** (режим антифрода). |
| Чёткость целей | Каждый эксперимент отвечает на конкретный RQ (см. §2). |
| Оформление/воспроизводимость | W&B-логирование, фиксация конфигов/сидов, открытый GitHub, генерируемые таблицы/графики для отчёта. |

**Принцип приоритизации (при сжатом сроке):** сначала закрыть **Minimum Viable Thesis (MVT)**, потом stretch. Лучше один чистый, защищаемый результат по паттернам, чем три недоделанных трека.

### MVT (обязательно, в этом порядке)
1. **Фаза A** — W&B + per-pattern метрика + мелкие правки (быстро).
2. **Фаза B** — IBM AML HI-Small: загрузка + граф «узел = счёт, ребро = транзакция» (edge-classification).
3. **Фаза C** — бейзлайны на IBM: XGBoost(+graph-фичи) + базовая edge-GNN (GINe/PNA с edge-фичами).
4. **Фаза D** — Multi-GNN адаптации + **ablation** (RQ2). ← главный вклад.
5. **Фаза E** — разбивка по 8 паттернам + минимальные эвристики (RQ3).
6. **Фаза G1** — Streamlit, **только режим антифрода** с визуализацией цепочки.
7. **Фаза H** — воспроизводимость + генерация финальных артефактов для отчёта.

### Stretch (если останется время — иначе в «Перспективы»)
- **Фаза F1** — RQ1: альтернативное представление «узел = транзакция» (line-graph) на IBM, честное сравнение с edge-classification.
- **Фаза F2** — EvolveGCN на Elliptic (роль времени, строгий temporal split).
- **Фаза G2** — Streamlit PF-режим (синтетика, follow-the-money).
- **Фаза E+** — GARG-AML-стиль (фичи 2-го порядка для smurfing), GNNExplainer.

> Если время поджимает, MVT-пунктов 1–7 достаточно для уверенно высокой оценки. F-фазы честно описываются в разделе «Перспективы» отчёта.

---

## 2. Исследовательские вопросы и какой эксперимент на них отвечает

- **RQ1** (представление: счёт/ребро=транзакция vs узел=транзакция) → Фаза B (edge-class.) + Фаза F1 (node=tx). При сжатом сроке частично: edge-classification сделана, node=tx — в перспективы.
- **RQ2** (вклад мультиграфовых адаптаций) → **Фаза D ablation**. Это самый сильный пункт; не срезать.
- **RQ3** (на каких паттернах GNN бьёт XGBoost/эвристики) → **Фаза E** (F1 по 8 паттернам, сравнение трёх семейств).

---

## ФАЗА A — Логирование и правки Этапа 1

Цель: включить воспроизводимость и подготовить метрики под паттерны. Маленькая, делается первой.

### A1. Включить W&B по-настоящему
- В `src/utils.py` добавить хелпер `init_wandb(config, run_name) -> Optional[wandb.Run]` (единая точка инициализации, мягкий fallback при отсутствии пакета/ключа).
- В `train.py` / `train_baseline.py` уже есть крючки — переключить дефолт: глобальный флаг через переменную окружения `WANDB_MODE` или `wandb.enabled` в конфиге. Логировать: полный `config`, per-epoch `loss`, `val/auc_pr`, `val/f1`; финальные `test/*`; PR-кривую как `wandb.Image`; итоговую таблицу сравнения как `wandb.Table`.
- В `requirements.txt` раскомментировать `wandb`.
- Документировать в README: `wandb login` + как выключить (`wandb.enabled: false` или `WANDB_MODE=offline`).

### A2. Метрика per-pattern (готовим под Фазу E заранее)
В `src/metrics.py` добавить:
```python
def evaluate_per_group(y_true, y_score, group_labels, threshold,
                       groups=None) -> dict:
    """F1/precision/recall/recall@P для позитивов, разбитых по типу паттерна.

    y_true: бинарные метки (1 = illicit/laundering).
    group_labels: для каждого позитива — строковый тип паттерна
        (fan_out, fan_in, ... , stack) или 'none' для негативов.
    threshold: единый порог (тот же, что для общей метрики).
    Возвращает {pattern: {f1, precision, recall, n_pos, n_detected}}.
    Логика: для каждого паттерна позитивы этого типа vs все негативы;
    показывает, какие цепочки модель ловит, а какие — нет.
    """
```
Юнит-тест в `tests/test_metrics.py` на игрушечном примере (2–3 паттерна).

### A3. (Опционально, дёшево) Строгий temporal-split флаг на Elliptic
- В `make_val_split` / `train.py` добавить опцию `temporal_val: true` — брать val как последние временны́е шаги внутри train, а не случайно. Усиливает пункт §5.5 плана (роль времени, отсутствие утечки). Конфиг-вариант `configs/elliptic_gcn_temporal.yaml`.

**Acceptance A:** `pytest -q` зелёный; один прогон Elliptic пишет run в W&B (или offline-дамп); `evaluate_per_group` покрыт тестом.

---

## ФАЗА B — IBM AML HI-Small: данные и граф (edge-classification)

Цель: получить направленный мультиграф транзакций «узел = счёт, ребро = транзакция» с метками `Is Laundering` и привязкой позитивов к типу паттерна.

### B1. Данные
- Датасет: Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, вариант **HI-Small** (НЕ Medium/Large).
- Файлы: `HI-Small_Trans.csv` (транзакции) + `HI-Small_Patterns.txt` (инстансы паттернов по 8 типологиям).
- В `.gitignore` уже исключены `data/`, `*.csv` — данные не коммитим. Добавить в README инструкцию по скачиванию (`kaggle datasets download ...` или ручная ссылка) и ожидаемые пути `data/ibm_aml/HI-Small_Trans.csv`, `.../HI-Small_Patterns.txt`.
- **Все размеры/доли проверять на рантайме** (как в Elliptic), не хардкодить: число транзакций, число счетов, доля illicit (ожидается крайне низкая, < 0.2% — дисбаланс жёстче Elliptic).

### B2. Загрузчик и парсер паттернов
В `src/datasets.py` добавить:
```python
def load_ibm_aml(root="data/ibm_aml", variant="HI-Small") -> tuple["Data", dict]:
    """Прочитать CSV транзакций, собрать направленный мультиграф PyG.

    Узел = счёт, ключ = f"{bank}_{account}" (счета повторяются между банками).
    Ребро = транзакция (направление source_acct -> dest_acct).
    Edge features (числовые + закодированные категориальные):
      - log1p(Amount Paid), log1p(Amount Received)
      - Receiving/Payment Currency (label-encoding, фиксировать словарь)
      - Payment Format (label-encoding)
      - timestamp -> unix / нормированное время (для temporal split и порядка)
    edge_label = Is Laundering (0/1).
    edge_pattern = тип паттерна для позитивов (из B3), 'none' для негативов.
    Возвращает (data, meta): meta с size/доля illicit/кодировки/границей temporal split.
    """

def parse_ibm_patterns(path) -> dict:
    """Распарсить *_Patterns.txt в карту транзакция -> тип паттерна.

    Файл состоит из блоков:
      BEGIN LAUNDERING ATTEMPT - <TYPE>
      <строки транзакций в формате CSV>
      END LAUNDERING ATTEMPT
    <TYPE> нормализовать к 8 каноническим: fan_out, fan_in, gather_scatter,
    scatter_gather, cycle, random, bipartite, stack (составить маппинг, т.к.
    в файле названия могут отличаться регистром/дефисами).
    Ключ транзакции = кортеж (timestamp, from_acct, to_acct, amount) или
    стабильный хеш — согласовать с тем, как строятся рёбра в load_ibm_aml.
    """
```
- **Узловые признаки:** на старте — константа/степенные фичи (in/out-degree, log суммы по счёту). Полноценные узловые атрибуты в IBM скудны; основная информация на рёбрах. Это нормально для edge-classification.
- **Temporal split:** train = ранние timestamps, test = поздние; val = последняя доля train по времени. Строго по времени (антиутечка) — это требование плана §5.5 и аргумент в отзыве.

### B3. Масштаб и мини-батчи (критично)
HI-Small — миллионы рёбер; полный граф в память под edge-classification не лезет на средней GPU. Использовать:
- `torch_geometric.loader.LinkNeighborLoader` для edge-level мини-батчей (семплинг соседей вокруг каждого классифицируемого ребра).
- Параметры в конфиг: `num_neighbors`, `batch_size`, `num_workers`.
- Под XGBoost-бейзлайн — фичи считаются без мини-батчей (табличка по рёбрам + graph-derived фичи).

**Acceptance B:** `load_ibm_aml` возвращает `Data` с `edge_index`, `edge_attr`, `edge_label`, `edge_pattern`; `meta` печатает реальные размеры и долю illicit; temporal split детерминирован; есть быстрый smoke-тест `tests/test_graph_build.py` (сейчас скип — заменить) на мини-сэмпле (первые N строк), проверяющий согласованность ключей рёбер и привязки паттернов.

---

## ФАЗА C — Бейзлайны на IBM (edge-classification)

Цель: честная «нижняя планка», к которой будем сравнивать GNN-адаптации.

### C1. Неграфовый: XGBoost на рёбрах
- Фичи ребра: edge_attr + **graph-derived** (in/out-degree обоих счетов, число параллельных рёбер между парой, агрегаты сумм по счёту). Последнее — важно: деревья на табличке часто конкурентны GNN, и сильный бейзлайн усиливает выводы.
- Переиспользовать `train_xgboost` из `baselines.py`; новый CLI-вход или ветка в `train_baseline.py` по `dataset.name == "ibm_aml"`.
- `scale_pos_weight` обязательно (дисбаланс ещё жёстче).
- Конфиг `configs/ibm_xgb.yaml`.

### C2. Базовая edge-GNN с edge-фичами
- В `src/models.py` добавить edge-feature-aware блок (отдельно от Elliptic-моделей — те не трогать):
```python
class EdgeGNN(torch.nn.Module):
    """Message-passing с учётом edge_attr + голова edge-классификации.

    conv: GINEConv или PNAConv с edge_dim (выбор по config).
    Голова: MLP([h_u || h_v || e_uv]) -> 2 логита для каждого ребра.
    Поддерживает флаги адаптаций (см. Фаза D): reverse_mp, ports, ego_ids.
    """
def build_edge_model(name, in_node, in_edge, hidden=64, **adapt_flags): ...
```
- Обучение: новый `src/train_edge.py` (мини-батч через LinkNeighborLoader, class weights / при необходимости focal loss, early stopping по val AUC-PR, фиксация порога по val). Тот же интерфейс метрик/сохранения, что у `train.py`.
- Конфиги `configs/ibm_gine.yaml`, `configs/ibm_pna.yaml`.

**Acceptance C:** XGBoost и базовая edge-GNN дают `results/ibm_<model>_metrics.json` с общей метрикой (AUC-PR, F1-minority, recall@P) **и** per-pattern (через `evaluate_per_group`). Один прогон логируется в W&B.

---

## ФАЗА D — Multi-GNN адаптации + ablation (ЯДРО, RQ2)

Цель: воспроизвести и проанализировать эффект адаптаций Egressy. Реализовать как **модель-агностичные преобразования**, включаемые флагами, чтобы можно было делать ablation.

### D1. Адаптации (по отдельности включаемые)
- **reverse MP (направленность):** добавить обратные рёбра с бинарным признаком направления / раздельная агрегация входящих и исходящих сообщений. Ожидаемо самая результативная одиночная адаптация — реализовать первой.
- **port numbering:** для каждого узла пронумеровать инцидентные рёбра (различение кратных рёбер мультиграфа); порт как доп. edge-фича.
- **ego-IDs:** для классифицируемого ребра пометить два его конечных узла бинарным признаком = 1 (в подграфе мини-батча сид-ребро известно).

> Рекомендация плана: проще **портировать** из `github.com/IBM/Multi-GNN` (там GIN/GAT/PNA/RGCN + эти адаптации под edge-classification на IBM AML), чем писать с нуля. Claude Code: сверить интерфейсы, аккуратно перенести только нужное, не тянуть весь репозиторий; сохранить наш слой метрик/логирования/конфигов.

### D2. Сетка ablation (фиксируем базовый конфиг, меняем по одному)
Базовая модель (например GINe или PNA) на IBM, далее варианты:
1. base
2. base + reverse
3. base + port
4. base + ego
5. base + reverse + port + ego (полный Multi-GNN)

Каждый вариант → отдельный конфиг `configs/ibm_<base>_<flags>.yaml` и строка в сводке. Это **прямая проверка вывода Egressy** — главный граф맣ик/таблица отчёта.

**Acceptance D:** `src/compare.py` расширен на IBM: собирает все `results/ibm_*_metrics.json`, строит (а) общую сравнительную таблицу (XGBoost vs base GNN vs +адаптации) и (б) **график прироста F1-minority от каждой адаптации** (ablation). Видно, какая адаптация и сколько даёт.

---

## ФАЗА E — Разбивка по 8 паттернам + эвристики (RQ3)

Цель: ответить, **на каких цепочках** GNN бьёт XGBoost и классические эвристики.

### E1. Per-pattern сравнение
- Для всех трёх семейств (XGBoost, base GNN, Multi-GNN) посчитать `evaluate_per_group` → таблица F1 × 8 паттернов × 3 модели.
- График: сгруппированный bar-chart F1 по паттернам. Ожидаемая история: GNN с адаптациями выигрывает на «структурных» паттернах (cycle, scatter-gather), деревья — там, где хватает локальных фич.

### E2. Классические графовые эвристики (минимум)
В `src/graph_build.py` / новый `src/heuristics.py` через NetworkX:
- перечисление простых циклов до длины k (`simple_cycles`) → детектор cycle/random;
- степенные правила для fan-in/fan-out (порог по in/out-degree);
- (stretch) GARG-AML-стиль: фичи 2-го порядка соседства для smurfing (gather/scatter).
- Прогнать на тех же test-рёбрах, посчитать per-pattern, добавить в сравнение как интерпретируемый baseline.

**Acceptance E:** `results/per_pattern.md` + `results/per_pattern.png`; в сравнении присутствуют XGBoost, Multi-GNN и эвристики; для каждого паттерна видно лучшую модель.

---

## ФАЗА F — Stretch (в перспективы, если не успеваем)

- **F1 (RQ1, представление):** альтернативный граф «узел = транзакция» (line-graph: транзакции — узлы, связь, если делят счёт) → node-classification теми же GNN; сравнить с edge-classification из Фазы B по общей и per-pattern метрике. Закрывает RQ1 количественно.
- **F2:** EvolveGCN на Elliptic (роль времени; строгий temporal split). Дополняет Этап 1.
- **E+:** GNNExplainer для объяснения, какие рёбра/подграф «сделали» предсказание (для скриншотов продукта и раздела объяснимости).

---

## ФАЗА G — Streamlit-продукт

Делать **рано** (риск §10 плана: не оставлять продукт на конец). Минимум — режим антифрода.

### G1. Режим «Антифрод/AML» (обязательно)
`app/streamlit_app.py`:
- загрузка подграфа транзакций (сэмпл из IBM test или загруженный CSV);
- инференс обученной Multi-GNN (загрузка чекпоинта из `checkpoints/`, путь в конфиге);
- подсветка подозрительных рёбер/счетов по score;
- **визуализация найденной цепочки** (цикл / scatter-gather) поверх графа через PyVis или Plotly;
- объяснимость: подсветка паттерна (или GNNExplainer из E+).
- В `requirements.txt` добавить `streamlit`, `pyvis`/`plotly`.

### G2. Режим «Personal finance» (stretch)
- На синтетическом/PaySim кэшфлоу: follow-the-money — самопереводы, рекуррентные платежи, транзитные цепочки, агрегация по контрагентам. Честно: эвристики + визуализация, без претензии на SOTA. Переиспользовать граф-билдеры и визуализацию из G1.

**Acceptance G:** `streamlit run app/streamlit_app.py` запускается; режим антифрода грузит чекпоинт, показывает граф и подсвечивает хотя бы одну цепочку. Скриншоты сохранить для отчёта (раздел 7).

---

## ФАЗА H — Воспроизводимость и артефакты для отчёта

Цель: чтобы текст отчёта писался по готовым таблицам/графикам, а GitHub был защитопригоден.

- `src/compare.py` генерирует все финальные артефакты одной командой: общие сравнения (Elliptic + IBM), ablation-график, per-pattern таблица/график.
- README: полный раздел «Воспроизведение Этапа 2» — скачивание IBM, команды обучения по фазам, где лежат результаты, как включить/выключить W&B.
- `docs/lit_review.md`: дозаполнить раздел «Выводы» (обозначить пробел, который закрывает работа), сверить выходные данные, пометить, что финальное оформление — по ГОСТ Р 7.0.5-2008.
- `docs/problem_statement.md`: снять TODO (предметная область/актуальность), привести формулировки в соответствие с фактической постановкой.
- Тесты: `pytest -q` зелёный; CI-дружелюбные скипы при отсутствии данных (как сейчас в `test_datasets.py`).
- Финальная чистка: единый стиль конфигов, докстринги, `requirements.txt` зафиксирован.

**Acceptance H:** свежий клон + скачивание данных + команды из README воспроизводят все `results/*`. Все графики/таблицы для разделов 6–7 отчёта сгенерированы.

---

## 3. Новые / изменённые файлы (карта)

```
src/
  datasets.py      [+] load_ibm_aml, parse_ibm_patterns
  graph_build.py   [NEW] account-graph build, (stretch) line-graph
  heuristics.py    [NEW] cycle/motif детекторы (NetworkX), GARG-стиль (stretch)
  models.py        [+] EdgeGNN, build_edge_model, адаптации reverse/port/ego
  train_edge.py    [NEW] обучение edge-GNN (LinkNeighborLoader, чекпоинты)
  train_baseline.py[+] ветка ibm_aml для XGBoost на рёбрах
  metrics.py       [+] evaluate_per_group
  compare.py       [+] сбор/таблицы/графики для IBM + ablation + per-pattern
  eval.py          [FILL] вынести общий инференс/чекпоинт-загрузку (для app)
  explain.py       [FILL/stretch] GNNExplainer / выделение паттерна
  utils.py         [+] init_wandb, (опц.) temporal val helper
app/
  streamlit_app.py [FILL] режим антифрода (G1), PF (G2 stretch)
configs/
  ibm_xgb.yaml, ibm_gine.yaml, ibm_pna.yaml,
  ibm_<base>_<flags>.yaml (ablation), ibm_multignn.yaml [был пуст — заполнить]
tests/
  test_graph_build.py [REPLACE skip] smoke-тест графа+паттернов на мини-сэмпле
  test_metrics.py     [+] тест evaluate_per_group
results/                 новые ibm_*_metrics.json, per_pattern.*, ablation.*
```

---

## 4. Подводные камни (держать в голове)

- **Дисбаланс жёстче Elliptic** (illicit-рёбер < 0.2%): только AUC-PR / F1-minority / recall@P; `scale_pos_weight`, взвешенный лосс, при необходимости focal loss; не отчитываться по accuracy.
- **Масштаб HI-Small:** полный граф под edge-class. не влезет — обязательны мини-батчи (LinkNeighborLoader). Large/Medium не брать.
- **Согласованность ключей:** ключ ребра в `load_ibm_aml` и в `parse_ibm_patterns` должен совпадать байт-в-байт, иначе привязка паттернов «потеряется». Проверить тестом на мини-сэмпле.
- **Названия паттернов** в `*_Patterns.txt` нормализовать к 8 каноническим (регистр/дефисы/синонимы).
- **Временна́я утечка:** строгий temporal split на IBM (train = ранние t, test = поздние); val — последняя доля train по времени.
- **Валюты:** суммы в разных валютах — на старте оставить (log-amount + currency как фича), не усложнять конвертацией.
- **Детерминизм/OOM на GPU:** фиксировать сиды; при OOM уменьшать `num_neighbors`/`batch_size`; XGBoost — `n_jobs` под детерминизм.
- **Чекпоинты:** сохранять лучший по val в `checkpoints/` (в `.gitignore` уже есть `*.pt`/`checkpoints/`); путь — в конфиг, чтобы продукт грузил.
- **Не ломать Elliptic-ветку:** IBM-модели и тренер — отдельные сущности; общие — только метрики/утилиты/логирование.

---

## 5. Как кормить Claude Code (порядок задач)

Давать по одной фазе, проверяя acceptance перед следующей. Рекомендуемые «команды задачи»:

1. «Фаза A: включи W&B (utils.init_wandb + конфиг-флаг), добавь metrics.evaluate_per_group с тестом, опц. temporal-val флаг. Прогони один Elliptic-ран в offline W&B. Покажи диффы и `pytest -q`.»
2. «Фаза B: реализуй load_ibm_aml + parse_ibm_patterns + temporal split. Добавь smoke-тест на первых N строках. Не грузи полный датасет в тесте.»
3. «Фаза C: XGBoost на рёбрах (graph-derived фичи) + EdgeGNN (GINe/PNA, edge_attr) + train_edge.py с LinkNeighborLoader. Конфиги ibm_xgb/ibm_gine/ibm_pna. Прогон + per-pattern в results.»
4. «Фаза D: адаптации reverse/port/ego как флаги модели (свериться с IBM/Multi-GNN, перенести аккуратно). Конфиги ablation. Расширь compare.py: таблица + график прироста F1.»
5. «Фаза E: per-pattern сравнение трёх семейств + эвристики (NetworkX). results/per_pattern.*»
6. «Фаза G1: Streamlit режим антифрода с загрузкой чекпоинта и визуализацией цепочки (PyVis/Plotly). Сделай скриншот-инструкцию.»
7. «Фаза H: финальный compare-all, README-раздел воспроизведения, дозаполни docs/.»
8. (Если есть время) F1/F2/G2/E+.

---

## 6. Тайм-бокс при дедлайне ~28 июня (ориентир)

| Дни | Фазы | Результат |
|---|---|---|
| 1 | A | Логирование + per-pattern метрика готовы |
| 2–4 | B | IBM граф + паттерны, temporal split, smoke-тест |
| 5–6 | C | Бейзлайны на IBM (XGBoost + базовая edge-GNN) |
| 7–10 | D | Адаптации + ablation (ядро, RQ2) |
| 11–12 | E | Per-pattern + эвристики (RQ3) |
| 13–14 | G1 + H | Продукт (антифрод) + воспроизводимость/артефакты |
| резерв | F/G2 | Stretch или буфер под текст отчёта |

Написание отчёта (разделы 4–8) идёт параллельно по мере появления таблиц/графиков — это задача для чата с Claude, не для Claude Code.

> **Если сроки сорваны:** гарантированно довести MVT-фазы A→E→G1; F-фазы и PF-режим описать в «Перспективах». Один чистый ablation + per-pattern результат + рабочий продукт = защитопригодная КР высокого уровня.
