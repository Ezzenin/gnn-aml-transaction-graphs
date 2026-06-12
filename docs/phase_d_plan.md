# Фаза D — Multi-GNN адаптации + ablation (ядро Этапа 2, RQ2)

> ## ⚡ СТАТУС: код готов (написан и проверен на Mac). ПК запускает ТОЛЬКО обучение.
>
> Гибрид-тактика: весь код Фазы D реализован и smoke-проверен на Mac (CPU).
> На ПК с RTX 3070 нужна **только вычислительная мощь CUDA для обучения** — ни
> писать, ни отлаживать код не требуется. Разделы 1–3 ниже — описание уже
> сделанного (для понимания). Твоя задача на ПК — **раздел «Что делать на ПК»**.
>
> ### Что делать на ПК
> ```bash
> git pull                              # подтянуть готовый код Фазы D
> python -c "import torch; print(torch.cuda.is_available())"   # → True
> python -m src.compare --run-ibm       # XGBoost (если нет) + 5 GNN на CUDA → results/ibm_*
> git add results/ibm_*.json results/ibm_*_pr_curve.png results/ibm_comparison.md results/ablation.png
> git commit -m "feat(phase D): результаты ablation Multi-GNN на CUDA"
> git push
> ```
>
> ### ⚠️ ОБНОВЛЕНИЕ (фикс reverse MP)
> Первый прогон дал аномалию: reverse MP проседал втрое (обратно выводу Egressy).
> Причина — обратные рёбра добавлялись ПОСЛЕ семплинга loader'ом, окрестность не
> была настоящей двунаправленной. **Исправлено:** reverse теперь применяется к
> графу ДО семплинга (`add_reverse_edges` в `models.py`, вызывается в `train_edge`).
> Семантически изменились ТОЛЬКО варианты с reverse — перепрогнать **2 конфига**
> (остальные результаты валидны):
> ```bash
> git pull
> python -m src.train_edge --config configs/ibm_gine_rev.yaml
> python -m src.train_edge --config configs/ibm_multignn.yaml
> git add results/ibm_gine_rev_metrics.json results/ibm_gine_rev_pr_curve.png \
>         results/ibm_multignn_metrics.json results/ibm_multignn_pr_curve.png
> git commit -m "fix(phase D): перепрогон reverse-вариантов после фикса reverse MP" && git push
> ```
> Сводку/график (`--ibm`) и анализ доделываю на Mac.
> Дальше сборку финальных артефактов и анализ доделываю я на Mac (`git pull`).
> Если 16GB RAM упрётся при чтении 475MB CSV — раскомментируй `max_rows` в
> конфигах `configs/ibm_*.yaml` или см. caveat в разделе 0.
>
> ---
>
> ### Что уже реализовано на Mac (коммит — см. git log `feat(phase D)`)
> - `src/models.py`: логика `reverse_mp/ports/ego_ids` в `EdgeGNN.forward` + хелпер `compute_ports`.
> - `src/utils.py`: `resolve_device` (конфиг `device: cuda` сам откатывается на cpu вне ПК).
> - `configs/`: `ibm_gine_rev/port/ego.yaml` + заполнен `ibm_multignn.yaml`; base `ibm_gine.yaml` → `device: cuda`.
> - `src/compare.py`: `--run-ibm` (обучение, ПК) и `--ibm` (сборка таблицы+графика, Mac).
> - `tests/test_models.py`: `compute_ports` + forward всех адаптаций. `pytest -q` зелёный.
> - Smoke полного Multi-GNN пути (reverse+port+ego) на подвыборке прошёл на Mac/CPU.

---

## Историческая спецификация (как реализовывалось; раздел 0 — настройка ПК)

## Context

Проект `gnn-aml-transaction-graphs`. Фазы A–C завершены и в `main`.
Текущие test-результаты на полном IBM AML HI-Small (~5.08M рёбер, 515k узлов,
illicit 0.10%): **XGBoost AUC-PR 0.129**, **GINe-base AUC-PR 0.081**.

Цель Фазы D — воспроизвести и количественно проверить вывод Egressy: дают ли
мультиграфовые адаптации (reverse MP / port numbering / ego-IDs) прирост на
edge-classification. Реализуем как **флаг-управляемые, модель-агностичные
преобразования**, чтобы прогнать честный ablation (5 вариантов, меняем по одному).

Ключевое наблюдение: каркас флагов `reverse_mp/ports/ego_ids` в `EdgeGNN`
(`src/models.py`, класс начинается на строке 117) уже есть — конструктор
резервирует размерности (`node_in = in_node + ego`,
`edge_in = in_edge + reverse + port`), а тренер `src/train_edge.py` уже
пробрасывает флаги из конфига. **Не хватает только самой логики адаптаций в
`forward`** — сейчас она их игнорирует (`models.py:150`).

---

## 0. Настройка на ПК (один раз)

```bash
git clone https://github.com/Ezzenin/gnn-aml-transaction-graphs.git
cd gnn-aml-transaction-graphs
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# torch с CUDA (cu121 под RTX 3070), если базовый torch встал CPU-only:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
# затем PyG-расширения под свою версию torch/cuda.
```

Данные (в `.gitignore`, на GitHub их нет) — докачать с Kaggle в `data/ibm_aml/`:
```
data/ibm_aml/HI-Small_Trans.csv      (~475 MB)
data/ibm_aml/HI-Small_Patterns.txt
```

Проверка CUDA: `python -c "import torch; print(torch.cuda.is_available())"` → True.

**Caveat 16GB RAM:** `load_ibm_aml` читает весь CSV через pandas `dtype=str`
(пик несколько ГБ при `factorize`). Если упрётся в память — сначала отлаживать
с `dataset.max_rows: 300000`, финал гнать на полном датасете; при OOM на полном
рассмотреть чанковое чтение CSV в `load_ibm_aml` (отдельная мелкая правка).

---

## 1. Реализация адаптаций — `src/models.py`

Вся логика — внутри `EdgeGNN.forward` (модель-агностично через флаги), плюс
один хелпер. Конструктор и `build_edge_model` уже корректны — не трогать
(размерности `node_in`/`edge_in` уже учитывают флаги).

### 1.1. Хелпер port-numbering (новая функция в models.py)
```python
def compute_ports(edge_index, num_nodes):
    """Порядковый номер ребра среди входящих в узел-получатель (различение
    кратных рёбер мультиграфа). Векторно через argsort по dst. Возвращает
    LongTensor [E] на устройстве edge_index."""
    dst = edge_index[1]
    order = torch.argsort(dst, stable=True)
    sorted_dst = dst[order]
    _, counts = torch.unique_consecutive(sorted_dst, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(dst.numel(), device=dst.device) - torch.repeat_interleave(starts, counts)
    ports = torch.empty_like(dst)
    ports[order] = within
    return ports
```

### 1.2. Логика в `EdgeGNN.forward` (заменить тело до `h = self.node_enc(x)`)
Порядок: ego (узловая колонка) → port (рёберная колонка) → reverse (рёберная
колонка + удвоение рёбер). Суммарные размерности совпадут с тем, что уже задал
конструктор (Linear к порядку колонок безразличен).

- **ego-IDs** (`if self.ego_ids`): `ego = x.new_zeros((N,1)); ego[edge_label_index.reshape(-1)] = 1.0; x = cat([x, ego], 1)`. Помечает два конца классифицируемого сид-ребра (в локальной индексации мини-батча).
- **port** (`if self.ports`): `p = compute_ports(edge_index, N).to(x.dtype); edge_attr = cat([edge_attr, torch.log1p(p).unsqueeze(1)], 1)`. `log1p` ограничивает диапазон (хабы → большой port).
- **reverse MP** (`if self.reverse_mp`): добавить обратные рёбра + бинарный флаг направления (0=прямое, 1=обратное):
  ```python
  fwd = cat([edge_attr, edge_attr.new_zeros((E,1))], 1)
  rev = cat([edge_attr, edge_attr.new_ones((E,1))], 1)
  edge_attr = cat([fwd, rev], 0)
  edge_index = cat([edge_index, edge_index.flip(0)], 1)
  ```
  Сообщения текут в обе стороны, флаг сохраняет различение направления.

Важно: ego меняет колонки `x` ДО `node_enc`; port/reverse меняют `edge_attr` ДО
`edge_enc`. `_eval_edges` использует тот же `forward`, поэтому адаптации
автоматически работают и в инференсе. Тренер менять не нужно.

---

## 2. Ablation-конфиги — `configs/`

Базовая модель GINe (как в `configs/ibm_gine.yaml` — это вариант **base**,
все флаги false). Скопировать его, поменяв `model.params` флаги, `output_name`,
и добавив `device: cuda`. 5 вариантов из плана §D2:

| Конфиг | reverse | port | ego | output_name |
|---|---|---|---|---|
| `ibm_gine.yaml` (есть) | – | – | – | `ibm_gine` (base) |
| `ibm_gine_rev.yaml` | ✔ | – | – | `ibm_gine_rev` |
| `ibm_gine_port.yaml` | – | ✔ | – | `ibm_gine_port` |
| `ibm_gine_ego.yaml` | – | – | ✔ | `ibm_gine_ego` |
| `ibm_multignn.yaml` (пуст) | ✔ | ✔ | ✔ | `ibm_multignn` (full) |

Во всех IBM-конфигах выставить `device: cuda`. На CUDA можно поднять
`batch_size` (напр. 4096) и `num_neighbors` (напр. [15,15]) — VRAM 8GB при
мини-батчах позволяет; начать с текущих [10,10]/2048, повышать если запас есть.

---

## 3. Расширение `src/compare.py` (Acceptance D)

Сейчас `compare.py` — только Elliptic. Добавить IBM-ветку, не ломая Elliptic:

- `collect_ibm(results_dir)` — собрать `results/ibm_*_metrics.json` (XGBoost +
  все GINe-варианты), вытащить `test_metrics` (auc_pr, f1, recall@P90, recall).
- `write_ibm_table(...)` → `results/ibm_comparison.md` + `.csv`: строки
  XGBoost / base / +rev / +port / +ego / full (Multi-GNN).
- `plot_ablation(...)` → `results/ablation.png`: **прирост F1-minority (и/или
  AUC-PR) каждой адаптации относительно base** (bar-chart: base, +rev, +port,
  +ego, full; либо дельты к base). Это главный график отчёта по RQ2.
- CLI: добавить флаг `--run-ibm` (прогнать `ibm_gine` + 3 ablation + `ibm_multignn`
  через `src.train_edge.run`; XGBoost через `src.train_baseline` при отсутствии
  `results/ibm_xgboost_metrics.json`) и `--ibm` (только собрать таблицу/график).
  Существующий `--run`/сводка по Elliptic не трогать.

Переиспользовать существующие `collect`/`write_table`/`plot` как образец стиля.

---

## 4. Порядок выполнения на ПК

1. Реализовать §1 (models.py), §2 (конфиги), §3 (compare.py).
2. **Smoke** на подвыборке для отлова багов формы/устройства:
   добавить `dataset.max_rows: 300000` временно (или отдельный smoke-конфиг),
   прогнать `python -m src.train_edge --config configs/ibm_multignn.yaml`
   — самый нагруженный путь (все 3 адаптации). Убедиться, что shape edge_attr
   после reverse = `in_edge + port + 1`, без ошибок CUDA.
3. Полный ablation: `python -m src.compare --run-ibm` (или по одному конфигу).
4. Сводка/график: `python -m src.compare --ibm`.

---

## 5. Verification (Acceptance D)

- `pytest -q` зелёный (тесты не должны ломаться; новый хелпер `compute_ports`
  желательно покрыть мелким юнит-тестом на игрушечном мультиграфе — проверить,
  что параллельные рёбра в один узел получают разные порты 0,1,2…).
- Для каждого варианта создан `results/ibm_<variant>_metrics.json` с общей
  метрикой (AUC-PR, F1-minority, recall@P90) и per-pattern.
- `results/ibm_comparison.md` содержит строки XGBoost / base / +rev / +port /
  +ego / full; `results/ablation.png` показывает вклад каждой адаптации.
- Ожидание (проверяемая гипотеза, не гарантия): reverse MP — самая
  результативная одиночная адаптация; full (Multi-GNN) ≥ base. Если адаптации
  НЕ помогают — это тоже валидный результат для отчёта (честно фиксируем).
- W&B: хотя бы один вариант логируется (как в C2).

---

## 6. Коммиты

Отдельные коммиты по логике: (a) логика адаптаций + хелпер + тест;
(b) ablation-конфиги; (c) расширение compare.py; (d) результаты прогона
(`results/ibm_*`, `ablation.png`, `ibm_comparison.*`). Сообщения в стиле
существующих: `feat(phase D): ...`. Co-Authored-By Claude в конце.

---

## Не входит в Фазу D (следующие фазы)
- Фаза E: per-pattern сравнение трёх семейств + эвристики (NetworkX) — `src/heuristics.py`.
- Фаза G1: Streamlit-режим антифрода.
- edge-PNA как вторая база ablation — stretch (сейчас только GINe).
