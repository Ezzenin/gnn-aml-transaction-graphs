# CLAUDE.md — контекст проекта для Claude Code

Этот файл загружается автоматически в начале сессии. Он переносит контекст между
машинами (Mac → ПК) через git. Память Claude Code (`~/.claude/...`) и история чата
между машинами НЕ переносятся — только содержимое репозитория.

## О проекте
`gnn-aml-transaction-graphs` — курсовая (НИУ ВШЭ, ФКН). Тема: графовые
представления транзакций + GNN для выявления цепочек отмывания (8 паттернов AML).
Подробности — `README.md`. Полное задание Этапа 2 (критический путь, фазы) —
`docs/stage2_assignment.md`. Обзор литературы и постановка — `docs/lit_review.md`,
`docs/problem_statement.md`.

## Статус (на момент переноса на ПК)
- **Этап 1 — готов:** Elliptic node-classification. XGBoost test AUC-PR 0.80 ≥
  лучший GNN (SAGE 0.66). См. README.
- **Этап 2, фазы A–C — готовы и в `main`:**
  - A: W&B (`src.utils.init_wandb`), per-pattern метрика (`src.metrics.evaluate_per_group`), temporal-val.
  - B: загрузчик IBM AML (`src.datasets.load_ibm_aml`, `parse_ibm_patterns`), мультиграф «узел=счёт, ребро=транзакция», строгий temporal split. Реально: 515k узлов, 5.08M рёбер, illicit 0.10%.
  - C: бейзлайны на IBM — XGBoost на рёбрах (test AUC-PR **0.129**) и базовая edge-GNN GINe (`src.train_edge`, AUC-PR **0.081**).
- **Фаза D — КОД ГОТОВ (написан/проверен на Mac), ждёт обучения на CUDA:**
  адаптации reverse/port/ego в `EdgeGNN.forward`, `compute_ports`, ablation-конфиги,
  `compare.py --run-ibm/--ibm`. **На ПК осталось только запустить обучение** —
  см. блок «Что делать на ПК» в начале `docs/phase_d_plan.md`.
- Дальше: E (per-pattern + эвристики), G1 (Streamlit антифрод), H (воспроизводимость).

## ⚠️ НУЖНО ПЕРЕНЕСТИ ЧЕКПОИНТ С ПК (для Фазы G1 Streamlit)
Чекпоинты (`*.pt`) в `.gitignore`, по git не приезжают. Для G1 нужен ОДИН свежий
чекпоинт обученной edge-GNN с ПК. Рекомендуемый: **`checkpoints/ibm_multignn_fulldata.pt`**
(полный Multi-GNN, демонстрирует метод КР) ИЛИ **`checkpoints/ibm_gine_fulldata.pt`**
(лучший GNN по AUC-PR 0.054). Оба — режим full-data/no-time, актуальная архитектура
(edge-head + in_edge_label, pos_weight=100).

Как перенести (любой способ):
- git (надёжно, файл ~150KB): на ПК `git add -f checkpoints/ibm_multignn_fulldata.pt`,
  закоммитить, запушить → на Mac `git pull`. (`-f` обходит .gitignore.)
- или вручную скопировать `.pt` в `checkpoints/` на Mac (scp / облако / USB).

ВАЖНО: локальный `checkpoints/ibm_gine.pt` (9 июня) — УСТАРЕЛ (старая голова без
`label_edge_enc`, до P0.2), текущим кодом не загрузится. Не использовать для G1.
Чекпоинт хранит state_dict + config + in_node/in_edge/in_edge_label + threshold —
их использует загрузчик в `src/eval.py` (TODO G1).

## Гибрид-тактика (Mac ↔ ПК)
Код, конфиги, сборка артефактов/графиков — на **Mac** (CPU достаточно). На **ПК с
RTX 3070** выполняется ТОЛЬКО CUDA-тяжёлое обучение GNN. Обмен — через git:
результаты (`results/*.json`, `*.png`) трекаются и возвращаются `git pull`;
чекпоинты (`*.pt`) в `.gitignore` — при нужде переносить вручную. На Mac любой
IBM-конфиг с `device: cuda` сам откатывается на cpu (`resolve_device`), так что
smoke-прогоны на подвыборке (`dataset.max_rows`) работают и здесь.

## Железо / окружение
- CUDA-обучение — **на ПК: Ryzen 3700X, RTX 3070 (8GB VRAM, CUDA), 16GB RAM**.
- Узкое место — **не GPU, а 16GB RAM**: `load_ibm_aml` читает 475MB CSV через
  pandas `dtype=str` (пик несколько ГБ). Отлаживать на `dataset.max_rows: 300000`,
  финал — на полном датасете; при OOM добавить чанковое чтение CSV.
- 8GB VRAM хватает: `LinkNeighborLoader` держит память ограниченной (мини-батчи).
- В IBM-конфигах ставить `device: cuda`. Проверка: `python -c "import torch;print(torch.cuda.is_available())"`.
- Данные не в git (`.gitignore`): докачать с Kaggle в `data/ibm_aml/`
  (`HI-Small_Trans.csv`, `HI-Small_Patterns.txt`).

## Конвенции (соблюдать)
- **Не ломать Elliptic-ветку.** IBM-модели/тренер — отдельные сущности
  (`EdgeGNN`/`train_edge.py`); общие только метрики/утилиты/логирование.
- Метрики при дисбалансе: только **AUC-PR / F1-minority / recall@precision**;
  accuracy не отчитывать. Порог фиксируется по val, метрики — на test.
- Каждый эксперимент = YAML-конфиг в `configs/`; результат = `results/<name>_metrics.json`
  (+ per-pattern) и `results/<name>_pr_curve.png`. Чекпоинты — `checkpoints/` (в .gitignore).
- Сиды фиксировать (`src.utils.set_seed`). При OOM на GPU — снижать `num_neighbors`/`batch_size`.
- Докстринги и комментарии — на русском (как в существующем коде).
- Тесты зелёные: `pytest -q`. CI-дружелюбные скипы при отсутствии данных.

## Git
- Remote: `github.com/Ezzenin/gnn-aml-transaction-graphs`, ветка `main`.
- Коммиты в стиле `feat(phase D): ...` / `docs(...)`. Завершать:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Коммитить/пушить только по просьбе пользователя.

## Точки входа
```bash
python -m src.train_edge --config configs/ibm_gine.yaml   # edge-GNN на IBM (Фаза C/D)
python -m src.train_baseline --config configs/ibm_xgb.yaml # XGBoost на рёбрах
python -m src.compare --run                                # Elliptic: сводка
pytest -q
```
```
src/  datasets.py models.py train.py train_edge.py train_baseline.py
      metrics.py compare.py utils.py baselines.py eval.py explain.py graph_build.py
```
