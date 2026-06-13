# Записка: альтернативный чекпоинт для Фазы G1 (Streamlit)

Эта ветка (`alt-checkpoint-gine-fulldata`) — **альтернатива** к чекпоинту,
перенесённому в `main`. Mac выбирает, какой из двух использовать для G1; оба
с ПК (RTX 3070), актуальная архитектура (edge-head `label_edge_enc`, P0.2),
режим **full-data / no-time, `pos_weight=100`** (не вырожденные — авто
`pos_weight≈1290` коллапсировал в «всё-позитив», понижен до 100).

## Два чекпоинта на выбор

| чекпоинт | где | модель | test AUC-PR | test F1 | когда брать |
|---|---|---|---|---|---|
| `ibm_multignn_fulldata.pt` | `main` (df9e44e) | полный Multi-GNN (reverse+port+ego) | 0.041 | 0.113 | **демонстрирует метод КР** (RQ2: мультиграфовые адаптации) |
| `ibm_gine_fulldata.pt` | **эта ветка** | base GINe (без адаптаций) | **0.054** | 0.116 | **лучший GNN по метрике** среди всех режимов |

Tradeoff: Multi-GNN показывает сам метод курсовой (адаптации), но base GINe
даёт numerically лучший AUC-PR. Для демо-витрины «работающей модели» — base;
для иллюстрации вклада адаптаций — Multi-GNN. Метрики близкие, разница в пределах
шума прогона.

## Метаданные альтернативного чекпоинта (`ibm_gine_fulldata.pt`)
- `in_node=5, in_edge=5, in_edge_label=5`, `threshold≈0.492`
- флаги модели: `reverse_mp=False, ports=False, ego_ids=False`
- ключи: `state_dict / config / in_node / in_edge / in_edge_label / threshold`
  — грузится загрузчиком `src/eval.py` (TODO G1).

## Как забрать на Mac
```bash
git fetch origin
git checkout alt-checkpoint-gine-fulldata -- checkpoints/ibm_gine_fulldata.pt
# или влить всю ветку: git merge alt-checkpoint-gine-fulldata
```
Чекпоинт добавлен через `git add -f` (в `.gitignore` есть `checkpoints/`/`*.pt`).
Это разовый перенос — после забора можно `git rm --cached checkpoints/*.pt`,
чтобы `.gitignore` снова действовал.
