# Демо-чекпоинты в `main` (Фаза G1, Streamlit)

В репозитории (`main`) лежат **два** обученных на ПК (RTX 3070) чекпоинта edge-GNN,
актуальная архитектура (edge-head `label_edge_enc`, P0.2), режим **full-data /
no-time, `pos_weight=100`**. Добавлены через `git add -f` (в `.gitignore` есть
`checkpoints/`/`*.pt`); ~150–170KB каждый. Оба грузятся `src.eval.load_edge_model`.

> Замечание о `pos_weight`: авто-значение neg/pos (~1290) при дисбалансе 0.1%
> коллапсирует модель в «всё-позитив» (recall=1.0, AUC-PR≈случайность); поэтому
> в `configs/ibm_*_fulldata.yaml` задан `pos_weight=100`.

## Два чекпоинта

| чекпоинт | модель | test AUC-PR | test F1 | роль |
|---|---|---|---|---|
| **`checkpoints/ibm_gine_fulldata.pt`** | base GINe (без адаптаций) | **0.054** | 0.116 | **дефолт Streamlit** — лучший GNN по метрике |
| `checkpoints/ibm_multignn_fulldata.pt` | полный Multi-GNN (reverse+port+ego) | 0.041 | 0.113 | для скриншота «иллюстрация метода» (RQ2) |

Метрики близкие (разница в пределах шума прогона). Дефолт продукта — base GINe;
переключение в сайдбаре `app/streamlit_app.py`. Загрузчик чекпоинт-агностичен:
архитектура восстанавливается из метаданных (`in_node/in_edge/in_edge_label` +
флаги адаптаций из `config`), ключи: `state_dict / config / in_node / in_edge /
in_edge_label / threshold`.

## Как пересоздать на CUDA
```bash
python -m src.train_edge --config configs/ibm_gine_fulldata.yaml      # base
python -m src.train_edge --config configs/ibm_multignn_fulldata.yaml  # Multi-GNN
```
Чекпоинт сохраняется в `checkpoints/<output_name>.pt` (см. `save_checkpoint` в конфиге).

> Старый `checkpoints/ibm_gine.pt` (до фикса P0.2) НЕ совместим с текущим кодом
> (голова `2*hidden` vs `3*hidden`) — не использовать.
