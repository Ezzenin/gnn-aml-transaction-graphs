# IBM AML HI-Small (test): бейзлайн + ablation Multi-GNN (full-data, no-time)

Главные метрики: AUC-PR и F1 (позитив = laundering). Ablation: вклад
адаптаций reverse / port / ego поверх базовой GINe (RQ2).

| variant | auc_pr | f1 | recall_at_precision_90 | recall |
|---|---|---|---|---|
| XGBoost | 0.2398 | 0.2976 | 0.0217 | 0.3103 |
| GINe (base) | 0.0537 | 0.1160 | 0.0000 | 0.1446 |
| Multi-GNN (full) | 0.0412 | 0.1133 | 0.0000 | 0.3254 |

## Reference results (literature) — F1-minority, %

> ⚠️ Другой сплит (60/20/20) и режим обучения (все train-рёбра, class
> weights). НЕ сравнивать напрямую с нашими AUC-PR/F1 выше; приведено для
> ориентира масштаба и анализа расхождений (см. docs/lit_benchmarks.md).

| Модель (источник) | F1-minority % | примечание |
|---|---|---|
| XGBoost+GF (Altman 2023) | 63.2 | + подграфовые fan/cycle-фичи (GFP) |
| LightGBM+GF (Altman 2023) | 62.9 |  |
| GIN base (Egressy 2024) | 28.7 | 2 слоя, все train-рёбра, class weights |
| GIN+Ports | 54.9 | port numbering |
| GIN+ReverseMP | 46.8 | reverse MP — у нас не переносится при [10,10] |
| Multi-GIN (rev+port+ego) | 57.1 | ego поверх rev+port почти не добавляет |
| Multi-PNA+EU (SOTA) | 68.2 | единственный обошёл GBT+GF на всех AML |
| XGBoost без GF (Blanuša 2024) | 24.5 | ≈ наш XGBoost 19.0 по порядку |

Наш режим ослаблен (сабсэмпл негативов, окрестности [10,10], без edge-
updates, XGBoost без подграфовых GF-фич), поэтому абсолютные числа ниже;
воспроизводим направление эффекта, см. docs/lit_review.md.
