# IBM AML HI-Small (test): бейзлайн + ablation Multi-GNN

Главные метрики: AUC-PR и F1 (позитив = laundering). Ablation: вклад
адаптаций reverse / port / ego поверх базовой GINe (RQ2).

| variant | auc_pr | f1 | recall_at_precision_90 | recall |
|---|---|---|---|---|
| XGBoost | 0.1289 | 0.1900 | 0.0000 | 0.2508 |
| GINe (base) | 0.0190 | 0.0562 | 0.0000 | 0.2442 |
| +reverse | 0.0396 | 0.0851 | 0.0000 | 0.1529 |
| +port | 0.0385 | 0.1055 | 0.0000 | 0.1329 |
| +ego | 0.0366 | 0.1077 | 0.0000 | 0.1774 |
| Multi-GNN (full) | 0.0510 | 0.1202 | 0.0000 | 0.1529 |
