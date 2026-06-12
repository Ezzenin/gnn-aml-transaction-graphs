# IBM AML HI-Small (test): бейзлайн + ablation Multi-GNN

Главные метрики: AUC-PR и F1 (позитив = laundering). Ablation: вклад
адаптаций reverse / port / ego поверх базовой GINe (RQ2).

| variant | auc_pr | f1 | recall_at_precision_90 | recall |
|---|---|---|---|---|
| XGBoost | 0.1289 | 0.1900 | 0.0000 | 0.2508 |
| GINe (base) | 0.0492 | 0.1344 | 0.0000 | 0.1746 |
| +reverse | 0.0290 | 0.0776 | 0.0017 | 0.0795 |
| +port | 0.0507 | 0.1248 | 0.0006 | 0.1591 |
| +ego | 0.0543 | 0.1505 | 0.0000 | 0.1691 |
| Multi-GNN (full) | 0.0290 | 0.0664 | 0.0000 | 0.0712 |
