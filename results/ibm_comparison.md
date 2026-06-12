# IBM AML HI-Small (test): бейзлайн + ablation Multi-GNN

Главные метрики: AUC-PR и F1 (позитив = laundering). Ablation: вклад
адаптаций reverse / port / ego поверх базовой GINe (RQ2).

| variant | auc_pr | f1 | recall_at_precision_90 | recall |
|---|---|---|---|---|
| XGBoost | 0.1289 | 0.1900 | 0.0000 | 0.2508 |
| GINe (base) | 0.0806 | 0.1441 | 0.0006 | 0.1296 |
