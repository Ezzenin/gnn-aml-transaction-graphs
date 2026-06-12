# IBM AML HI-Small (test): бейзлайн + ablation Multi-GNN (без norm_time)

Главные метрики: AUC-PR и F1 (позитив = laundering). Ablation: вклад
адаптаций reverse / port / ego поверх базовой GINe (RQ2).

| variant | auc_pr | f1 | recall_at_precision_90 | recall |
|---|---|---|---|---|
| XGBoost | 0.2398 | 0.2976 | 0.0217 | 0.3103 |
| GINe (base) | 0.0442 | 0.1199 | 0.0000 | 0.1357 |
| +reverse | 0.0255 | 0.0510 | 0.0000 | 0.5028 |
| +port | 0.0354 | 0.1002 | 0.0000 | 0.1051 |
| +ego | 0.0491 | 0.1245 | 0.0000 | 0.1418 |
| Multi-GNN (full) | 0.0125 | 0.0332 | 0.0000 | 0.2297 |
