# Обзор литературы

> Заготовка. Конспект ≥8 статей по шаблону. Выходные данные перепроверить
> и оформить по ГОСТ Р 7.0.5-2008.

Шаблон строки: `Источник | Задача | Метод | Датасет | Метрика | Результат | Релевантность`

| Источник | Задача | Метод | Датасет | Метрика | Результат | Релевантность |
|---|---|---|---|---|---|---|
| Weber et al., 2019 (arXiv:1908.02591) | node-class. (illicit/licit) | GCN + табличные бейзлайны | Elliptic | F1-illicit | граф vs табличка; деревья конкурентны GNN | базовый датасет + бейзлайн |
| Altman et al., 2023 (NeurIPS D&B, arXiv:2306.16424) | edge-class. (AML) | синтетика + бенчмарк | IBM AML (AMLworld) | F1-minority | реалистичный размеченный AML-датасет по паттернам | датасет цепочек |
| Egressy et al., 2024 (AAAI, arXiv:2306.11586) | детекция подграфов/цепочек | Multi-GNN: ego-IDs + port numbering + reverse MP | синтетика, AML | F1-minority | +до 30% F1 на AML; provably powerful | ключевой метод КР |
| Kipf & Welling, 2017 (ICLR, arXiv:1609.02907) | node-class. | GCN | Cora/Citeseer и др. | accuracy | базовая spectral GNN | бейзлайн-архитектура |
| Veličković et al., 2018 (ICLR) | node-class. | GAT (attention) | citation graphs | accuracy | внимание на рёбрах | бейзлайн-архитектура |
| Xu et al., 2019 (ICLR) | graph-class. | GIN | graph benchmarks | accuracy | мощность ≈ WL-тест | бейзлайн-архитектура |
| Pareja et al., 2020 (AAAI) | temporal node-class. | EvolveGCN | Elliptic и др. | F1/AUROC | роль времени в графе | темпоральный вариант |
| Deprez et al., 2024 (arXiv:2405.19383) | обзор + эксперименты | network analytics для AML | разные | разные | систематический обзор методов AML | контекст/обзор |
| Corso et al., 2020 (NeurIPS) | graph-class. | PNA | benchmarks | разные | агрегация по соседям | бейзлайн-архитектура |
| Fey & Lenssen, 2019 (arXiv:1903.02428) | библиотека | PyTorch Geometric | — | — | инструментарий GNN | стек реализации |

## Заметки / выводы обзора
<!-- TODO: обозначить пробел, который закрывает работа -->
