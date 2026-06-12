"""Сводное сравнение моделей на Elliptic: таблица + bar-chart по results/.

Использование:
    python -m src.compare --run     # прогнать все конфиги, затем собрать сводку
    python -m src.compare           # только собрать сводку из готовых results/*.json

Собирает test-метрики из results/*_metrics.json в таблицу (CSV + Markdown) и
строит сравнительный график AUC-PR / F1. Главные метрики для несбалансированной
задачи — AUC-PR и F1 по позитивному классу.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# Конфиги бейзлайнов и GNN (порядок = порядок прогона).
BASELINE_CONFIGS = [
    "configs/elliptic_xgb.yaml",
    "configs/elliptic_logreg.yaml",
]
GNN_CONFIGS = [
    "configs/elliptic_gcn.yaml",
    "configs/elliptic_sage.yaml",
    "configs/elliptic_gat.yaml",
    "configs/elliptic_gin.yaml",
    "configs/elliptic_pna.yaml",
]

METRIC_KEYS = ["auc_pr", "f1", "recall_at_precision_90", "roc_auc", "precision", "recall"]

# ── IBM AML edge-classification: бейзлайн + ablation-сетка Multi-GNN (Фаза D) ──
IBM_XGB_CONFIG = "configs/ibm_xgb.yaml"
IBM_GNN_CONFIGS = [
    "configs/ibm_gine.yaml",        # base (все флаги выкл)
    "configs/ibm_gine_rev.yaml",    # + reverse MP
    "configs/ibm_gine_port.yaml",   # + port numbering
    "configs/ibm_gine_ego.yaml",    # + ego-IDs
    "configs/ibm_multignn.yaml",    # full (reverse + port + ego)
]
# (output_name, человекочитаемый ярлык) — порядок строк сводки/графика.
IBM_VARIANTS = [
    ("ibm_xgboost", "XGBoost"),
    ("ibm_gine", "GINe (base)"),
    ("ibm_gine_rev", "+reverse"),
    ("ibm_gine_port", "+port"),
    ("ibm_gine_ego", "+ego"),
    ("ibm_multignn", "Multi-GNN (full)"),
]
IBM_METRIC_KEYS = ["auc_pr", "f1", "recall_at_precision_90", "recall"]
BASE_LABEL = "GINe (base)"
GNN_ORDER = [BASE_LABEL, "+reverse", "+port", "+ego", "Multi-GNN (full)"]


def run_all() -> None:
    """Прогнать все бейзлайны и GNN по их конфигам."""
    from src.train import run as run_gnn
    from src.train_baseline import run as run_baseline
    from src.utils import load_config

    for cfg_path in BASELINE_CONFIGS:
        print(f"\n===== RUN baseline: {cfg_path} =====")
        run_baseline(load_config(cfg_path))
    for cfg_path in GNN_CONFIGS:
        print(f"\n===== RUN gnn: {cfg_path} =====")
        run_gnn(load_config(cfg_path))


def collect(results_dir: str = "results") -> list[dict]:
    """Собрать test-метрики из всех *_metrics.json (кроме служебных)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_metrics.json"))):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        test = d.get("test_metrics", {})
        model = d.get("model_type") or os.path.basename(path).replace("_metrics.json", "")
        row = {"model": model, "file": os.path.basename(path)}
        row.update({k: test.get(k) for k in METRIC_KEYS})
        rows.append(row)
    # Сортировка по AUC-PR убыванию (главная метрика).
    rows.sort(key=lambda r: (r.get("auc_pr") is not None, r.get("auc_pr") or 0), reverse=True)
    return rows


def write_table(rows: list[dict], results_dir: str = "results") -> None:
    """Записать сводку в CSV и Markdown, напечатать в консоль."""
    import csv

    csv_path = os.path.join(results_dir, "comparison.csv")
    md_path = os.path.join(results_dir, "comparison.md")
    cols = ["model"] + METRIC_KEYS

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join([str(r["model"])] + [fmt(r.get(k)) for k in METRIC_KEYS]) + " |")
    md = "# Сравнение моделей на Elliptic (test)\n\nГлавные метрики: AUC-PR и F1 (позитив = illicit).\n\n" + "\n".join(lines) + "\n"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + "\n".join(lines))
    print(f"\n[saved] {csv_path}\n[saved] {md_path}")


def plot(rows: list[dict], results_dir: str = "results") -> None:
    """Сравнительный bar-chart по AUC-PR и F1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = [r["model"] for r in rows]
    auc_pr = [r.get("auc_pr") or 0 for r in rows]
    f1 = [r.get("f1") or 0 for r in rows]

    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.1), 4.5))
    ax.bar(x - width / 2, auc_pr, width, label="AUC-PR", color="#4c72b0")
    ax.bar(x + width / 2, f1, width, label="F1", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Elliptic (test): сравнение моделей")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(results_dir, "comparison_models.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


# ─────────────────────────── IBM AML (Фаза D) ───────────────────────────
def run_ibm() -> None:
    """Прогнать IBM-сетку: XGBoost-бейзлайн (если нет) + 5 edge-GNN вариантов.

    Тяжёлая часть (CUDA) — выполняется на ПК. XGBoost считается один раз и
    переиспользуется; GNN-варианты — это ablation (base + 3 одиночные + full).
    """
    from src.train_baseline import run as run_baseline
    from src.train_edge import run as run_edge
    from src.utils import load_config

    if not os.path.exists("results/ibm_xgboost_metrics.json"):
        print(f"\n===== RUN ibm baseline: {IBM_XGB_CONFIG} =====")
        run_baseline(load_config(IBM_XGB_CONFIG))
    for cfg_path in IBM_GNN_CONFIGS:
        print(f"\n===== RUN ibm edge-GNN: {cfg_path} =====")
        run_edge(load_config(cfg_path))


def collect_ibm(results_dir: str = "results") -> list[dict]:
    """Собрать test-метрики IBM-вариантов в порядке IBM_VARIANTS (пропуская отсутствующие)."""
    rows = []
    for name, label in IBM_VARIANTS:
        path = os.path.join(results_dir, f"{name}_metrics.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        test = d.get("test_metrics", {})
        row = {"variant": label, "name": name}
        row.update({k: test.get(k) for k in IBM_METRIC_KEYS})
        rows.append(row)
    return rows


def write_ibm_table(rows: list[dict], results_dir: str = "results") -> None:
    """Сводная таблица IBM (CSV + Markdown): XGBoost vs base GNN vs +адаптации."""
    import csv

    cols = ["variant"] + IBM_METRIC_KEYS
    csv_path = os.path.join(results_dir, "ibm_comparison.csv")
    md_path = os.path.join(results_dir, "ibm_comparison.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join([str(r["variant"])] + [fmt(r.get(k)) for k in IBM_METRIC_KEYS]) + " |")
    md = ("# IBM AML HI-Small (test): бейзлайн + ablation Multi-GNN\n\n"
          "Главные метрики: AUC-PR и F1 (позитив = laundering). Ablation: вклад\n"
          "адаптаций reverse / port / ego поверх базовой GINe (RQ2).\n\n"
          + "\n".join(lines) + "\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + "\n".join(lines))
    print(f"\n[saved] {csv_path}\n[saved] {md_path}")


def plot_ablation(rows: list[dict], results_dir: str = "results") -> None:
    """Bar-chart ablation: F1 и AUC-PR по GNN-вариантам + пунктир уровня base.

    Показывает вклад каждой адаптации относительно базовой GINe (главный график
    отчёта по RQ2). XGBoost — как горизонтальная референс-линия, если есть.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by = {r["variant"]: r for r in rows}
    variants = [v for v in GNN_ORDER if v in by]
    if BASE_LABEL not in variants:
        print("[ablation] нет базовой GINe — график пропущен")
        return

    f1 = [by[v].get("f1") or 0 for v in variants]
    auc = [by[v].get("auc_pr") or 0 for v in variants]
    base_f1 = by[BASE_LABEL].get("f1") or 0
    base_auc = by[BASE_LABEL].get("auc_pr") or 0

    x = np.arange(len(variants))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(variants) * 1.3), 4.8))
    ax.bar(x - width / 2, auc, width, label="AUC-PR", color="#4c72b0")
    ax.bar(x + width / 2, f1, width, label="F1-minority", color="#c44e52")
    ax.axhline(base_auc, ls="--", lw=1, color="#4c72b0", alpha=0.6)
    ax.axhline(base_f1, ls="--", lw=1, color="#c44e52", alpha=0.6)

    # Подписать дельту к base над барами GNN-адаптаций.
    for i, v in enumerate(variants):
        if v == BASE_LABEL:
            continue
        d_auc, d_f1 = auc[i] - base_auc, f1[i] - base_f1
        ax.annotate(f"{d_auc:+.3f}", (x[i] - width / 2, auc[i]), ha="center", va="bottom", fontsize=7, color="#26456e")
        ax.annotate(f"{d_f1:+.3f}", (x[i] + width / 2, f1[i]), ha="center", va="bottom", fontsize=7, color="#7a2c2f")

    if "XGBoost" in by:
        ax.axhline(by["XGBoost"].get("auc_pr") or 0, ls=":", lw=1.2, color="black", alpha=0.7,
                   label=f"XGBoost AUC-PR ({by['XGBoost'].get('auc_pr') or 0:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.set_ylabel("score")
    ax.set_title("IBM AML: ablation мультиграфовых адаптаций (пунктир = base GINe)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(results_dir, "ablation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


def summarize_ibm(results_dir: str = "results") -> None:
    """Собрать IBM-сводку: таблица + ablation-график из готовых results/."""
    rows = collect_ibm(results_dir)
    if not rows:
        print("Нет IBM-результатов в", results_dir, "(прогони --run-ibm на ПК с CUDA)")
        return
    write_ibm_table(rows, results_dir)
    plot_ablation(rows, results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение моделей (Elliptic + IBM AML)")
    parser.add_argument("--run", action="store_true", help="прогнать все Elliptic-конфиги")
    parser.add_argument("--run-ibm", action="store_true",
                        help="прогнать IBM-сетку (XGBoost + ablation edge-GNN) — нужен CUDA (ПК)")
    parser.add_argument("--ibm", action="store_true",
                        help="собрать IBM-сводку (таблица + ablation.png) из готовых results/")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    # IBM-режим самостоятелен (Elliptic-сводка — отдельным вызовом без флагов).
    if args.run_ibm or args.ibm:
        if args.run_ibm:
            run_ibm()
        summarize_ibm(args.results_dir)
        return

    if args.run:
        run_all()
    rows = collect(args.results_dir)
    if not rows:
        print("Нет результатов в", args.results_dir)
        return
    write_table(rows, args.results_dir)
    plot(rows, args.results_dir)


if __name__ == "__main__":
    main()
