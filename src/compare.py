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


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение моделей на Elliptic")
    parser.add_argument("--run", action="store_true", help="сначала прогнать все конфиги")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

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
