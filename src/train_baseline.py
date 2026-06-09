"""CLI: обучение неграфового бейзлайна (XGBoost / LogReg) на Elliptic.

Пример:
    python -m src.train_baseline --config configs/elliptic_xgb.yaml

Логика: только размеченные узлы; сплит train/test — встроенный временно́й;
val выделяется стратифицированно из train; порог бинаризации фиксируется по val
и применяется к test; метрики и PR-кривая сохраняются в results/.
"""
from __future__ import annotations

import argparse
import os

from src.baselines import predict_scores, train_logreg, train_xgboost
from src.datasets import (
    CANONICAL_PATTERNS,
    build_edge_features,
    get_xy,
    load_elliptic,
    load_ibm_aml,
    make_val_split,
)
from src.metrics import evaluate, evaluate_per_group, pr_curve_figure
from src.utils import init_wandb, load_config, save_json, set_seed


def run(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    if config.get("dataset", {}).get("name") == "ibm_aml":
        return run_ibm_baseline(config)

    ds_cfg = config.get("dataset", {})
    data, meta = load_elliptic(root=ds_cfg.get("root", "data/elliptic"))
    pos = meta["positive_class"]
    print(f"[data] positive_class (illicit) = {pos}; "
          f"train={meta['splits']['train']}, test={meta['splits']['test']}")

    # Признаки и бинарные метки по встроенным временны́м маскам.
    X_train_full, y_train_full = get_xy(data, data.train_mask, pos)
    X_test, y_test = get_xy(data, data.test_mask, pos)

    # Стратифицированный val ВНУТРИ train (без временно́й утечки).
    val_fraction = float(config.get("val_fraction", 0.2))
    tr_pos, va_pos = make_val_split(y_train_full, val_fraction, seed)
    X_tr, y_tr = X_train_full[tr_pos], y_train_full[tr_pos]
    X_va, y_va = X_train_full[va_pos], y_train_full[va_pos]
    print(f"[split] train={len(y_tr)} (pos={int(y_tr.sum())}), "
          f"val={len(y_va)} (pos={int(y_va.sum())}), "
          f"test={len(y_test)} (pos={int(y_test.sum())})")

    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "xgboost").lower()
    params = dict(model_cfg.get("params", {}))
    params.setdefault("random_state", seed)

    if model_type == "xgboost":
        model = train_xgboost(X_tr, y_tr, X_va, y_va, params)
    elif model_type == "logreg":
        model = train_logreg(X_tr, y_tr, params)
    else:
        raise ValueError(f"Неизвестный model.type: {model_type!r} (xgboost|logreg)")

    # Порог фиксируем по val (максимум F1), применяем к test.
    val_scores = predict_scores(model, X_va)
    val_metrics = evaluate(y_va, val_scores, threshold=None)
    threshold = val_metrics["threshold"]

    test_scores = predict_scores(model, X_test)
    test_metrics = evaluate(y_test, test_scores, threshold=threshold)

    _print_metrics("VAL", val_metrics)
    _print_metrics("TEST", test_metrics)

    # Сохранение результатов.
    out_dir = config.get("output_dir", "results")
    name = config.get("output_name", f"elliptic_{model_type}")
    metrics_path = os.path.join(out_dir, f"{name}_metrics.json")
    pr_path = os.path.join(out_dir, f"{name}_pr_curve.png")

    result = {
        "config": config,
        "dataset_meta": meta,
        "model_type": model_type,
        "fixed_threshold_from_val": threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(result, metrics_path)
    pr_curve_figure(y_test, test_scores, pr_path)
    print(f"[saved] {metrics_path}\n[saved] {pr_path}")

    _log_wandb(config, name, val_metrics, test_metrics, pr_path)
    return result


def run_ibm_baseline(config: dict) -> dict:
    """XGBoost/LogReg на рёбрах IBM AML (edge-classification) + per-pattern.

    Признаки рёбер = build_edge_features (edge_attr + узлы-концы + параллельные
    рёбра). Сплит — встроенный временно́й (train/val/test маски рёбер). Порог
    фиксируется по val. Метрики: общая (AUC-PR, F1, recall@P) + по 8 паттернам.
    """
    ds_cfg = config.get("dataset", {})
    data, meta = load_ibm_aml(
        root=ds_cfg.get("root", "data/ibm_aml"),
        variant=ds_cfg.get("variant", "HI-Small"),
        max_rows=ds_cfg.get("max_rows"),
    )
    print(f"[data] IBM {meta['variant']}: {meta['num_edges']:,} рёбер, "
          f"illicit={meta['n_illicit']} ({meta['illicit_rate']*100:.3f}%); "
          f"splits={{train:{meta['splits']['train']['n']}, "
          f"val:{meta['splits']['val']['n']}, test:{meta['splits']['test']['n']}}}")

    X = build_edge_features(data)
    y = data.edge_label.numpy()
    tr, va, te = data.train_mask.numpy(), data.val_mask.numpy(), data.test_mask.numpy()
    print(f"[features] X={X.shape}; pos train/val/test = "
          f"{int(y[tr].sum())}/{int(y[va].sum())}/{int(y[te].sum())}")

    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "xgboost").lower()
    params = dict(model_cfg.get("params", {}))
    params.setdefault("random_state", int(config.get("seed", 42)))

    if model_type == "xgboost":
        model = train_xgboost(X[tr], y[tr], X[va], y[va], params)
    elif model_type == "logreg":
        model = train_logreg(X[tr], y[tr], params)
    else:
        raise ValueError(f"Неизвестный model.type: {model_type!r} (xgboost|logreg)")

    val_scores = predict_scores(model, X[va])
    val_metrics = evaluate(y[va], val_scores, threshold=None)
    threshold = val_metrics["threshold"]
    test_scores = predict_scores(model, X[te])
    test_metrics = evaluate(y[te], test_scores, threshold=threshold)

    # Разбивка по паттернам на test (позитивы типа vs все негативы).
    groups = list(CANONICAL_PATTERNS) + ["unknown"]
    per_pattern = evaluate_per_group(
        y[te], test_scores, data.edge_pattern[te], threshold, groups=groups
    )

    _print_metrics("VAL", val_metrics)
    _print_metrics("TEST", test_metrics)
    print("[per-pattern recall] " + ", ".join(
        f"{k}:{v['recall']:.2f}({v['n_detected']}/{v['n_pos']})" for k, v in per_pattern.items()))

    out_dir = config.get("output_dir", "results")
    name = config.get("output_name", f"ibm_{model_type}")
    metrics_path = os.path.join(out_dir, f"{name}_metrics.json")
    pr_path = os.path.join(out_dir, f"{name}_pr_curve.png")
    result = {
        "config": config,
        "dataset_meta": meta,
        "model_type": model_type,
        "fixed_threshold_from_val": threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "per_pattern": per_pattern,
    }
    save_json(result, metrics_path)
    pr_curve_figure(y[te], test_scores, pr_path)
    print(f"[saved] {metrics_path}\n[saved] {pr_path}")

    _log_wandb(config, name, val_metrics, test_metrics, pr_path)
    return result


def _print_metrics(tag: str, m: dict) -> None:
    print(
        f"[{tag}] AUC-PR={m['auc_pr']:.4f} ROC-AUC={m['roc_auc']:.4f} "
        f"F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
        f"R@P90={m['recall_at_precision_90']:.4f} thr={m['threshold']:.4f}"
    )


def _log_wandb(config: dict, run_name: str, val_metrics: dict, test_metrics: dict, pr_path: str) -> None:
    """Логирование бейзлайна в W&B через единый init_wandb (off по умолчанию)."""
    wb = init_wandb(config, run_name=run_name)
    if wb is None:
        return
    wb.log({f"val/{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})
    wb.log({f"test/{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})
    if os.path.exists(pr_path):
        wb.log({"pr_curve": wb.Image(pr_path)})
    wb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Бейзлайн на Elliptic (XGBoost/LogReg)")
    parser.add_argument("--config", required=True, help="путь к YAML-конфигу")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
