"""CLI: обучение GNN на Elliptic (GCN/GraphSAGE/GAT/GIN/PNA).

Тот же временно́й сплit, та же система метрик и фиксация порога по val, что и у
неграфового бейзлайна — для честного сравнения. Архитектура задаётся в конфиге
(model.type). W&B-логирование — опциональный крючок (по умолчанию выключен).

Примеры:
    python -m src.train --config configs/elliptic_gcn.yaml
    python -m src.train --config configs/elliptic_gat.yaml
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from src.datasets import load_elliptic, make_val_split
from src.metrics import evaluate, pr_curve_figure
from src.models import build_model, compute_degree_histogram
from src.utils import load_config, save_json, set_seed


def _build_masks(data, pos: int, val_fraction: float, seed: int):
    """Из train_mask выделить стратифицированный val; вернуть (train_sub, val)."""
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    y_train_bin = (data.y[train_idx] == pos).long().cpu().numpy()
    tr_pos, va_pos = make_val_split(y_train_bin, val_fraction, seed)

    n = data.num_nodes
    train_sub = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    train_sub[train_idx[tr_pos]] = True
    val_mask[train_idx[va_pos]] = True
    return train_sub, val_mask


def run(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))

    ds_cfg = config.get("dataset", {})
    data, meta = load_elliptic(root=ds_cfg.get("root", "data/elliptic"))
    pos = meta["positive_class"]
    y_bin = (data.y == pos).long()

    val_fraction = float(config.get("val_fraction", 0.2))
    train_sub, val_mask = _build_masks(data, pos, val_fraction, seed)
    test_mask = data.test_mask

    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "gcn").lower()
    m_params = model_cfg.get("params", {})

    # PNA требует гистограмму степеней обучающего графа.
    deg = compute_degree_histogram(data.edge_index, data.num_nodes) if model_type == "pna" else None

    data = data.to(device)
    y_bin = y_bin.to(device)
    train_sub, val_mask, test_mask = train_sub.to(device), val_mask.to(device), test_mask.to(device)
    if deg is not None:
        deg = deg.to(device)

    model = build_model(
        model_type,
        in_channels=data.num_features,
        hidden_channels=int(m_params.get("hidden_channels", 64)),
        out_channels=2,
        dropout=float(m_params.get("dropout", 0.5)),
        heads=int(m_params.get("heads", 8)),
        deg=deg,
    ).to(device)
    print(f"[model] {model_type} | params={sum(p.numel() for p in model.parameters()):,}")

    t_cfg = config.get("train", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(t_cfg.get("lr", 0.01)),
        weight_decay=float(t_cfg.get("weight_decay", 5e-4)),
    )
    epochs = int(t_cfg.get("epochs", 200))
    patience = int(t_cfg.get("patience", 20))

    # Веса классов против дисбаланса (~2% illicit).
    n_pos = int(y_bin[train_sub].sum())
    n_neg = int(train_sub.sum()) - n_pos
    class_weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device)

    wb = _maybe_init_wandb(config, model_type)

    best_val, best_state, bad = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_sub], y_bin[train_sub], weight=class_weight)
        loss.backward()
        optimizer.step()

        val_m = _eval_split(model, data, y_bin, val_mask)
        if wb is not None:
            wb.log({"epoch": epoch, "loss": float(loss.item()),
                    "val/auc_pr": val_m["auc_pr"], "val/f1": val_m["f1"]})
        if val_m["auc_pr"] > best_val:
            best_val, best_state, bad = val_m["auc_pr"], _clone_state(model), 0
        else:
            bad += 1
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | loss {loss.item():.4f} | val AUC-PR {val_m['auc_pr']:.4f}")
        if bad >= patience:
            print(f"early stop на эпохе {epoch} (best val AUC-PR {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_m = _eval_split(model, data, y_bin, val_mask, threshold=None)
    threshold = val_m["threshold"]
    test_m = _eval_split(model, data, y_bin, test_mask, threshold=threshold)
    print(f"[VAL ] AUC-PR={val_m['auc_pr']:.4f} F1={val_m['f1']:.4f}")
    print(f"[TEST] AUC-PR={test_m['auc_pr']:.4f} F1={test_m['f1']:.4f} "
          f"R@P90={test_m['recall_at_precision_90']:.4f}")

    out_dir = config.get("output_dir", "results")
    name = config.get("output_name", f"elliptic_{model_type}")
    metrics_path = os.path.join(out_dir, f"{name}_metrics.json")
    pr_path = os.path.join(out_dir, f"{name}_pr_curve.png")

    test_scores = _scores(model, data)[test_mask.cpu().numpy()]
    save_json(
        {"config": config, "model_type": model_type, "dataset_meta": meta,
         "fixed_threshold_from_val": threshold, "val_metrics": val_m, "test_metrics": test_m},
        metrics_path,
    )
    pr_curve_figure(y_bin[test_mask].cpu().numpy(), test_scores, pr_path)
    print(f"[saved] {metrics_path}\n[saved] {pr_path}")

    if wb is not None:
        wb.log({f"test/{k}": v for k, v in test_m.items() if isinstance(v, (int, float))})
        wb.finish()
    return {"val_metrics": val_m, "test_metrics": test_m}


@torch.no_grad()
def _scores(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    return F.softmax(out, dim=1)[:, 1].cpu().numpy()


@torch.no_grad()
def _eval_split(model, data, y_bin, mask, threshold=None) -> dict:
    model.eval()
    out = model(data.x, data.edge_index)
    prob = F.softmax(out, dim=1)[:, 1]
    y_true = y_bin[mask].cpu().numpy()
    y_score = prob[mask].cpu().numpy()
    return evaluate(y_true, y_score, threshold=threshold)


def _clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _maybe_init_wandb(config: dict, model_type: str):
    """Опциональная инициализация W&B (выключена по умолчанию)."""
    wb_cfg = config.get("wandb", {})
    if not wb_cfg.get("enabled", False):
        return None
    try:
        import wandb

        wandb.init(
            project=wb_cfg.get("project", "gnn-aml"),
            entity=wb_cfg.get("entity"),
            name=f"elliptic-{model_type}",
            config=config,
        )
        return wandb
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] пропущено: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение GNN на Elliptic")
    parser.add_argument("--config", required=True, help="путь к YAML-конфигу")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
