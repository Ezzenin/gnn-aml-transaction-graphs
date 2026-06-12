"""CLI: обучение edge-GNN (GINe) на IBM AML через LinkNeighborLoader.

Edge-classification на направленном мультиграфе «узел=счёт, ребро=транзакция».
Дисбаланс ~0.1%: train-набор = все illicit-рёбра + подвыборка негативов
(neg_ratio). Полный граф используется для сэмплинга соседей. Порог фиксируется
по val, метрики (общая + per-pattern) считаются на полном test.

Пример:
    python -m src.train_edge --config configs/ibm_gine.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from src.datasets import CANONICAL_PATTERNS, load_ibm_aml
from src.metrics import evaluate, evaluate_per_group, pr_curve_figure
from src.models import build_edge_model
from src.utils import init_wandb, load_config, resolve_device, save_json, set_seed


def _context_data(data, edge_mask, reverse_mp: bool) -> Data:
    """Message-passing граф ТОЛЬКО из рёбер edge_mask (антиутечка по времени).

    P0.1: контекст для сид-ребра не должен содержать будущих транзакций. Train/val
    классифицируются на train-контексте, test — на train+val. Узлы — то же
    пространство (num_nodes), сид-рёбра передаются отдельно через edge_label_index,
    в edge_index контекста их быть не обязано. reverse_mp применяется к контексту.
    """
    ei = data.edge_index[:, edge_mask]
    ea = data.edge_attr[edge_mask].float()
    if reverse_mp:
        from src.models import add_reverse_edges

        ei, ea = add_reverse_edges(ei, ea)
    d = Data(x=data.x.float(), edge_index=ei, edge_attr=ea)
    d.num_nodes = data.num_nodes
    return d


def _balanced_train_edges(label, train_mask, neg_ratio: int, seed: int) -> np.ndarray:
    """Индексы train-рёбер: все позитивы + neg_ratio×позитивов негативов."""
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(train_mask)
    pos = tr[label[tr] == 1]
    neg = tr[label[tr] == 0]
    n_neg = min(len(neg), neg_ratio * max(len(pos), 1))
    neg_sample = rng.choice(neg, size=n_neg, replace=False)
    idx = np.concatenate([pos, neg_sample])
    rng.shuffle(idx)
    return idx


def _eval_edges(model, ctx, data, edge_idx, num_neighbors, batch_size, device) -> tuple:
    """Прогнать модель по рёбрам edge_idx на контексте ctx → (y_true, y_score).

    Признаки самого классифицируемого ребра (P0.2) берутся из СЫРОГО data.edge_attr
    по batch.input_id (индекс сид-ребра в переданном edge_label_index).
    """
    seed_attr = data.edge_attr[edge_idx].float()
    loader = LinkNeighborLoader(
        ctx, num_neighbors=num_neighbors,
        edge_label_index=data.edge_index[:, edge_idx],
        edge_label=data.edge_label[edge_idx],
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ela = seed_attr[batch.input_id.cpu()].to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, ela)
            ss.append(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            ys.append(batch.edge_label.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ss)


def run(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    print(f"[device] {device}")

    ds_cfg = config.get("dataset", {})
    data, meta = load_ibm_aml(
        root=ds_cfg.get("root", "data/ibm_aml"),
        variant=ds_cfg.get("variant", "HI-Small"),
        max_rows=ds_cfg.get("max_rows"),
        include_time=ds_cfg.get("include_time", True),
    )
    label = data.edge_label.numpy()

    t_cfg = config.get("train", {})
    num_neighbors = list(t_cfg.get("num_neighbors", [10, 10]))
    batch_size = int(t_cfg.get("batch_size", 2048))
    neg_ratio = int(t_cfg.get("neg_ratio", 20))
    val_neg_cap = int(t_cfg.get("val_neg_cap", 50000))
    epochs = int(t_cfg.get("epochs", 15))
    patience = int(t_cfg.get("patience", 5))

    # Режим обучения: сабсэмпл негативов (neg_ratio) ИЛИ все train-рёбра
    # (full_data, как у Egressy — литература: вероятно главный фактор разрыва).
    full_data = bool(t_cfg.get("full_data", False))
    tr_all = np.flatnonzero(data.train_mask.numpy())
    if full_data:
        train_idx = tr_all.copy()
        np.random.default_rng(seed).shuffle(train_idx)
        n_pos = int(label[tr_all].sum())
        pos_w = float(t_cfg.get("pos_weight", (len(tr_all) - n_pos) / max(n_pos, 1)))
        print(f"[train] FULL-DATA: все {len(train_idx):,} train-рёбер (pos={n_pos}), "
              f"pos_weight={pos_w:.1f}")
    else:
        train_idx = _balanced_train_edges(label, data.train_mask.numpy(), neg_ratio, seed)
        pos_w = float(t_cfg.get("pos_weight", neg_ratio))
        print(f"[train] сид-рёбер {len(train_idx):,} (pos={int(label[train_idx].sum())}); "
              f"neg_ratio={neg_ratio}, pos_weight={pos_w:.1f}")
    print(f"[data] IBM {meta['variant']}: illicit={meta['n_illicit']} ({meta['illicit_rate']*100:.3f}%); "
          f"num_neighbors={num_neighbors}, batch={batch_size}")

    # Подвыборка val для ранней остановки (все позитивы + ограниченные негативы).
    rng = np.random.default_rng(seed)
    va_all = np.flatnonzero(data.val_mask.numpy())
    va_pos = va_all[label[va_all] == 1]
    va_neg = va_all[label[va_all] == 0]
    va_neg = rng.choice(va_neg, size=min(len(va_neg), val_neg_cap), replace=False)
    val_sample_idx = np.concatenate([va_pos, va_neg])

    model_cfg = config.get("model", {})
    m_params = model_cfg.get("params", {})
    reverse_mp = bool(m_params.get("reverse_mp", False))

    # P0.1: строгие контексты — train для train/val-сидов, train+val для test
    # (никакое сид-ребро не видит будущие транзакции). reverse_mp применяется к
    # каждому контексту (add_reverse_edges внутри _context_data).
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    train_ctx = _context_data(data, train_mask, reverse_mp)
    trainval_ctx = _context_data(data, train_mask | val_mask, reverse_mp)
    if reverse_mp:
        print(f"[reverse_mp] контекст двунаправленный (train: {train_ctx.edge_index.shape[1]:,} рёбер, "
              f"edge_attr {train_ctx.edge_attr.shape[1]} фич)")
    in_edge_eff = train_ctx.edge_attr.shape[1]          # контекст (+reverse-флаг)
    in_edge_label = data.edge_attr.shape[1]             # сырые признаки сид-ребра (P0.2)
    seed_attr_train = data.edge_attr[train_idx].float()

    model = build_edge_model(
        model_cfg.get("type", "gine"),
        in_node=data.x.shape[1], in_edge=in_edge_eff, in_edge_label=in_edge_label,
        hidden=int(m_params.get("hidden", 64)),
        num_layers=int(m_params.get("num_layers", 2)),
        dropout=float(m_params.get("dropout", 0.5)),
        reverse_mp=reverse_mp,
        ports=bool(m_params.get("ports", False)),
        ego_ids=bool(m_params.get("ego_ids", False)),
    ).to(device)
    print(f"[model] {model_cfg.get('type','gine')} | params={sum(p.numel() for p in model.parameters()):,}")

    train_loader = LinkNeighborLoader(
        train_ctx, num_neighbors=num_neighbors,
        edge_label_index=data.edge_index[:, train_idx],
        edge_label=data.edge_label[train_idx],
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(t_cfg.get("lr", 0.005)),
                                 weight_decay=float(t_cfg.get("weight_decay", 5e-4)))
    # Вес позитива в лоссе (pos_w задан выше по режиму обучения).
    class_weight = torch.tensor([1.0, float(pos_w)], device=device)

    wb = init_wandb(config, run_name=config.get("output_name", "ibm_gine"))

    best_val, best_state, bad = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            ela = seed_attr_train[batch.input_id.cpu()].to(device)  # признаки сид-ребра (P0.2)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, ela)
            loss = F.cross_entropy(out, batch.edge_label.long(), weight=class_weight)
            loss.backward()
            optimizer.step()
            total += float(loss.item())

        # val классифицируется на TRAIN-контексте (P0.1: без будущего).
        yv, sv = _eval_edges(model, train_ctx, data, val_sample_idx, num_neighbors, batch_size, device)
        vm = evaluate(yv, sv, threshold=None)
        print(f"epoch {epoch:2d} | loss {total/len(train_loader):.4f} | val(sample) AUC-PR {vm['auc_pr']:.4f}")
        if wb is not None:
            wb.log({"epoch": epoch, "loss": total / len(train_loader),
                    "val/auc_pr": vm["auc_pr"], "val/f1": vm["f1"]})
        if vm["auc_pr"] > best_val:
            best_val, best_state, bad = vm["auc_pr"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            print(f"early stop на эпохе {epoch} (best val AUC-PR {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Порог — на ПОЛНОМ val (train-контекст); метрики — на ПОЛНОМ test (train+val).
    print("[eval] полный val (порог, train-контекст) и полный test (train+val-контекст)...")
    yv_full, sv_full = _eval_edges(model, train_ctx, data, np.flatnonzero(val_mask),
                                   num_neighbors, batch_size, device)
    val_metrics = evaluate(yv_full, sv_full, threshold=None)
    threshold = val_metrics["threshold"]

    test_idx = np.flatnonzero(data.test_mask.numpy())
    yt, st = _eval_edges(model, trainval_ctx, data, test_idx, num_neighbors, batch_size, device)
    test_metrics = evaluate(yt, st, threshold=threshold)
    groups = list(CANONICAL_PATTERNS) + ["unknown"]
    per_pattern = evaluate_per_group(yt, st, data.edge_pattern[test_idx], threshold, groups=groups)

    print(f"[VAL ] AUC-PR={val_metrics['auc_pr']:.4f} F1={val_metrics['f1']:.4f}")
    print(f"[TEST] AUC-PR={test_metrics['auc_pr']:.4f} F1={test_metrics['f1']:.4f} "
          f"R@P90={test_metrics['recall_at_precision_90']:.4f}")
    print("[per-pattern recall] " + ", ".join(
        f"{k}:{v['recall']:.2f}({v['n_detected']}/{v['n_pos']})" for k, v in per_pattern.items()))

    out_dir = config.get("output_dir", "results")
    name = config.get("output_name", "ibm_gine")
    metrics_path = os.path.join(out_dir, f"{name}_metrics.json")
    pr_path = os.path.join(out_dir, f"{name}_pr_curve.png")
    result = {
        "config": config, "dataset_meta": meta, "model_type": model_cfg.get("type", "gine"),
        "fixed_threshold_from_val": threshold,
        "val_metrics": val_metrics, "test_metrics": test_metrics, "per_pattern": per_pattern,
    }
    save_json(result, metrics_path)
    pr_curve_figure(yt, st, pr_path)
    print(f"[saved] {metrics_path}\n[saved] {pr_path}")

    ckpt_dir = config.get("checkpoint_dir", "checkpoints")
    if config.get("save_checkpoint", True):
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = os.path.join(ckpt_dir, f"{name}.pt")
        torch.save({"state_dict": model.state_dict(), "config": config,
                    "in_node": data.x.shape[1], "in_edge": in_edge_eff,
                    "in_edge_label": in_edge_label, "threshold": threshold}, ckpt)
        print(f"[saved] {ckpt}")

    if wb is not None:
        wb.log({f"test/{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})
        if os.path.exists(pr_path):
            wb.log({"pr_curve": wb.Image(pr_path)})
        wb.finish()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение edge-GNN на IBM AML")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
