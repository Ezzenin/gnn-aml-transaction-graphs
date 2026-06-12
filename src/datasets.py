"""Загрузчик датасета Elliptic (Bitcoin) через PyTorch Geometric.

Elliptic: node-classification, размеченные узлы illicit/licit + большой пласт
unknown. Сплит train/test — встроенный временно́й (ранние шаги → train, поздние
→ test). positive_class (illicit) определяется программно как миноритарный
размеченный класс — не хардкодим целое значение метки.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import torch
from torch_geometric.datasets import EllipticBitcoinDataset

# Ожидаемые размеры PyG-версии Elliptic (для предупреждений, не для падения).
EXPECTED = {"num_nodes": 203769, "num_edges": 234355, "num_features": 165}


def load_elliptic(root: str = "data/elliptic") -> Tuple["torch.Tensor", dict]:
    """Загрузить Elliptic и вернуть (data, meta).

    data — единый граф PyG: data.x, data.edge_index, data.y,
           data.train_mask, data.test_mask.
    meta — dict со статистикой: num_nodes, num_edges, num_features,
           label_counts (по всем узлам), positive_class (illicit, миноритарный),
           размеры и баланс классов в train/test.
    """
    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]

    # Размеченные узлы = объединение train/test масок (unknown исключён масками).
    labeled_mask = data.train_mask | data.test_mask
    vals, counts = torch.unique(data.y[labeled_mask], return_counts=True)
    # positive = illicit = размеченный класс с наименьшим числом узлов.
    positive_class = int(vals[torch.argmin(counts)].item())

    all_vals, all_counts = torch.unique(data.y, return_counts=True)
    label_counts = {int(v): int(c) for v, c in zip(all_vals.tolist(), all_counts.tolist())}

    meta = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(data.num_features),
        "label_counts": label_counts,
        "positive_class": positive_class,
        "n_labeled": int(labeled_mask.sum()),
        "splits": {
            "train": _split_stats(data, data.train_mask, positive_class),
            "test": _split_stats(data, data.test_mask, positive_class),
        },
    }

    _warn_on_mismatch(meta)
    return data, meta


def _split_stats(data, mask, positive_class: int) -> dict:
    """Статистика по сплиту: размер, число pos/neg, доля позитивного класса."""
    y = data.y[mask]
    n = int(mask.sum())
    n_pos = int((y == positive_class).sum())
    n_neg = n - n_pos
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": (n_pos / n) if n else 0.0,
    }


def _warn_on_mismatch(meta: dict) -> None:
    """Предупредить, если размеры отличаются от ожидаемых (другая версия PyG)."""
    for key, expected in EXPECTED.items():
        actual = meta[key]
        if actual != expected:
            warnings.warn(
                f"Elliptic {key}={actual}, ожидалось {expected}. "
                f"Возможно, иная версия PyG/датасета — проверьте на рантайме.",
                stacklevel=2,
            )


def get_xy(data, mask, positive_class: int):
    """Признаки и бинарные метки (1 = positive/illicit) для узлов под маской."""
    X = data.x[mask].cpu().numpy()
    y = (data.y[mask] == positive_class).long().cpu().numpy()
    return X, y


def make_val_split(y_binary, val_fraction: float = 0.2, seed: int = 42):
    """Стратифицированно разбить позиции [0..n) на train/val по меткам y_binary.

    Возвращает (train_pos, val_pos) — массивы позиций внутри переданной выборки.
    Случайный сплит делается ТОЛЬКО внутри train-узлов (временна́я утечка
    исключена: train-узлы — это ранние временны́е шаги).
    """
    from sklearn.model_selection import train_test_split

    positions = np.arange(len(y_binary))
    train_pos, val_pos = train_test_split(
        positions,
        test_size=val_fraction,
        stratify=y_binary,
        random_state=seed,
    )
    return train_pos, val_pos


# ──────────────────────────── IBM AML (AMLworld) ────────────────────────────
# Колонки HI-Small_Trans.csv (вторая "Account" → pandas переименует в "Account.1").
IBM_COLS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
]
IBM_KEY_COLS = IBM_COLS[:10]  # ключ транзакции = всё, кроме Is Laundering
CANONICAL_PATTERNS = [
    "fan_out", "fan_in", "gather_scatter", "scatter_gather",
    "cycle", "random", "bipartite", "stack",
]


def _normalize_pattern(raw: str) -> str:
    """Привести имя паттерна из *_Patterns.txt к каноническому виду.

    Пример: 'FAN-OUT:  Max 16-degree Fan-Out' -> 'fan_out';
            'SCATTER-GATHER' -> 'scatter_gather'.
    """
    base = raw.split(":")[0].strip().lower().replace("-", "_").replace(" ", "_")
    return base


def _ibm_edge_key(fields) -> str:
    """Стабильный ключ транзакции из первых 10 CSV-полей (без Is Laundering).

    ЕДИНАЯ функция для парсера паттернов и загрузчика — гарантирует, что ключи
    совпадают байт-в-байт (иначе привязка паттернов потеряется).
    """
    return ",".join(str(f) for f in fields[:10])


def parse_ibm_patterns(path: str) -> dict:
    """Распарсить HI-Small_Patterns.txt в карту ключ_транзакции -> тип паттерна.

    Файл состоит из блоков:
        BEGIN LAUNDERING ATTEMPT - <TYPE>
        <CSV-строки транзакций>
        END LAUNDERING ATTEMPT
    <TYPE> нормализуется к 8 каноническим типам. При пересечении паттернов
    сохраняется первое присвоение (детерминированно).
    """
    mapping: dict = {}
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                raw = line.split(" - ", 1)[1] if " - " in line else line
                current = _normalize_pattern(raw)
            elif line.startswith("END LAUNDERING ATTEMPT"):
                current = None
            elif line and current is not None:
                key = _ibm_edge_key(line.split(","))
                mapping.setdefault(key, current)  # первое присвоение
    return mapping


def load_ibm_aml(
    root: str = "data/ibm_aml",
    variant: str = "HI-Small",
    max_rows: Optional[int] = None,
    test_fraction: float = 0.2,
    val_fraction: float = 0.15,
    include_time: bool = True,
) -> Tuple["object", dict]:
    """Загрузить IBM AML (variant) как направленный мультиграф PyG (edge-classification).

    Узел = счёт (ключ f"{bank}_{account}"). Ребро = транзакция (source -> dest).
    edge_attr: log1p(Amount Paid/Received), label-кодированные Receiving/Payment
    Currency и Payment Format, нормированное время.
    edge_label = Is Laundering (0/1). edge_pattern = тип паттерна для позитивов
    ('none' для негативов, 'unknown' для illicit без совпадения в Patterns.txt).
    Узловые признаки (degree + агрегаты сумм) считаются ТОЛЬКО по train-рёбрам
    (антиутечка). Сплит — строго временно́й (train=ранние t, val=поздняя доля train,
    test=поздние t). max_rows — для быстрых smoke-тестов (не грузить весь датасет).

    Возвращает (data, meta).
    """
    import os

    import pandas as pd
    import torch

    trans_path = os.path.join(root, f"{variant}_Trans.csv")
    patterns_path = os.path.join(root, f"{variant}_Patterns.txt")

    # dtype=str: сохраняем исходные токены, чтобы ключ совпал с Patterns.txt.
    df = pd.read_csv(trans_path, dtype=str, nrows=max_rows)
    df.columns = IBM_COLS  # фиксируем имена (две колонки "Account")
    n_edges = len(df)

    # Узлы: f"{bank}_{account}" для отправителя и получателя.
    from_node = df["From Bank"] + "_" + df["Account"]
    to_node = df["To Bank"] + "_" + df["Account.1"]
    codes, uniques = pd.factorize(pd.concat([from_node, to_node], ignore_index=True))
    src = codes[:n_edges].astype(np.int64)
    dst = codes[n_edges:].astype(np.int64)
    num_nodes = len(uniques)

    # Числовые поля рёбер.
    amt_paid = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0.0).to_numpy(float)
    amt_recv = pd.to_numeric(df["Amount Received"], errors="coerce").fillna(0.0).to_numpy(float)
    recv_cur, recv_cur_vocab = pd.factorize(df["Receiving Currency"])
    pay_cur, pay_cur_vocab = pd.factorize(df["Payment Currency"])
    fmt, fmt_vocab = pd.factorize(df["Payment Format"])
    # datetime64[s] → честные unix-секунды (pandas 3.0 по умолчанию хранит в us).
    ts = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M").to_numpy().astype("datetime64[s]").astype(np.int64)
    label = pd.to_numeric(df["Is Laundering"], errors="coerce").fillna(0).to_numpy().astype(np.int64)

    # Строгий временно́й сплит по квантилям времени.
    q_test = np.quantile(ts, 1.0 - test_fraction)
    q_val = np.quantile(ts, 1.0 - test_fraction - val_fraction)
    train_mask = ts < q_val
    val_mask = (ts >= q_val) & (ts < q_test)
    test_mask = ts >= q_test

    # Привязка паттернов — только для illicit-рёбер (их мало).
    pattern_map = parse_ibm_patterns(patterns_path)
    edge_pattern = np.array(["none"] * n_edges, dtype=object)
    illicit_idx = np.flatnonzero(label == 1)
    key_cols = df[IBM_KEY_COLS].to_numpy()
    n_matched = 0
    for i in illicit_idx:
        pat = pattern_map.get(_ibm_edge_key(key_cols[i]))
        if pat is not None:
            edge_pattern[i] = pat
            n_matched += 1
        else:
            edge_pattern[i] = "unknown"

    # Узловые признаки по TRAIN-рёбрам (антиутечка): degree + log-суммы.
    in_deg = np.bincount(dst[train_mask], minlength=num_nodes).astype(float)
    out_deg = np.bincount(src[train_mask], minlength=num_nodes).astype(float)
    in_sum = np.bincount(dst[train_mask], weights=amt_recv[train_mask], minlength=num_nodes)
    out_sum = np.bincount(src[train_mask], weights=amt_paid[train_mask], minlength=num_nodes)
    x = np.stack(
        [in_deg, out_deg, np.log1p(in_sum), np.log1p(out_sum), np.log1p(in_deg + out_deg)],
        axis=1,
    ).astype(np.float32)

    # Признаки рёбер. norm_time опционален (P1.6): под temporal split абсолютное
    # время может быть shortcut между train/val/test — include_time=False даёт
    # ablation «без времени».
    t_min, t_max = float(ts.min()), float(ts.max())
    norm_time = (ts - t_min) / max(t_max - t_min, 1.0)
    edge_feats = [np.log1p(amt_paid), np.log1p(amt_recv),
                  recv_cur.astype(float), pay_cur.astype(float), fmt.astype(float)]
    if include_time:
        edge_feats.append(norm_time)
    edge_attr = np.stack(edge_feats, axis=1).astype(np.float32)

    from torch_geometric.data import Data

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(np.stack([src, dst])),
        edge_attr=torch.from_numpy(edge_attr),
        edge_label=torch.from_numpy(label),
        edge_time=torch.from_numpy(ts.astype(np.int64)),
        num_nodes=num_nodes,
    )
    data.edge_pattern = edge_pattern  # numpy object array (строки паттернов)
    data.train_mask = torch.from_numpy(train_mask)
    data.val_mask = torch.from_numpy(val_mask)
    data.test_mask = torch.from_numpy(test_mask)

    def _split_illicit(mask):
        m = mask
        n = int(m.sum())
        npos = int(label[m].sum())
        return {"n": n, "n_pos": npos, "pos_rate": (npos / n) if n else 0.0}

    pat_vals, pat_counts = np.unique(edge_pattern[illicit_idx], return_counts=True) if len(illicit_idx) else ([], [])
    meta = {
        "variant": variant,
        "num_nodes": num_nodes,
        "num_edges": n_edges,
        "num_node_features": x.shape[1],
        "num_edge_features": edge_attr.shape[1],
        "include_time": include_time,
        "n_illicit": int(label.sum()),
        "illicit_rate": float(label.mean()),
        "patterns_matched": n_matched,
        "patterns_total_illicit": int(len(illicit_idx)),
        "pattern_counts": {str(k): int(v) for k, v in zip(pat_vals, pat_counts)},
        "encoders": {
            "receiving_currency": list(map(str, recv_cur_vocab)),
            "payment_currency": list(map(str, pay_cur_vocab)),
            "payment_format": list(map(str, fmt_vocab)),
        },
        "time_range_unix": [int(t_min), int(t_max)],
        "splits": {
            "train": _split_illicit(train_mask),
            "val": _split_illicit(val_mask),
            "test": _split_illicit(test_mask),
        },
    }
    return data, meta


def build_edge_features(data) -> np.ndarray:
    """Табличные признаки ребра для XGBoost-бейзлайна на IBM AML.

    Состав: edge_attr (6) + узловые фичи отправителя (5) + узловые фичи
    получателя (5) + log1p числа параллельных рёбер пары (1) = 17.
    Число параллельных рёбер считается ПО TRAIN-рёбрам (антиутечка), узловые
    фичи в data.x тоже посчитаны по train. Векторизовано через целочисленный
    ключ пары src*num_nodes+dst.
    """
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    x = data.x.numpy()
    ea = data.edge_attr.numpy()
    num_nodes = int(data.num_nodes)

    key_all = src.astype(np.int64) * num_nodes + dst.astype(np.int64)
    tm = data.train_mask.numpy()
    uniq, cnt = np.unique(key_all[tm], return_counts=True)
    idx = np.clip(np.searchsorted(uniq, key_all), 0, max(len(uniq) - 1, 0))
    parallel = np.where(uniq[idx] == key_all, cnt[idx], 0).astype(np.float32)

    return np.concatenate(
        [ea, x[src], x[dst], np.log1p(parallel).reshape(-1, 1)], axis=1
    ).astype(np.float32)


def load_node_time_steps(root: str = "data/elliptic") -> np.ndarray:
    """Временно́й шаг (1..49) каждого узла Elliptic из сырого features-CSV.

    Порядок строк features-CSV = порядок узлов в PyG-датасете (node i = строка i),
    поэтому колонка времени (индекс 1) напрямую выравнивается на индексы узлов.
    Нужно для строгого temporal-val (брать поздние шаги как val).
    """
    import os

    import pandas as pd

    path = os.path.join(root, "raw", "elliptic_txs_features.csv")
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 1].to_numpy().astype(int)


def make_temporal_val_split(time_steps, val_fraction: float = 0.2):
    """Val = узлы с самыми поздними временны́ми шагами (доля val_fraction).

    time_steps: массив временны́х шагов для подвыборки (обычно train-узлов).
    Возвращает (train_pos, val_pos) — позиции внутри переданной подвыборки.
    В отличие от случайного сплита, исключает утечку «из будущего в прошлое»
    при подборе порога: порог фиксируется на хронологически последних train-узлах.
    """
    time_steps = np.asarray(time_steps)
    order = np.argsort(time_steps, kind="stable")  # по возрастанию времени
    n = len(order)
    n_val = max(1, int(round(n * val_fraction)))
    return order[: n - n_val], order[n - n_val :]


if __name__ == "__main__":
    # Быстрая ручная проверка структуры датасета.
    warnings.filterwarnings("ignore")
    data, meta = load_elliptic()
    print(data)
    import json

    print(json.dumps(meta, indent=2, ensure_ascii=False))
