"""Утилиты проекта: воспроизводимость, конфиги, устройство, сохранение JSON."""
from __future__ import annotations

import json
import os
import random
from typing import Any, Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Зафиксировать сиды random / numpy / torch (+ cuda) для воспроизводимости.

    Замечание: часть CUDA/MPS-операций и XGBoost на GPU недетерминированы,
    поэтому небольшие расхождения метрик между запусками допустимы.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str) -> dict:
    """Прочитать YAML-конфиг и вернуть как dict."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device():
    """Вернуть лучшее доступное устройство: cuda → mps → cpu."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_wandb(config: dict, run_name: Optional[str] = None):
    """Единая точка инициализации W&B с мягким fallback.

    Включается, если в конфиге `wandb.enabled: true` ИЛИ задана переменная
    окружения WANDB_MODE (online|offline). При отсутствии пакета/ключа или любой
    ошибке — возвращает None и не роняет обучение.

    Возвращает объект run (модуль wandb) либо None.
    """
    wb_cfg = (config or {}).get("wandb", {}) or {}
    env_mode = os.environ.get("WANDB_MODE")
    enabled = bool(wb_cfg.get("enabled", False)) or (env_mode in ("online", "offline"))
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] пакет не установлен — логирование пропущено (pip install wandb)")
        return None
    try:
        wandb.init(
            project=wb_cfg.get("project", "gnn-aml"),
            entity=wb_cfg.get("entity"),
            name=run_name,
            config=config,
            mode=env_mode,  # None → online по умолчанию
        )
        return wandb
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] init не удался: {e} — логирование пропущено")
        return None


def save_json(obj: dict, path: str) -> None:
    """Сохранить словарь в JSON (создаёт директорию при необходимости)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2, ensure_ascii=False)


def _to_jsonable(obj: Any) -> Any:
    """Рекурсивно привести numpy-типы к нативным python-типам для json.dump."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
