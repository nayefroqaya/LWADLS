import json
import random
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def normalize_log_text(text: str, text_cfg: Dict[str, Any]) -> str:
    text = "" if text is None else str(text)

    if text_cfg.get("lowercase", False):
        text = text.lower()

    if text_cfg.get("mask_ips", True):
        text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>", text)

    if text_cfg.get("mask_paths", True):
        text = re.sub(r"(/[^\s]+)+", "<PATH>", text)
        text = re.sub(r"[A-Za-z]:\\[^\s]+", "<PATH>", text)

    if text_cfg.get("mask_numbers", True):
        text = re.sub(r"\b\d+\b", "<NUM>", text)

    return " ".join(text.split())
