import shutil
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dir(cfg: Dict[str, Any]) -> Path:
    out = Path(cfg["experiment"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_resolved_config(config_path: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yml")


def dataset_split_path(cfg: Dict[str, Any], dataset: str, split: str) -> Path:
    data_dir = Path(cfg["paths"]["data_dir"])
    template = cfg["paths"]["dataset_path_template"]
    split_files = cfg["split_files"]

    if split not in split_files:
        raise ValueError(f"Unknown split '{split}'. Available: {list(split_files.keys())}")

    rel_dir = template.format(dataset=dataset)
    return data_dir / rel_dir / split_files[split]
