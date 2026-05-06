"""
PyCharm entry point for AdaLogSLM.

You can run this file directly from PyCharm.

Default:
    python main.py

Optional:
    python main.py --config configs/adalogslm_unified_config.yml

To change mode, edit:
    configs/adalogslm_unified_config.yml

Supported modes:
    experiment.mode: "in_domain"
    experiment.mode: "fewshot_target_adaptation"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Make project root importable when running directly from PyCharm.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import load_config
from src.run import run_in_domain, run_fewshot_target_adaptation


def parse_args():
    parser = argparse.ArgumentParser(description="Run AdaLogSLM from PyCharm main.py")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "adalogslm_unified_config.yml"),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(str(config_path))
    mode = cfg["experiment"]["mode"]

    print("=" * 80)
    print("AdaLogSLM")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Config file  : {config_path}")
    print(f"Run mode     : {mode}")
    print("=" * 80)

    if mode == "in_domain":
        run_in_domain(cfg, str(config_path))
    elif mode == "fewshot_target_adaptation":
        run_fewshot_target_adaptation(cfg, str(config_path))
    else:
        raise ValueError(
            f"Unsupported mode: {mode}. "
            "Use 'in_domain' or 'fewshot_target_adaptation'."
        )


if __name__ == "__main__":
    main()
