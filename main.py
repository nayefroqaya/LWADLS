"""
PyCharm entry point for SLMADLS / AdaLogSLM.

Run this file directly from PyCharm or terminal.

Default:
    python main.py

Optional:
    python main.py --config configs/adalogslm_unified_config.yml

YAML controls the mode and stage:

experiment:
  mode: "in_domain"
  stage: "train"

Supported modes:
    in_domain
    fewshot_target_adaptation

Supported stages for BOTH modes:
    train
        Train and save model, tokenizer, and normal center/prototypes.
        Does NOT predict.

    predict
        Load saved model, tokenizer, and center/prototypes.
        Predict test data.
        Save reports and predictions.
        Does NOT retrain.

    posthoc_only
        Load saved predictions.csv.
        Run alpha/beta/threshold grid search.
        Does NOT train or predict.

    train_predict
        Train + predict + posthoc calibration in one run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ======================================================
# PROJECT SETUP
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ======================================================
# IMPORT PIPELINE FUNCTIONS
# ======================================================

from src.config import load_config
from src.run import (
    run_in_domain,
    train_in_domain,
    predict_in_domain,
    run_fewshot_target_adaptation,
    train_fewshot_target_adaptation,
    predict_fewshot_target_adaptation,
    run_posthoc_only,
)


# ======================================================
# ARGUMENTS
# ======================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SLMADLS / AdaLogSLM from main.py"
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "adalogslm_unified_config.yml"),
        help="Path to YAML config file.",
    )

    return parser.parse_args()


# ======================================================
# RUNTIME HELPERS
# ======================================================

def format_minutes(seconds: float) -> str:
    return f"{seconds / 60.0:.2f}"


def print_time_summary(runtime_summary: dict):
    """
    Print runtime summary at the very end so it is visible
    in server logs without scrolling back.
    """

    print("\n" + "=" * 80)
    print("[FINAL RUNTIME SUMMARY]")
    print("=" * 80)

    if not runtime_summary:
        print("No runtime information recorded.")
    else:
        for name, seconds in runtime_summary.items():
            print(
                f"{name:<20}: "
                f"{seconds:.2f} seconds | "
                f"{format_minutes(seconds)} minutes"
            )

    print("=" * 80 + "\n")


# ======================================================
# MAIN ROUTER
# ======================================================

def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    cfg = load_config(str(config_path))

    mode = cfg["experiment"]["mode"]
    stage = cfg["experiment"].get("stage", "train_predict")

    runtime_summary = {}

    print("=" * 80)
    print("SLMADLS / AdaLogSLM")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Config file  : {config_path}")
    print(f"Run mode     : {mode}")
    print(f"Run stage    : {stage}")
    print("=" * 80)

    # --------------------------------------------------
    # POSTHOC ONLY
    # --------------------------------------------------
    # This stage is shared by both modes.
    # It only loads predictions.csv and runs grid search.
    if stage == "posthoc_only":
        start_time = time.perf_counter()

        run_posthoc_only(cfg)

        runtime_summary["posthoc_only"] = time.perf_counter() - start_time
        print_time_summary(runtime_summary)
        return

    # --------------------------------------------------
    # MODE 1: IN-DOMAIN
    # --------------------------------------------------
    if mode == "in_domain":

        if stage == "train":
            start_time = time.perf_counter()

            train_in_domain(
                cfg,
                str(config_path),
            )

            runtime_summary["train"] = time.perf_counter() - start_time
            print_time_summary(runtime_summary)

        elif stage == "predict":
            start_time = time.perf_counter()

            predict_in_domain(
                cfg,
                str(config_path),
            )

            runtime_summary["predict"] = time.perf_counter() - start_time
            print_time_summary(runtime_summary)

        elif stage == "train_predict":
            total_start_time = time.perf_counter()

            train_start_time = time.perf_counter()
            train_in_domain(
                cfg,
                str(config_path),
            )
            runtime_summary["train"] = time.perf_counter() - train_start_time

            predict_start_time = time.perf_counter()
            predict_in_domain(
                cfg,
                str(config_path),
            )
            runtime_summary["predict"] = time.perf_counter() - predict_start_time

            runtime_summary["total"] = time.perf_counter() - total_start_time
            print_time_summary(runtime_summary)

        else:
            raise ValueError(
                f"Unsupported stage for in_domain: {stage}\n"
                "Supported stages are:\n"
                "  train\n"
                "  predict\n"
                "  posthoc_only\n"
                "  train_predict"
            )

    # --------------------------------------------------
    # MODE 2: FEW-SHOT TARGET ADAPTATION
    # --------------------------------------------------
    elif mode == "fewshot_target_adaptation":

        if stage == "train":
            start_time = time.perf_counter()

            train_fewshot_target_adaptation(
                cfg,
                str(config_path),
            )

            runtime_summary["train"] = time.perf_counter() - start_time
            print_time_summary(runtime_summary)

        elif stage == "predict":
            start_time = time.perf_counter()

            predict_fewshot_target_adaptation(
                cfg,
                str(config_path),
            )

            runtime_summary["predict"] = time.perf_counter() - start_time
            print_time_summary(runtime_summary)

        elif stage == "train_predict":
            total_start_time = time.perf_counter()

            train_start_time = time.perf_counter()
            train_fewshot_target_adaptation(
                cfg,
                str(config_path),
            )
            runtime_summary["train"] = time.perf_counter() - train_start_time

            predict_start_time = time.perf_counter()
            predict_fewshot_target_adaptation(
                cfg,
                str(config_path),
            )
            runtime_summary["predict"] = time.perf_counter() - predict_start_time

            runtime_summary["total"] = time.perf_counter() - total_start_time
            print_time_summary(runtime_summary)

        else:
            raise ValueError(
                f"Unsupported stage for fewshot_target_adaptation: {stage}\n"
                "Supported stages are:\n"
                "  train\n"
                "  predict\n"
                "  posthoc_only\n"
                "  train_predict"
            )

    else:
        raise ValueError(
            f"Unsupported mode: {mode}\n"
            "Supported modes are:\n"
            "  in_domain\n"
            "  fewshot_target_adaptation"
        )


if __name__ == "__main__":
    main()