from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import colorama
import pandas as pd
import psutil
import torch
#from anomaly_detection import AnomalyDetector
#from features_engineering import FeaturesEngineering
#from features_extracting import FeaturesExtractor
#from logdata_read import LogdataRead
#from model_evaluation import ModelEvaluation

from utility import Utilities

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
#from src.unsupervised_run import (train_in_domain, predict_in_domain, train_fewshot_target_adaptation,
#                                  predict_fewshot_target_adaptation, run_posthoc_only, )
from src.Supervised_run import (train_in_domain, predict_in_domain, train_fewshot_target_adaptation,
                                  predict_fewshot_target_adaptation, run_posthoc_only, )

# ======================================================
# ARGUMENTS
# ======================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run SLMADLS / AdaLogSLM from main.py")

    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "adalogslm_unified_config.yml"),
                        help="Path to YAML config file.", )

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
            print(f"{name:<20}: "
                  f"{seconds:.2f} seconds | "
                  f"{format_minutes(seconds)} minutes")

    print("=" * 80 + "\n")


# ======================================================
# MAIN ROUTER
# ======================================================

def main():


    # ---------------- Section A  : run config for data spliting (60%-10%-30%):----------------
    '''
    warnings.filterwarnings('ignore')
    colorama.init()

    GREEN = colorama.Fore.GREEN
    GRAY = colorama.Fore.LIGHTBLACK_EX
    RESET = colorama.Fore.RESET
    YELLOW = colorama.Fore.YELLOW

    DATASET = 'BGL'
    DATASETS_FOLDER = 'datasets'
    Round = '1'  # 1 dataset for first run. 2 for second run. 3 for third run.
    Mix_or_stable = '0'  # 0 Full stable subset  / 1 mix subset

    # path to save files
    # ALL_DATASET_CSV_PATH = f'../../LWADLS/{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv' # data second paper
    ALL_DATASET_CSV_PATH = f'../../LogSLM/{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv'  # data second paper

    # Object classes :
    logdata_read_obj = LogdataRead()
    utilities_obj = Utilities()

    # ---------------- Data as CSV ----------------
    logdata_read_obj.read_original_data_log_from_log_to_csv(DATASET, ALL_DATASET_CSV_PATH)
    print(f"{GREEN}Reading the file was done successfully{RESET}")

    # ---------------- Dataset Splitting ----------------
    print(f"{GRAY}Splitting dataset into training, validation, and test sets...{RESET}")
    train_df, validate_df, test_df, df_features = utilities_obj.dataset_splitting(ALL_DATASET_CSV_PATH, DATASET, Round,
                                                                                  Mix_or_stable)
    exit()
    '''

    # End of the section A  --------------------------------------------------------------------------------------------

    # ---------------- Section B ----- AD pipeline ------------------------------------------------------------------- :
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

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

            train_in_domain(cfg, str(config_path), )

            runtime_summary["train"] = time.perf_counter() - start_time
            print_time_summary(runtime_summary)

        elif stage == "predict":
            start_time = time.perf_counter()

            predict_in_domain(cfg, str(config_path), )

            runtime_summary["predict"] = time.perf_counter() - start_time
            print_time_summary(runtime_summary)

        elif stage == "train_predict":
            total_start_time = time.perf_counter()

            train_start_time = time.perf_counter()
            train_in_domain(cfg, str(config_path), )
            runtime_summary["train"] = time.perf_counter() - train_start_time

            predict_start_time = time.perf_counter()
            predict_in_domain(cfg, str(config_path), )
            runtime_summary["predict"] = time.perf_counter() - predict_start_time

            runtime_summary["total"] = time.perf_counter() - total_start_time
            print_time_summary(runtime_summary)

        else:
            raise ValueError(f"Unsupported stage for in_domain: {stage}\n"
                             "Supported stages are:\n"
                             "  train\n"
                             "  predict\n"
                             "  posthoc_only\n"
                             "  train_predict")

    # --------------------------------------------------
    # MODE 2: FEW-SHOT TARGET ADAPTATION
    # --------------------------------------------------
    elif mode == "fewshot_target_adaptation":

        if stage == "train":
            start_time = time.perf_counter()

            train_fewshot_target_adaptation(cfg, str(config_path), )

            runtime_summary["train"] = time.perf_counter() - start_time
            print_time_summary(runtime_summary)

        elif stage == "predict":
            start_time = time.perf_counter()

            predict_fewshot_target_adaptation(cfg, str(config_path), )

            runtime_summary["predict"] = time.perf_counter() - start_time
            print_time_summary(runtime_summary)

        elif stage == "train_predict":
            total_start_time = time.perf_counter()

            train_start_time = time.perf_counter()
            train_fewshot_target_adaptation(cfg, str(config_path), )
            runtime_summary["train"] = time.perf_counter() - train_start_time

            predict_start_time = time.perf_counter()
            predict_fewshot_target_adaptation(cfg, str(config_path), )
            runtime_summary["predict"] = time.perf_counter() - predict_start_time

            runtime_summary["total"] = time.perf_counter() - total_start_time
            print_time_summary(runtime_summary)

        else:
            raise ValueError(f"Unsupported stage for fewshot_target_adaptation: {stage}\n"
                             "Supported stages are:\n"
                             "  train\n"
                             "  predict\n"
                             "  posthoc_only\n"
                             "  train_predict")

    else:
        raise ValueError(f"Unsupported mode: {mode}\n"
                         "Supported modes are:\n"
                         "  in_domain\n"
                         "  fewshot_target_adaptation")


if __name__ == "__main__":
    main()