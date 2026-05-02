import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ⛔ Disable GPU completely

import warnings
import subprocess
import sys

import colorama
import pandas as pd
import torch

from anomaly_detection import AnomalyDetector
from features_engineering import FeaturesEngineering
from features_extracting import FeaturesExtractor
from logdata_read import LogdataRead
from model_evaluation import ModelEvaluation
from utility import Utilities
import psutil
import platform


# ====================== Setup ======================
warnings.filterwarnings('ignore')
colorama.init()

GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


def run_command(command):
    print(f"\n{GREEN}Running: {' '.join(command)}{RESET}")
    subprocess.run(command, check=True)


# ====================== Main ======================
def main():

    # ---------------- Device check ------------------
    if torch.cuda.is_available():
        print(f"{GREEN}GPU detected. Using GPU for encoding.{RESET}")
    else:
        print(f"{YELLOW}No GPU detected. Using CPU for encoding.{RESET}")

    # ---------------- Display options ----------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # ---------------- Project configuration ----------------
    DATASET = 'BGL'
    DATASETS_FOLDER = 'datasets'

    ALL_DATASET_LOG_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.LOG'
    ALL_DATASET_CSV_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv'

    CONFIG_PATH = "configs/experiment.yaml"

    # ---------------- Initialize classes ----------------
    logdata_read_obj = LogdataRead()

    # ---------------- Data as CSV ----------------
    logdata_read_obj.read_original_data_log_from_log_to_csv(
        DATASET,
        ALL_DATASET_CSV_PATH
    )

    print(f"{GREEN}Reading the file was done successfully{RESET}")

    # ---------------- Run experiments ----------------

    # 1. Optional MLM
    run_command([
        sys.executable,
        "run_mlm.py",
        "--config",
        CONFIG_PATH
    ])

    # 2. Teacher
    run_command([
        sys.executable,
        "run_experiment.py",
        "--config",
        CONFIG_PATH,
        "--mode",
        "teacher"
    ])

    # 3. Student baseline
    run_command([
        sys.executable,
        "run_experiment.py",
        "--config",
        CONFIG_PATH,
        "--mode",
        "student"
    ])

    # 4. Distilled student
    run_command([
        sys.executable,
        "run_experiment.py",
        "--config",
        CONFIG_PATH,
        "--mode",
        "distill"
    ])

    print(f"\n{GREEN}All steps completed successfully.{RESET}")


if __name__ == "__main__":
    main()
