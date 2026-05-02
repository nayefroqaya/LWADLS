import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ⛔ Disable GPU completely

import warnings
import subprocess
import sys

import colorama
import pandas as pd
import torch

from logdata_read import LogdataRead
from utils import utils
import psutil

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
    Round = '1'
    mode = 'M'  # M multi classifier - S single classifier
    Mix_or_stable = '0'  # 0 Full stable subset  / 1 mix subset

    ALL_DATASET_LOG_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.LOG'
    ALL_DATASET_CSV_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv'

    CONFIG_PATH = "configs/experiment.yaml"

    # ---------------- Initialize classes ----------------
    logdata_read_obj = LogdataRead()
    utilities_obj = utils()

    # ---------------- Data as CSV ----------------
    logdata_read_obj.read_original_data_log_from_log_to_csv(DATASET, ALL_DATASET_CSV_PATH)

    print(f"{GREEN}Reading the file was done successfully{RESET}")

    # ---------------- Dataset Splitting ----------------
    print(f"{GRAY}Splitting dataset into training, validation, and test sets...{RESET}")
    train_df, validate_df, test_df, df_features = utilities_obj.dataset_splitting(ALL_DATASET_CSV_PATH, DATASET, Round,
                                                                                  Mix_or_stable)
    # exit()
    # ---------------- Process normal data ----------------

    if Mix_or_stable == '0' and DATASET == 'S_BGL':  # Stable
        # Create folder to save splits
        print(f"{GRAY}Processing normal data portion in the dataset...{RESET}")
        save_path = os.path.join(f"../datasets/{DATASET}", f"{Round}_{DATASET}_'Stable'_Splitted_Datasets")
    elif Mix_or_stable == '1' and DATASET == 'S_BGL':  # Mix
        # Create folder to save splits
        print(f"{GRAY}Processing normal data portion in the dataset...{RESET}")
        save_path = os.path.join(f"../datasets/{DATASET}", f"{Round}_{DATASET}_'Mix'_Splitted_Datasets")
    else:

        print(f"{GRAY}Processing normal data portion in the dataset...{RESET}")
        save_path = os.path.join(f"../datasets/{DATASET}", f"{Round}_{DATASET}_Splitted_Datasets")

    train_df = pd.read_pickle(os.path.join(save_path, "train_df.pkl"))
    val_df = pd.read_pickle(os.path.join(save_path, "val_df.pkl"))
    test_df = pd.read_pickle(os.path.join(save_path, "test_df.pkl"))

    final_train_with_test_with_val = utilities_obj.processing_data_portion(train_df, val_df, test_df)

    exit()

    # ---------------- Run experiments ----------------

    # 1. Optional MLM
    run_command([sys.executable, "run_mlm.py", "--config", CONFIG_PATH])

    # 2. Teacher
    run_command([sys.executable, "run_experiment.py", "--config", CONFIG_PATH, "--mode", "teacher"])

    # 3. Student baseline
    run_command([sys.executable, "run_experiment.py", "--config", CONFIG_PATH, "--mode", "student"])

    # 4. Distilled student
    run_command([sys.executable, "run_experiment.py", "--config", CONFIG_PATH, "--mode", "distill"])

    print(f"\n{GREEN}All steps completed successfully.{RESET}")


if __name__ == "__main__":
    main()
