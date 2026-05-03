import os

#os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ⛔ Disable GPU completely

import warnings
import subprocess
import sys

import colorama
import pandas as pd
import torch

from load_datalog import LogdataRead
from utils import utils
import psutil

import os
import warnings
import subprocess
import sys
import yaml

import colorama
import pandas as pd
import torch

from load_datalog import LogdataRead
from utils import utils

# ====================== Setup ======================
warnings.filterwarnings("ignore")
colorama.init()

GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW
RED = colorama.Fore.RED


def run_command(command):
    print(f"\n{GREEN}Running: {' '.join(command)}{RESET}")
    subprocess.run(command, check=True)


def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_name(items):
    return "_".join([str(x).replace(" ", "") for x in items])


def make_experiment_name(mode, train_datasets, val_datasets, test_datasets):
    train_name = clean_name(train_datasets)
    val_name = clean_name(val_datasets)
    test_name = clean_name(test_datasets)

    return f"{mode}__train-{train_name}__val-{val_name}__test-{test_name}"


def read_csv_if_exists(path):
    if not os.path.exists(path):
        print(f"{RED}File not found: {path}{RESET}")
        return None

    return pd.read_csv(path)


def print_model_summary(model_name, metrics_df, report_df, per_dataset_df):
    print(f"\n{GREEN}{model_name}{RESET}")
    print("=" * 80)

    if metrics_df is not None:
        row = metrics_df.iloc[0]

        print("\nOverall metrics:")
        print("-" * 50)
        print(f"Precision: {float(row['precision']):.4f}")
        print(f"Recall:    {float(row['recall']):.4f}")
        print(f"F1-score:  {float(row['f1']):.4f}")
        print(f"Accuracy:  {float(row['accuracy']):.4f}")

    if report_df is not None:
        print("\nClassification report:")
        print("-" * 50)
        print(report_df)

    if per_dataset_df is not None:
        print("\nPer-dataset metrics:")
        print("-" * 50)
        print(per_dataset_df)


def run_prediction_for_saved_model(
    config_path,
    output_dir,
    model_mode,
    train_datasets,
    val_datasets,
    test_datasets,
    split="test",
):
    exp_name = make_experiment_name(
        model_mode,
        train_datasets,
        val_datasets,
        test_datasets,
    )

    model_path = os.path.join(output_dir, exp_name, "best_model")

    if not os.path.exists(model_path):
        print(f"{RED}Saved model not found: {model_path}{RESET}")
        return None, None, None

    run_command(
        [
            sys.executable,
            "predict_saved_model.py",
            "--config",
            config_path,
            "--model_path",
            model_path,
            "--split",
            split,
            "--datasets",
            *test_datasets,
        ]
    )

    prediction_folder = os.path.join(
        output_dir,
        "predictions",
        f"{exp_name}__best_model__split-{split}__datasets-{clean_name(test_datasets)}",
    )

    metrics_path = os.path.join(prediction_folder, "metrics.csv")
    report_path = os.path.join(prediction_folder, "classification_report.csv")
    per_dataset_metrics_path = os.path.join(prediction_folder, "per_dataset_metrics.csv")

    metrics_df = read_csv_if_exists(metrics_path)
    report_df = read_csv_if_exists(report_path)
    per_dataset_df = read_csv_if_exists(per_dataset_metrics_path)

    return metrics_df, report_df, per_dataset_df



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
    DATASET = 'HDFS'
    DATASETS_FOLDER = 'datasets'
    Round = '1'
    mode = 'M'  # M multi classifier - S single classifier
    Mix_or_stable = '0'  # 0 Full stable subset  / 1 mix subset

    ALL_DATASET_LOG_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.LOG'
    ALL_DATASET_CSV_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv'

    CONFIG_PATH =  "../config/experiment.yaml"
    '''

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
    '''



    # ---------------- Training stages ----------------
    CONFIG_PATH = "../config/experiment.yaml"
    OUTPUT_DIR = "./outputs"

    # IMPORTANT:
    # These names must match your experiment.yaml exactly.
    TRAIN_DATASETS = [ "BGL"]
    VAL_DATASETS = [ "BGL"]
    TEST_DATASETS = ["HDFS"]

    RUN_TRAINING = True
    RUN_PREDICTION = True
    RUN_MLM = True

    if RUN_TRAINING:
        if RUN_MLM:
            print("# 1. Optional MLM")
            run_command(
                [
                    sys.executable,
                    "run_mlm.py",
                    "--config",
                    CONFIG_PATH,
                ]
            )

        print("# 2. Teacher")
        run_command(
            [
                sys.executable,
                "run_experiment.py",
                "--config",
                CONFIG_PATH,
                "--mode",
                "teacher",
            ]
        )

        print("# 3. Student baseline")
        run_command(
            [
                sys.executable,
                "run_experiment.py",
                "--config",
                CONFIG_PATH,
                "--mode",
                "student",
            ]
        )

        print("# 4. Distilled student")
        run_command(
            [
                sys.executable,
                "run_experiment.py",
                "--config",
                CONFIG_PATH,
                "--mode",
                "distill",
            ]
        )

    # ---------------- Prediction only ----------------
    if RUN_PREDICTION:
        print(f"\n{GREEN}Running prediction using saved models...{RESET}")

        all_metrics = {}

        teacher_metrics = run_prediction_for_saved_model(
            config_path=CONFIG_PATH,
            output_dir=OUTPUT_DIR,
            model_mode="teacher",
            train_datasets=TRAIN_DATASETS,
            val_datasets=VAL_DATASETS,
            test_datasets=TEST_DATASETS,
            split="test",
        )
        all_metrics["teacher"] = teacher_metrics

        student_metrics = run_prediction_for_saved_model(
            config_path=CONFIG_PATH,
            output_dir=OUTPUT_DIR,
            model_mode="student_no_distill",
            train_datasets=TRAIN_DATASETS,
            val_datasets=VAL_DATASETS,
            test_datasets=TEST_DATASETS,
            split="test",
        )
        all_metrics["student_no_distill"] = student_metrics

        distilled_metrics = run_prediction_for_saved_model(
            config_path=CONFIG_PATH,
            output_dir=OUTPUT_DIR,
            model_mode="student_distilled",
            train_datasets=TRAIN_DATASETS,
            val_datasets=VAL_DATASETS,
            test_datasets=TEST_DATASETS,
            split="test",
        )
        all_metrics["student_distilled"] = distilled_metrics

        # ---------------- Final comparison ----------------
        print(f"\n{GREEN}Final prediction metrics comparison{RESET}")
        print("=" * 70)

        summary_rows = []

        for model_name, metrics in all_metrics.items():
            print_model_metrics(model_name, metrics)

            if metrics is not None:
                summary_rows.append(
                    {
                        "model": model_name,
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1": metrics.get("f1"),
                        "accuracy": metrics.get("accuracy"),
                    }
                )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)

            summary_path = os.path.join(
                OUTPUT_DIR,
                f"final_prediction_summary__test-{clean_name(TEST_DATASETS)}.csv",
            )

            summary_df.to_csv(summary_path, index=False)

            print(f"\n{GREEN}Saved final comparison to:{RESET}")
            print(summary_path)

            print(f"\n{GREEN}Summary table:{RESET}")
            print(summary_df)

    print(f"\n{GREEN}All steps completed successfully.{RESET}")



if __name__ == "__main__":
    main()
