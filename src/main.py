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


def main():
    # ---------------- Device check ------------------
    if torch.cuda.is_available():
        print(f"{GREEN}GPU detected. Using GPU.{RESET}")
    else:
        print(f"{YELLOW}No GPU detected. Using CPU.{RESET}")

    # ---------------- Display options ----------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # ---------------- Config ----------------
    CONFIG_PATH = "../config/experiment.yaml"
    config = load_yaml_config(CONFIG_PATH)

    OUTPUT_DIR = config["output_dir"]
    TRAIN_DATASETS = config["train_datasets"]
    VAL_DATASETS = config["val_datasets"]
    TEST_DATASETS = config["test_datasets"]

    # ---------------- Control switches ----------------
    # For prediction only after models are already trained:
    RUN_MLM = False
    RUN_TRAINING = False
    RUN_PREDICTION = True

    # If you want to train everything again, set:
    # RUN_MLM = True
    # RUN_TRAINING = True
    # RUN_PREDICTION = True

    # ---------------- Initialize classes ----------------
    logdata_read_obj = LogdataRead()
    utilities_obj = utils()

    print(f"\n{GREEN}Experiment datasets from YAML{RESET}")
    print("-" * 60)
    print(f"Train datasets: {TRAIN_DATASETS}")
    print(f"Val datasets:   {VAL_DATASETS}")
    print(f"Test datasets:  {TEST_DATASETS}")
    '''
    # ---------------- Training stages ----------------
    if RUN_TRAINING:
        if RUN_MLM:
            print("\n# 1. Optional MLM")
            run_command(
                [
                    sys.executable,
                    "run_mlm.py",
                    "--config",
                    CONFIG_PATH,
                ]
            )

        print("\n# 2. Teacher")
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

        print("\n# 3. Student baseline")
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

        print("\n# 4. Distilled student")
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
    '''
    # ---------------- Prediction stages ----------------
    if RUN_PREDICTION:
        print(f"\n{GREEN}Running prediction using saved models...{RESET}")

        model_modes = [
            "teacher",
            "student_no_distill",
            "student_distilled",
        ]

        summary_rows = []

        for model_mode in model_modes:
            metrics_df, report_df, per_dataset_df = run_prediction_for_saved_model(
                config_path=CONFIG_PATH,
                output_dir=OUTPUT_DIR,
                model_mode=model_mode,
                train_datasets=TRAIN_DATASETS,
                val_datasets=VAL_DATASETS,
                test_datasets=TEST_DATASETS,
                split="test",
            )

            print_model_summary(
                model_name=model_mode,
                metrics_df=metrics_df,
                report_df=report_df,
                per_dataset_df=per_dataset_df,
            )

            if metrics_df is not None:
                row = metrics_df.iloc[0]

                summary_rows.append(
                    {
                        "model": model_mode,
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1"],
                        "accuracy": row["accuracy"],
                    }
                )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)

            summary_path = os.path.join(
                OUTPUT_DIR,
                f"final_prediction_summary__test-{clean_name(TEST_DATASETS)}.csv",
            )

            summary_df.to_csv(summary_path, index=False)

            print(f"\n{GREEN}Final model comparison{RESET}")
            print("=" * 80)
            print(summary_df)

            print(f"\nSaved final summary to:")
            print(summary_path)

    print(f"\n{GREEN}All steps completed successfully.{RESET}")


if __name__ == "__main__":
    main()