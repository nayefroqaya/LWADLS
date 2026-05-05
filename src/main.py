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


warnings.filterwarnings("ignore")
colorama.init()

GREEN = colorama.Fore.GREEN
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


def make_experiment_name(train_datasets, val_datasets, test_datasets):
    train_name = clean_name(train_datasets)
    val_name = clean_name(val_datasets)
    test_name = clean_name(test_datasets)

    return f"student__train-{train_name}__val-{val_name}__test-{test_name}"


def read_csv_if_exists(path):
    if not os.path.exists(path):
        print(f"{RED}File not found: {path}{RESET}")
        return None

    return pd.read_csv(path)


def print_model_summary(metrics_df, report_df, per_dataset_df):
    print(f"\n{GREEN}student{RESET}")
    print("=" * 80)

    if metrics_df is not None:
        row = metrics_df.iloc[0]

        print("\nOverall metrics:")
        print("-" * 50)
        print(f"Precision: {float(row['precision']):.4f}")
        print(f"Recall:    {float(row['recall']):.4f}")
        print(f"F1-score:  {float(row['f1']):.4f}")
        print(f"Accuracy:  {float(row['accuracy']):.4f}")

        if "anomaly_threshold" in row:
            print(f"Threshold: {float(row['anomaly_threshold']):.4f}")

    if report_df is not None:
        print("\nClassification report:")
        print("-" * 50)
        print(report_df)

    if per_dataset_df is not None:
        print("\nPer-dataset metrics:")
        print("-" * 50)
        print(per_dataset_df)


def run_prediction_for_student(
    config_path,
    output_dir,
    train_datasets,
    val_datasets,
    internal_test_datasets,
    predict_split,
    predict_datasets,
):
    config = load_yaml_config(config_path)

    exp_name = make_experiment_name(
        train_datasets,
        val_datasets,
        internal_test_datasets,
    )

    model_path = os.path.join(output_dir, exp_name, "best_model")

    if not os.path.exists(model_path):
        print(f"{RED}Saved model not found: {model_path}{RESET}")
        return None, None, None

    threshold = None

    if config.get("threshold_tuning", {}).get("enabled", False):
        print(f"\n{GREEN}Tuning threshold for student using source validation...{RESET}")

        run_command(
            [
                sys.executable,
                "tune_threshold.py",
                "--config",
                config_path,
                "--model_path",
                model_path,
            ]
        )

        tuning_datasets = config["threshold_tuning"].get(
            "datasets",
            val_datasets,
        )

        threshold_file = os.path.join(
            output_dir,
            "threshold_tuning",
            f"{exp_name}__val-{clean_name(tuning_datasets)}",
            "best_threshold.csv",
        )

        best_df = pd.read_csv(threshold_file)
        threshold = float(best_df.iloc[0]["threshold"])

        print(f"{GREEN}Using tuned threshold for student: {threshold}{RESET}")

    command = [
        sys.executable,
        "predict_saved_model.py",
        "--config",
        config_path,
        "--model_path",
        model_path,
        "--split",
        predict_split,
        "--datasets",
        *predict_datasets,
    ]

    if threshold is not None:
        command.extend(["--threshold", str(threshold)])

    run_command(command)

    if threshold is None:
        threshold = float(
            config.get("prediction_stage", {}).get("anomaly_threshold", 0.5)
        )

    threshold_name = str(round(float(threshold), 6)).replace(".", "p")

    prediction_folder = os.path.join(
        output_dir,
        "predictions",
        f"{exp_name}__best_model__split-{predict_split}"
        f"__datasets-{clean_name(predict_datasets)}__thr-{threshold_name}",
    )

    metrics_path = os.path.join(prediction_folder, "metrics.csv")
    report_path = os.path.join(prediction_folder, "classification_report.csv")
    per_dataset_metrics_path = os.path.join(
        prediction_folder,
        "per_dataset_metrics.csv",
    )

    metrics_df = read_csv_if_exists(metrics_path)
    report_df = read_csv_if_exists(report_path)
    per_dataset_df = read_csv_if_exists(per_dataset_metrics_path)

    return metrics_df, report_df, per_dataset_df


def save_final_student_report(
    output_dir,
    train_datasets,
    val_datasets,
    internal_test_datasets,
    predict_datasets,
    metrics_df,
):
    if metrics_df is None:
        return

    row = metrics_df.iloc[0]

    summary_row = {
        "model": "student",
        "train_datasets": clean_name(train_datasets),
        "val_datasets": clean_name(val_datasets),
        "internal_test_datasets": clean_name(internal_test_datasets),
        "prediction_datasets": clean_name(predict_datasets),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "accuracy": float(row["accuracy"]),
        "anomaly_threshold": float(row.get("anomaly_threshold", 0.5)),
    }

    summary_df = pd.DataFrame([summary_row])

    summary_csv_path = os.path.join(
        output_dir,
        f"final_student_summary__train-{clean_name(train_datasets)}"
        f"__predict-{clean_name(predict_datasets)}.csv",
    )

    summary_txt_path = os.path.join(
        output_dir,
        f"final_student_report__train-{clean_name(train_datasets)}"
        f"__predict-{clean_name(predict_datasets)}.txt",
    )

    summary_df.to_csv(summary_csv_path, index=False)

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("STUDENT-ONLY LOG ANOMALY DETECTION REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write("Experiment Setup\n")
        f.write("-" * 60 + "\n")
        f.write(f"Train datasets:         {train_datasets}\n")
        f.write(f"Validation datasets:    {val_datasets}\n")
        f.write(f"Internal test datasets: {internal_test_datasets}\n")
        f.write(f"Prediction datasets:    {predict_datasets}\n\n")

        f.write("Metrics for Class 1 = Anomaly\n")
        f.write("-" * 60 + "\n")
        f.write(f"Precision: {summary_row['precision']:.6f}\n")
        f.write(f"Recall:    {summary_row['recall']:.6f}\n")
        f.write(f"F1-score:  {summary_row['f1']:.6f}\n")
        f.write(f"Accuracy:  {summary_row['accuracy']:.6f}\n")
        f.write(f"Threshold: {summary_row['anomaly_threshold']:.6f}\n\n")

        f.write("Notes\n")
        f.write("-" * 60 + "\n")
        f.write(
            "Student-only setup uses MiniLM with DAPT/TAPT, balanced fine-tuning, "
            "source-validation threshold tuning, and optional hybrid scoring.\n"
        )

    print(f"\n{GREEN}Saved final CSV summary:{RESET}")
    print(summary_csv_path)

    print(f"\n{GREEN}Saved final TXT report:{RESET}")
    print(summary_txt_path)

    print(f"\n{GREEN}Final student result:{RESET}")
    print(summary_df)


def main():
    if torch.cuda.is_available():
        print(f"{GREEN}GPU detected. Using GPU.{RESET}")
    else:
        print(f"{YELLOW}No GPU detected. Using CPU.{RESET}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    CONFIG_PATH = "../config/experiment.yaml"
    config = load_yaml_config(CONFIG_PATH)

    OUTPUT_DIR = config["output_dir"]

    TRAIN_DATASETS = config["train_datasets"]
    VAL_DATASETS = config["val_datasets"]
    INTERNAL_TEST_DATASETS = config["test_datasets"]

    prediction_stage = config.get("prediction_stage", {})
    PREDICT_SPLIT = prediction_stage.get("predict_split", "test")
    PREDICT_DATASETS = prediction_stage.get(
        "predict_datasets",
        INTERNAL_TEST_DATASETS,
    )

    RUN_MLM = True
    RUN_TRAINING = True
    RUN_PREDICTION = True

    # For prediction only after training:
    # RUN_MLM = False
    # RUN_TRAINING = False
    # RUN_PREDICTION = True

    logdata_read_obj = LogdataRead()
    utilities_obj = utils()

    print(f"\n{GREEN}Training stage{RESET}")
    print("-" * 60)
    print(f"Train datasets:         {TRAIN_DATASETS}")
    print(f"Validation datasets:    {VAL_DATASETS}")
    print(f"Internal test datasets: {INTERNAL_TEST_DATASETS}")

    print(f"\n{GREEN}Prediction stage{RESET}")
    print("-" * 60)
    print(f"Prediction split:       {PREDICT_SPLIT}")
    print(f"Prediction datasets:    {PREDICT_DATASETS}")

    if RUN_TRAINING:
        if RUN_MLM:
            mlm_cfg = config.get("mlm", {})

            if mlm_cfg.get("enabled", False):
                if mlm_cfg.get("dapt", {}).get("enabled", False):
                    print("\n# 1. DAPT MLM pretraining on source datasets")
                    run_command(
                        [
                            sys.executable,
                            "run_mlm.py",
                            "--config",
                            CONFIG_PATH,
                            "--stage",
                            "dapt",
                        ]
                    )

                if mlm_cfg.get("tapt", {}).get("enabled", False):
                    print("\n# 2. TAPT MLM pretraining on target dataset")
                    run_command(
                        [
                            sys.executable,
                            "run_mlm.py",
                            "--config",
                            CONFIG_PATH,
                            "--stage",
                            "tapt",
                        ]
                    )

        print("\n# 3. Student fine-tuning")
        run_command(
            [
                sys.executable,
                "run_experiment.py",
                "--config",
                CONFIG_PATH,
            ]
        )

    if RUN_PREDICTION:
        print(f"\n{GREEN}Running prediction using saved student model...{RESET}")

        metrics_df, report_df, per_dataset_df = run_prediction_for_student(
            config_path=CONFIG_PATH,
            output_dir=OUTPUT_DIR,
            train_datasets=TRAIN_DATASETS,
            val_datasets=VAL_DATASETS,
            internal_test_datasets=INTERNAL_TEST_DATASETS,
            predict_split=PREDICT_SPLIT,
            predict_datasets=PREDICT_DATASETS,
        )

        print_model_summary(
            metrics_df=metrics_df,
            report_df=report_df,
            per_dataset_df=per_dataset_df,
        )

        save_final_student_report(
            output_dir=OUTPUT_DIR,
            train_datasets=TRAIN_DATASETS,
            val_datasets=VAL_DATASETS,
            internal_test_datasets=INTERNAL_TEST_DATASETS,
            predict_datasets=PREDICT_DATASETS,
            metrics_df=metrics_df,
        )

    print(f"\n{GREEN}All steps completed successfully.{RESET}")


if __name__ == "__main__":
    main()