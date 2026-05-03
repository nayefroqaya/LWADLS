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
    internal_test_datasets,
    predict_split,
    predict_datasets,
):
    exp_name = make_experiment_name(
        model_mode,
        train_datasets,
        val_datasets,
        internal_test_datasets,
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
            predict_split,
            "--datasets",
            *predict_datasets,
        ]
    )

    prediction_folder = os.path.join(
        output_dir,
        "predictions",
        f"{exp_name}__best_model__split-{predict_split}__datasets-{clean_name(predict_datasets)}",
    )

    metrics_path = os.path.join(prediction_folder, "metrics.csv")
    report_path = os.path.join(prediction_folder, "classification_report.csv")
    per_dataset_metrics_path = os.path.join(prediction_folder, "per_dataset_metrics.csv")

    metrics_df = read_csv_if_exists(metrics_path)
    report_df = read_csv_if_exists(report_path)
    per_dataset_df = read_csv_if_exists(per_dataset_metrics_path)

    return metrics_df, report_df, per_dataset_df


def save_final_comparison_report(
    output_dir,
    train_datasets,
    val_datasets,
    internal_test_datasets,
    predict_datasets,
    summary_rows,
):
    summary_df = pd.DataFrame(summary_rows)

    summary_csv_path = os.path.join(
        output_dir,
        f"final_prediction_summary__train-{clean_name(train_datasets)}__predict-{clean_name(predict_datasets)}.csv",
    )

    summary_txt_path = os.path.join(
        output_dir,
        f"final_comparison__train-{clean_name(train_datasets)}__predict-{clean_name(predict_datasets)}.txt",
    )

    summary_df.to_csv(summary_csv_path, index=False)

    best_model = max(summary_rows, key=lambda x: float(x["f1"]))

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write("Experiment Setup\n")
        f.write("-" * 60 + "\n")
        f.write(f"Train datasets:         {train_datasets}\n")
        f.write(f"Validation datasets:    {val_datasets}\n")
        f.write(f"Internal test datasets: {internal_test_datasets}\n")
        f.write(f"Prediction datasets:    {predict_datasets}\n\n")

        f.write("Metric Meaning\n")
        f.write("-" * 60 + "\n")
        f.write("Precision, Recall, and F1 below are binary metrics for Class 1 = Anomaly.\n")
        f.write("Class 0 = Normal\n")
        f.write("Class 1 = Anomaly\n\n")

        f.write("Model Comparison\n")
        f.write("-" * 100 + "\n")
        f.write(
            f"{'Model':25s} {'Precision':>12s} {'Recall':>12s} {'F1-score':>12s} {'Accuracy':>12s}\n"
        )
        f.write("-" * 100 + "\n")

        for row in summary_rows:
            f.write(
                f"{row['model']:25s} "
                f"{float(row['precision']):12.6f} "
                f"{float(row['recall']):12.6f} "
                f"{float(row['f1']):12.6f} "
                f"{float(row['accuracy']):12.6f}\n"
            )

        f.write("\n")
        f.write("Best Model by F1-score\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"{best_model['model']} with F1-score = {float(best_model['f1']):.6f}\n\n"
        )

        f.write("Notes\n")
        f.write("-" * 60 + "\n")
        f.write(
            "For cross-dataset experiments, low F1 indicates domain shift/generalization difficulty.\n"
        )
        f.write(
            "Compare distill against student to evaluate the effect of distillation.\n"
        )

    print(f"\n{GREEN}Saved final CSV summary:{RESET}")
    print(summary_csv_path)

    print(f"\n{GREEN}Saved final TXT comparison report:{RESET}")
    print(summary_txt_path)

    print(f"\n{GREEN}Final model comparison:{RESET}")
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


    CONFIG_PATH = "../config/experiment.yaml"
    config = load_yaml_config(CONFIG_PATH)

    OUTPUT_DIR = config["output_dir"]

    TRAIN_DATASETS = config["train_datasets"]
    VAL_DATASETS = config["val_datasets"]
    INTERNAL_TEST_DATASETS = config["test_datasets"]

    prediction_stage = config.get("prediction_stage", {})
    PREDICT_SPLIT = prediction_stage.get("predict_split", "test")
    PREDICT_DATASETS = prediction_stage.get("predict_datasets", INTERNAL_TEST_DATASETS)

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
    '''
    if RUN_TRAINING:
        if RUN_MLM:
            print("\n# 1. MLM pretraining")
            run_command([sys.executable, "run_mlm.py", "--config", CONFIG_PATH])

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
    #-----------------------------------
    if RUN_PREDICTION:
        print(f"\n{GREEN}Running prediction using saved models...{RESET}")

        model_modes = [
            "teacher",
            "student",
            "distill",
        ]

        summary_rows = []

        for model_mode in model_modes:
            metrics_df, report_df, per_dataset_df = run_prediction_for_saved_model(
                config_path=CONFIG_PATH,
                output_dir=OUTPUT_DIR,
                model_mode=model_mode,
                train_datasets=TRAIN_DATASETS,
                val_datasets=VAL_DATASETS,
                internal_test_datasets=INTERNAL_TEST_DATASETS,
                predict_split=PREDICT_SPLIT,
                predict_datasets=PREDICT_DATASETS,
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
                        "train_datasets": clean_name(TRAIN_DATASETS),
                        "val_datasets": clean_name(VAL_DATASETS),
                        "internal_test_datasets": clean_name(INTERNAL_TEST_DATASETS),
                        "prediction_datasets": clean_name(PREDICT_DATASETS),
                        "precision": float(row["precision"]),
                        "recall": float(row["recall"]),
                        "f1": float(row["f1"]),
                        "accuracy": float(row["accuracy"]),
                    }
                )

        if summary_rows:
            save_final_comparison_report(
                output_dir=OUTPUT_DIR,
                train_datasets=TRAIN_DATASETS,
                val_datasets=VAL_DATASETS,
                internal_test_datasets=INTERNAL_TEST_DATASETS,
                predict_datasets=PREDICT_DATASETS,
                summary_rows=summary_rows,
            )

    print(f"\n{GREEN}All steps completed successfully.{RESET}")


if __name__ == "__main__":
    main()