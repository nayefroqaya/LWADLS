import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_loader import load_split_from_dataset_folders
from tqdm import tqdm
from aggregator import aggregate_by_block, print_sequence_stats
from dataset import LogSequenceDataset
from metrics import (
    print_metrics,
    get_classification_report_df,
    per_dataset_metrics,
    per_dataset_classification_reports,
)
from utils import load_config, get_device, ensure_dir, clean_name


@torch.no_grad()
def predict_model(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    all_prob_normal = []
    all_prob_anomaly = []

    progress_bar = tqdm(
        loader,
        desc="Predicting",
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in progress_bar:
        labels = batch["labels"].cpu().numpy().tolist()
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits

        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        all_labels.extend(labels)
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_prob_normal.extend(probabilities[:, 0].cpu().numpy().tolist())
        all_prob_anomaly.extend(probabilities[:, 1].cpu().numpy().tolist())

        progress_bar.set_postfix(
            processed=len(all_predictions),
            total=len(dataset),
        )

    print(f"\nPrediction completed: {len(all_predictions)} samples processed.")

    return all_labels, all_predictions, all_prob_normal, all_prob_anomaly

def build_prediction_output_dir(config, model_path, split, datasets):
    model_path = os.path.normpath(model_path)

    model_folder_name = os.path.basename(model_path)
    parent_folder_name = os.path.basename(os.path.dirname(model_path))
    datasets_name = clean_name(datasets)

    output_dir = os.path.join(
        config["output_dir"],
        "predictions",
        f"{parent_folder_name}__{model_folder_name}__split-{split}__datasets-{datasets_name}",
    )

    return output_dir


def run_prediction(config_path, model_path, split, datasets):
    config = load_config(config_path)
    device = get_device()

    print("=" * 80)
    print("Prediction only — no training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model path: {model_path}")
    print(f"Split: {split}")
    print(f"Datasets: {datasets}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    print("\nLoading saved model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("\nLoading dataset split...")
    raw_df = load_split_from_dataset_folders(
        config=config,
        dataset_names=datasets,
        split=split,
    )

    columns = config["columns"]
    labels = config["labels"]

    print("\nAggregating logs by Node_block_id...")
    seq_df = aggregate_by_block(
        raw_df,
        timestamp_col=columns["timestamp"],
        template_col=columns["template"],
        block_col=columns["block_id"],
        label_col=columns["label"],
        normal_values=labels["normal_values"],
        anomaly_values=labels["anomaly_values"],
    )

    print_sequence_stats("Prediction data", seq_df)

    pred_dataset = LogSequenceDataset(
        seq_df,
        tokenizer,
        config["model"]["max_length"],
    )

    print("\nRunning prediction...")
    y_true, y_pred, prob_normal, prob_anomaly = predict_model(
        model=model,
        dataset=pred_dataset,
        batch_size=config["model"]["batch_size"],
        device=device,
    )

    metrics = print_metrics("Prediction results", y_true, y_pred)

    result_df = seq_df.copy()
    result_df["prediction"] = y_pred
    result_df["prob_normal"] = prob_normal
    result_df["prob_anomaly"] = prob_anomaly

    output_dir = build_prediction_output_dir(
        config=config,
        model_path=model_path,
        split=split,
        datasets=datasets,
    )

    ensure_dir(output_dir)

    predictions_path = os.path.join(output_dir, "predictions.csv")
    metrics_path = os.path.join(output_dir, "metrics.csv")
    classification_report_path = os.path.join(output_dir, "classification_report.csv")
    per_dataset_metrics_path = os.path.join(output_dir, "per_dataset_metrics.csv")
    per_dataset_report_dir = os.path.join(
        output_dir,
        "per_dataset_classification_reports",
    )

    ensure_dir(per_dataset_report_dir)

    result_df.to_csv(predictions_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    overall_report_df = get_classification_report_df(y_true, y_pred)
    overall_report_df.to_csv(classification_report_path)

    per_metrics_df = per_dataset_metrics(result_df)
    per_metrics_df.to_csv(per_dataset_metrics_path, index=False)

    per_reports = per_dataset_classification_reports(result_df)

    for dataset_name, report_df in per_reports.items():
        report_path = os.path.join(
            per_dataset_report_dir,
            f"classification_report_{dataset_name}.csv",
        )
        report_df.to_csv(report_path)

    print("\nOverall classification report:")
    print(overall_report_df)

    print("\nPer-dataset metrics:")
    print(per_metrics_df)

    print("\nSaved files:")
    print(f"Predictions:                   {predictions_path}")
    print(f"Overall metrics:               {metrics_path}")
    print(f"Overall classification report: {classification_report_path}")
    print(f"Per-dataset metrics:           {per_dataset_metrics_path}")
    print(f"Per-dataset reports folder:    {per_dataset_report_dir}")

    return metrics, result_df, output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction using a saved anomaly detection model."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML file.",
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to saved model folder, e.g. ../outputs/.../best_model",
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
        help="Which split to predict on.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names, e.g. BGL HDFS TH_b",
    )

    args = parser.parse_args()

    run_prediction(
        config_path=args.config,
        model_path=args.model_path,
        split=args.split,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()