import os
import argparse
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_split_from_dataset_folders
from aggregator import aggregate_by_block, print_sequence_stats
from dataset import LogSequenceDataset
from metrics import (
    print_metrics,
    get_classification_report_df,
    per_dataset_metrics,
    per_dataset_classification_reports,
)
from utils import load_config, get_device, ensure_dir, clean_name
from hybrid_scoring import (
    build_hybrid_reference,
    extract_embeddings_and_probs,
    compute_distance_scores,
    compute_hybrid_scores,
)


def apply_threshold(scores, threshold):
    return [1 if p >= threshold else 0 for p in scores]


def build_prediction_output_dir(config, model_path, split, datasets, threshold):
    model_path = os.path.normpath(model_path)

    model_folder_name = os.path.basename(model_path)
    parent_folder_name = os.path.basename(os.path.dirname(model_path))
    datasets_name = clean_name(datasets)

    threshold_name = str(round(float(threshold), 6)).replace(".", "p")

    output_dir = os.path.join(
        config["output_dir"],
        "predictions",
        f"{parent_folder_name}__{model_folder_name}__split-{split}"
        f"__datasets-{datasets_name}__thr-{threshold_name}",
    )

    return output_dir


def save_text_report(
    file_path,
    metrics,
    y_true,
    y_pred,
    per_dataset_df,
    datasets,
    model_path,
    split,
    threshold,
    hybrid_enabled,
    alpha_classifier,
    alpha_distance,
):
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Class 0 - Normal", "Class 1 - Anomaly"],
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Class 0 - Normal", "Class 1 - Anomaly"],
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("ANOMALY DETECTION EVALUATION REPORT\n")
        f.write("=" * 90 + "\n\n")

        f.write(f"Model path: {model_path}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Datasets: {datasets}\n")
        f.write(f"Anomaly threshold: {threshold}\n")
        f.write(f"Hybrid scoring enabled: {hybrid_enabled}\n")
        f.write(f"Alpha classifier: {alpha_classifier}\n")
        f.write(f"Alpha distance: {alpha_distance}\n\n")

        f.write("Label Meaning\n")
        f.write("-" * 50 + "\n")
        f.write("Class 0 = Normal\n")
        f.write("Class 1 = Anomaly\n\n")

        f.write("Overall Metrics\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.6f}\n")
        f.write(f"Precision: {metrics['precision']:.6f}  # Class 1 / Anomaly\n")
        f.write(f"Recall:    {metrics['recall']:.6f}  # Class 1 / Anomaly\n")
        f.write(f"F1-score:  {metrics['f1']:.6f}  # Class 1 / Anomaly\n\n")

        f.write("Per-Class Metrics\n")
        f.write("-" * 50 + "\n")

        for class_name in ["Class 0 - Normal", "Class 1 - Anomaly"]:
            class_metrics = report_dict[class_name]

            f.write(f"{class_name}\n")
            f.write(f"  Precision: {class_metrics['precision']:.6f}\n")
            f.write(f"  Recall:    {class_metrics['recall']:.6f}\n")
            f.write(f"  F1-score:  {class_metrics['f1-score']:.6f}\n")
            f.write(f"  Support:   {int(class_metrics['support'])}\n\n")

        f.write("Full Classification Report\n")
        f.write("-" * 50 + "\n")
        f.write(report_text + "\n")

        f.write("Confusion Matrix\n")
        f.write("-" * 50 + "\n")
        f.write("Rows = true labels, Columns = predicted labels\n")
        f.write("Order: [Class 0 - Normal, Class 1 - Anomaly]\n\n")
        f.write(str(cm) + "\n\n")

        f.write("Confusion Matrix Details\n")
        f.write("-" * 50 + "\n")
        f.write(f"True Normal predicted Normal   (TN): {tn}\n")
        f.write(f"True Normal predicted Anomaly  (FP): {fp}\n")
        f.write(f"True Anomaly predicted Normal  (FN): {fn}\n")
        f.write(f"True Anomaly predicted Anomaly (TP): {tp}\n\n")

        if per_dataset_df is not None:
            f.write("Per-Dataset Metrics\n")
            f.write("-" * 50 + "\n")
            f.write(per_dataset_df.to_string(index=False))
            f.write("\n\n")

        f.write("=" * 90 + "\n")

    print(f"\nSaved readable report: {file_path}")


def run_prediction(
    config_path,
    model_path,
    split,
    datasets,
    threshold_override=None,
    alpha_classifier_override=None,
):
    config = load_config(config_path)
    device = get_device()

    prediction_stage = config.get("prediction_stage", {})
    hybrid_cfg = config.get("hybrid_scoring", {})

    threshold = float(prediction_stage.get("anomaly_threshold", 0.5))
    if threshold_override is not None:
        threshold = float(threshold_override)

    hybrid_enabled = bool(hybrid_cfg.get("enabled", False))

    alpha_classifier = float(hybrid_cfg.get("alpha_classifier", 1.0))
    if alpha_classifier_override is not None:
        alpha_classifier = float(alpha_classifier_override)

    alpha_distance = 1.0 - alpha_classifier

    print("=" * 80)
    print("Prediction only — no training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model path: {model_path}")
    print(f"Split: {split}")
    print(f"Datasets: {datasets}")
    print(f"Anomaly threshold: {threshold}")
    print(f"Hybrid scoring enabled: {hybrid_enabled}")
    print(f"Alpha classifier: {alpha_classifier}")
    print(f"Alpha distance: {alpha_distance}")

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
        config=config,
    )

    print_sequence_stats("Prediction data", seq_df)

    pred_dataset = LogSequenceDataset(
        seq_df,
        tokenizer,
        config["model"]["max_length"],
    )

    print("\nRunning model to get probabilities and embeddings...")
    y_true, prob_normal, prob_anomaly, embeddings = extract_embeddings_and_probs(
        model=model,
        dataset=pred_dataset,
        batch_size=config["model"]["batch_size"],
        device=device,
    )

    if hybrid_enabled:
        reference = build_hybrid_reference(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
        )

        distances, distance_scores = compute_distance_scores(
            embeddings=embeddings,
            reference=reference,
        )

        anomaly_scores = compute_hybrid_scores(
            prob_anomaly=prob_anomaly,
            distance_scores=distance_scores,
            alpha_classifier=alpha_classifier,
        )

        scoring_type = "hybrid"
    else:
        distances = [None] * len(prob_anomaly)
        distance_scores = [None] * len(prob_anomaly)
        anomaly_scores = prob_anomaly
        scoring_type = "classifier_probability"

    y_pred = apply_threshold(anomaly_scores, threshold)

    print(f"\nFinal anomaly threshold used: {threshold}")
    print(f"Scoring type: {scoring_type}")

    metrics = print_metrics("Prediction results", y_true, y_pred)

    result_df = seq_df.copy()
    result_df["prediction"] = y_pred
    result_df["prob_normal"] = prob_normal
    result_df["prob_anomaly"] = prob_anomaly
    result_df["embedding_distance"] = distances
    result_df["distance_score"] = distance_scores
    result_df["final_anomaly_score"] = anomaly_scores
    result_df["anomaly_threshold"] = threshold
    result_df["alpha_classifier"] = alpha_classifier
    result_df["alpha_distance"] = alpha_distance
    result_df["scoring_type"] = scoring_type

    output_dir = build_prediction_output_dir(
        config=config,
        model_path=model_path,
        split=split,
        datasets=datasets,
        threshold=threshold,
    )

    ensure_dir(output_dir)

    predictions_path = os.path.join(output_dir, "predictions.csv")
    metrics_path = os.path.join(output_dir, "metrics.csv")
    classification_report_path = os.path.join(output_dir, "classification_report.csv")
    per_dataset_metrics_path = os.path.join(output_dir, "per_dataset_metrics.csv")
    text_report_path = os.path.join(output_dir, "evaluation_report.txt")
    per_dataset_report_dir = os.path.join(
        output_dir,
        "per_dataset_classification_reports",
    )

    ensure_dir(per_dataset_report_dir)

    result_df.to_csv(predictions_path, index=False)

    metrics_with_threshold = metrics.copy()
    metrics_with_threshold["anomaly_threshold"] = threshold
    metrics_with_threshold["alpha_classifier"] = alpha_classifier
    metrics_with_threshold["alpha_distance"] = alpha_distance
    metrics_with_threshold["scoring_type"] = scoring_type
    metrics_with_threshold["hybrid_enabled"] = hybrid_enabled
    pd.DataFrame([metrics_with_threshold]).to_csv(metrics_path, index=False)

    overall_report_df = get_classification_report_df(y_true, y_pred)
    overall_report_df.to_csv(classification_report_path)

    per_metrics_df = per_dataset_metrics(result_df)
    per_metrics_df["anomaly_threshold"] = threshold
    per_metrics_df["alpha_classifier"] = alpha_classifier
    per_metrics_df["alpha_distance"] = alpha_distance
    per_metrics_df["scoring_type"] = scoring_type
    per_metrics_df.to_csv(per_dataset_metrics_path, index=False)

    save_text_report(
        file_path=text_report_path,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        per_dataset_df=per_metrics_df,
        datasets=datasets,
        model_path=model_path,
        split=split,
        threshold=threshold,
        hybrid_enabled=hybrid_enabled,
        alpha_classifier=alpha_classifier,
        alpha_distance=alpha_distance,
    )

    per_reports = per_dataset_classification_reports(result_df)

    for dataset_name, report_df in per_reports.items():
        report_path = os.path.join(
            per_dataset_report_dir,
            f"classification_report_{dataset_name}.csv",
        )
        report_df.to_csv(report_path)

    print("\nSaved files:")
    print(f"Predictions:                   {predictions_path}")
    print(f"Overall metrics:               {metrics_path}")
    print(f"Overall classification report: {classification_report_path}")
    print(f"Readable text report:          {text_report_path}")
    print(f"Per-dataset metrics:           {per_dataset_metrics_path}")
    print(f"Per-dataset reports folder:    {per_dataset_report_dir}")

    return metrics, result_df, output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction using a saved anomaly detection model."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--alpha_classifier", type=float, default=None)

    args = parser.parse_args()

    run_prediction(
        config_path=args.config,
        model_path=args.model_path,
        split=args.split,
        datasets=args.datasets,
        threshold_override=args.threshold,
        alpha_classifier_override=args.alpha_classifier,
    )


if __name__ == "__main__":
    main()