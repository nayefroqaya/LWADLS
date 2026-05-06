import os
import argparse
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data_loader import load_split_from_dataset_folders
from aggregator import aggregate_by_block, print_sequence_stats
from dataset import LogSequenceDataset
from utils import load_config, get_device, ensure_dir, clean_name
from hybrid_scoring import (
    build_hybrid_reference,
    extract_embeddings_and_probs,
    compute_distance_scores,
    compute_hybrid_scores,
)


def apply_threshold(scores, threshold):
    return [1 if p >= threshold else 0 for p in scores]


def evaluate_scores(y_true, scores, threshold):
    y_pred = apply_threshold(scores, threshold)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    accuracy = accuracy_score(y_true, y_pred)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def search_threshold_and_alpha(
    y_true,
    prob_anomaly,
    distance_scores,
    hybrid_enabled,
    threshold_start,
    threshold_end,
    threshold_step,
    alpha_classifier_values,
):
    rows = []

    if not hybrid_enabled:
        alpha_classifier_values = [1.0]

    for alpha_classifier in alpha_classifier_values:
        alpha_classifier = float(alpha_classifier)
        alpha_distance = 1.0 - alpha_classifier

        if hybrid_enabled:
            scores = compute_hybrid_scores(
                prob_anomaly=prob_anomaly,
                distance_scores=distance_scores,
                alpha_classifier=alpha_classifier,
            )
            scoring_type = "hybrid"
        else:
            scores = prob_anomaly
            scoring_type = "classifier_probability"

        threshold = float(threshold_start)

        while threshold <= float(threshold_end) + 1e-9:
            metrics = evaluate_scores(
                y_true=y_true,
                scores=scores,
                threshold=threshold,
            )

            rows.append(
                {
                    "threshold": float(threshold),
                    "alpha_classifier": alpha_classifier,
                    "alpha_distance": alpha_distance,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"],
                    "scoring_type": scoring_type,
                }
            )

            threshold += float(threshold_step)

    results_df = pd.DataFrame(rows)
    best_row = results_df.sort_values("f1", ascending=False).iloc[0].to_dict()

    return results_df, best_row


def run_threshold_tuning(config_path, model_path):
    config = load_config(config_path)
    device = get_device()

    tuning_cfg = config.get("threshold_tuning", {})
    hybrid_cfg = config.get("hybrid_scoring", {})

    split = tuning_cfg.get("split", "val")
    datasets = tuning_cfg.get("datasets", config["val_datasets"])

    hybrid_enabled = bool(hybrid_cfg.get("enabled", False))

    threshold_start = float(tuning_cfg.get("threshold_start", tuning_cfg.get("start", 0.01)))
    threshold_end = float(tuning_cfg.get("threshold_end", tuning_cfg.get("end", 0.90)))
    threshold_step = float(tuning_cfg.get("threshold_step", tuning_cfg.get("step", 0.01)))

    alpha_search_cfg = tuning_cfg.get("alpha_search", {})
    alpha_search_enabled = bool(alpha_search_cfg.get("enabled", True))

    if hybrid_enabled and alpha_search_enabled:
        alpha_classifier_values = alpha_search_cfg.get(
            "alpha_classifier_values",
            [1.0, 0.9, 0.8, 0.7, 0.6],
        )
    else:
        alpha_classifier_values = [
            float(hybrid_cfg.get("alpha_classifier", 1.0))
        ]

    print("=" * 80)
    print("THRESHOLD + ALPHA TUNING ON SOURCE VALIDATION DATA")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Split: {split}")
    print(f"Datasets: {datasets}")
    print(f"Hybrid scoring enabled: {hybrid_enabled}")
    print(f"Threshold range: {threshold_start} to {threshold_end}, step={threshold_step}")
    print(f"Alpha classifier values: {alpha_classifier_values}")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    raw_df = load_split_from_dataset_folders(
        config=config,
        dataset_names=datasets,
        split=split,
    )

    columns = config["columns"]
    labels = config["labels"]

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

    print_sequence_stats("Threshold tuning data", seq_df)

    dataset = LogSequenceDataset(
        seq_df,
        tokenizer,
        config["model"]["max_length"],
    )

    y_true, prob_normal, prob_anomaly, embeddings = extract_embeddings_and_probs(
        model=model,
        dataset=dataset,
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
    else:
        distance_scores = [0.0] * len(prob_anomaly)

    results_df, best = search_threshold_and_alpha(
        y_true=y_true,
        prob_anomaly=prob_anomaly,
        distance_scores=distance_scores,
        hybrid_enabled=hybrid_enabled,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        threshold_step=threshold_step,
        alpha_classifier_values=alpha_classifier_values,
    )

    parent_name = os.path.basename(os.path.dirname(os.path.normpath(model_path)))

    output_dir = os.path.join(
        config["output_dir"],
        "threshold_tuning",
        f"{parent_name}__val-{clean_name(datasets)}",
    )

    ensure_dir(output_dir)

    results_path = os.path.join(output_dir, "threshold_alpha_search_results.csv")
    best_path = os.path.join(output_dir, "best_threshold.csv")

    results_df.to_csv(results_path, index=False)
    pd.DataFrame([best]).to_csv(best_path, index=False)

    print("\nBest threshold + alpha:")
    print(best)

    print("\nSaved:")
    print(results_path)
    print(best_path)

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)

    args = parser.parse_args()

    run_threshold_tuning(
        config_path=args.config,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()