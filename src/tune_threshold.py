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


def search_threshold(y_true, scores, start, end, step):
    rows = []
    threshold = float(start)

    while threshold <= float(end) + 1e-9:
        y_pred = apply_threshold(scores, threshold)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        accuracy = accuracy_score(y_true, y_pred)

        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
            }
        )

        threshold += float(step)

    df = pd.DataFrame(rows)
    best = df.sort_values("f1", ascending=False).iloc[0].to_dict()

    return df, best


def run_threshold_tuning(config_path, model_path):
    config = load_config(config_path)
    device = get_device()

    tuning_cfg = config.get("threshold_tuning", {})

    split = tuning_cfg.get("split", "val")
    datasets = tuning_cfg.get("datasets", config["val_datasets"])

    hybrid_enabled = config.get("hybrid_scoring", {}).get("enabled", False)

    print("=" * 80)
    print("THRESHOLD TUNING ON SOURCE VALIDATION DATA")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Split: {split}")
    print(f"Datasets: {datasets}")
    print(f"Hybrid scoring enabled: {hybrid_enabled}")

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

        scores = compute_hybrid_scores(
            prob_anomaly=prob_anomaly,
            distance_scores=distance_scores,
            config=config,
        )

        scoring_type = "hybrid"
    else:
        scores = prob_anomaly
        scoring_type = "classifier_probability"

    results_df, best = search_threshold(
        y_true=y_true,
        scores=scores,
        start=float(tuning_cfg.get("start", 0.01)),
        end=float(tuning_cfg.get("end", 0.90)),
        step=float(tuning_cfg.get("step", 0.01)),
    )

    parent_name = os.path.basename(os.path.dirname(os.path.normpath(model_path)))

    output_dir = os.path.join(
        config["output_dir"],
        "threshold_tuning",
        f"{parent_name}__val-{clean_name(datasets)}",
    )

    ensure_dir(output_dir)

    results_path = os.path.join(output_dir, "threshold_search_results.csv")
    best_path = os.path.join(output_dir, "best_threshold.csv")

    results_df["scoring_type"] = scoring_type
    results_df.to_csv(results_path, index=False)

    best["scoring_type"] = scoring_type
    pd.DataFrame([best]).to_csv(best_path, index=False)

    print("\nBest threshold:")
    print(best)

    print("\nSaved:")
    print(results_path)
    print(best_path)

    return best["threshold"]


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