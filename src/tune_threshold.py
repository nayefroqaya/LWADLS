import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data_loader import load_split_from_dataset_folders
from aggregator import aggregate_by_block, print_sequence_stats
from dataset import LogSequenceDataset
from utils import load_config, get_device, ensure_dir, clean_name


@torch.no_grad()
def get_probs(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    y_true = []
    prob_anomaly = []

    for batch in tqdm(loader, desc="Getting validation probabilities", unit="batch"):
        labels = batch["labels"].cpu().numpy().tolist()
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        probs = torch.softmax(outputs.logits, dim=-1)

        y_true.extend(labels)
        prob_anomaly.extend(probs[:, 1].cpu().numpy().tolist())

    return y_true, prob_anomaly


def apply_threshold(prob_anomaly, threshold):
    return [1 if p >= threshold else 0 for p in prob_anomaly]


def search_threshold(y_true, prob_anomaly, start, end, step):
    rows = []
    threshold = float(start)

    while threshold <= float(end) + 1e-9:
        y_pred = apply_threshold(prob_anomaly, threshold)

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

    print("=" * 80)
    print("THRESHOLD TUNING ON SOURCE VALIDATION DATA")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Split: {split}")
    print(f"Datasets: {datasets}")

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

    y_true, prob_anomaly = get_probs(
        model=model,
        dataset=dataset,
        batch_size=config["model"]["batch_size"],
        device=device,
    )

    results_df, best = search_threshold(
        y_true=y_true,
        prob_anomaly=prob_anomaly,
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

    results_df.to_csv(results_path, index=False)
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