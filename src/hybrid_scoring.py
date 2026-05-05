import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import load_split_from_dataset_folders
from aggregator import aggregate_by_block, print_sequence_stats
from dataset import LogSequenceDataset


@torch.no_grad()
def mean_pool_last_hidden(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    masked_hidden = hidden_states * mask
    summed = masked_hidden.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def extract_embeddings_and_probs(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    all_labels = []
    all_prob_normal = []
    all_prob_anomaly = []
    all_embeddings = []

    for batch in tqdm(loader, desc="Extracting embeddings/probabilities", unit="batch"):
        labels = batch["labels"].cpu().numpy().tolist()

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        last_hidden = outputs.hidden_states[-1]
        embeddings = mean_pool_last_hidden(last_hidden, batch["attention_mask"])

        all_labels.extend(labels)
        all_prob_normal.extend(probs[:, 0].cpu().numpy().tolist())
        all_prob_anomaly.extend(probs[:, 1].cpu().numpy().tolist())
        all_embeddings.append(embeddings.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    return all_labels, all_prob_normal, all_prob_anomaly, all_embeddings


def load_reference_dataframe(config):
    hybrid_cfg = config.get("hybrid_scoring", {})

    split = hybrid_cfg.get("reference_split", "train")
    datasets = hybrid_cfg.get("reference_datasets", config["train_datasets"])

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

    if hybrid_cfg.get("normal_only", True):
        seq_df = seq_df[seq_df["label"] == 0].reset_index(drop=True)

    print_sequence_stats("Hybrid reference data", seq_df)

    return seq_df


def build_hybrid_reference(model, tokenizer, config, device):
    ref_df = load_reference_dataframe(config)

    ref_dataset = LogSequenceDataset(
        ref_df,
        tokenizer,
        config["model"]["max_length"],
    )

    _, _, _, ref_embeddings = extract_embeddings_and_probs(
        model=model,
        dataset=ref_dataset,
        batch_size=config["model"]["batch_size"],
        device=device,
    )

    centroid = ref_embeddings.mean(axis=0)

    distances = np.linalg.norm(ref_embeddings - centroid, axis=1)

    distance_mean = float(distances.mean())
    distance_std = float(distances.std() + 1e-9)

    reference = {
        "centroid": centroid,
        "distance_mean": distance_mean,
        "distance_std": distance_std,
    }

    print("\nHybrid reference built")
    print("-" * 60)
    print(f"Reference samples: {len(ref_embeddings)}")
    print(f"Distance mean:     {distance_mean:.6f}")
    print(f"Distance std:      {distance_std:.6f}")

    return reference


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_distance_scores(embeddings, reference):
    centroid = reference["centroid"]
    mean = reference["distance_mean"]
    std = reference["distance_std"]

    distances = np.linalg.norm(embeddings - centroid, axis=1)

    z_scores = (distances - mean) / std
    distance_scores = sigmoid(z_scores)

    return distances.tolist(), distance_scores.tolist()


def compute_hybrid_scores(prob_anomaly, distance_scores, config):
    hybrid_cfg = config.get("hybrid_scoring", {})

    alpha_classifier = float(hybrid_cfg.get("alpha_classifier", 0.7))
    alpha_distance = float(hybrid_cfg.get("alpha_distance", 0.3))

    prob_anomaly = np.asarray(prob_anomaly)
    distance_scores = np.asarray(distance_scores)

    hybrid_scores = (
        alpha_classifier * prob_anomaly
        + alpha_distance * distance_scores
    )

    return hybrid_scores.tolist()