import os
import time
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler

from metrics import compute_metrics, print_metrics
from utils import ensure_dir, save_model_and_tokenizer


def build_dataset_balanced_sampler(dataset):
    dataset_names = dataset.get_dataset_names()

    dataset_counts = {}
    for name in dataset_names:
        dataset_counts[name] = dataset_counts.get(name, 0) + 1

    weights = [1.0 / dataset_counts[name] for name in dataset_names]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    return sampler


def build_class_weights(config, device):
    training_cfg = config.get("training", {})

    normal_weight = float(training_cfg.get("class_weight_normal", 1.0))
    anomaly_weight = float(training_cfg.get("class_weight_anomaly", 1.0))

    return torch.tensor(
        [normal_weight, anomaly_weight],
        dtype=torch.float,
        device=device,
    )


def train_classifier(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    config,
    output_dir,
    device,
):
    ensure_dir(output_dir)

    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = float(config["lr"])

    training_cfg = config.get("training", {})

    use_dataset_balanced_sampler = training_cfg.get(
        "use_dataset_balanced_sampler", False
    )

    use_class_weights = training_cfg.get(
        "use_class_weights", False
    )

    if use_dataset_balanced_sampler:
        sampler = build_dataset_balanced_sampler(train_dataset)
        shuffle = False
        print("Using dataset-balanced sampler for fine-tuning.")
    else:
        sampler = None
        shuffle = True
        print("Using normal shuffled sampler for fine-tuning.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if use_class_weights:
        class_weights = build_class_weights(config, device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        print("Using class-weighted loss.")
        print(f"Class 0 Normal weight:  {class_weights[0].item()}")
        print(f"Class 1 Anomaly weight: {class_weights[1].item()}")
    else:
        loss_fn = nn.CrossEntropyLoss()
        print("Using standard cross-entropy loss.")

    best_f1 = -1.0
    best_path = os.path.join(output_dir, "best_model")

    history = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Fine-tuning epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch["labels"]

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(len(train_loader), 1)

        val_metrics, _ = evaluate_classifier(
            model=model,
            dataset=val_dataset,
            batch_size=batch_size,
            device=device,
            title=f"Validation epoch {epoch + 1}",
            print_output=False,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={avg_train_loss:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_accuracy={val_metrics['accuracy']:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_model_and_tokenizer(model, tokenizer, best_path)

    total_time = time.time() - start_time

    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, "training_history.csv"),
        index=False,
    )

    pd.DataFrame(
        [
            {
                "best_val_f1": best_f1,
                "training_time_seconds": total_time,
                "use_dataset_balanced_sampler": use_dataset_balanced_sampler,
                "use_class_weights": use_class_weights,
            }
        ]
    ).to_csv(
        os.path.join(output_dir, "training_summary.csv"),
        index=False,
    )

    print(f"\nTraining finished in {total_time:.2f} seconds")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Best model saved to: {best_path}")

    return best_path


@torch.no_grad()
def evaluate_classifier(
    model,
    dataset,
    batch_size,
    device,
    title="Evaluation",
    print_output=True,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    progress_bar = tqdm(
        loader,
        desc=title,
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in progress_bar:
        labels = batch["labels"].cpu().numpy().tolist()
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        y_true.extend(labels)
        y_pred.extend(predictions.cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred)

    if print_output:
        print_metrics(title, y_true, y_pred)

    return metrics, y_pred