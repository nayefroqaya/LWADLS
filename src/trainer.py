import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from metrics import compute_metrics, print_metrics, per_dataset_metrics
from utils import ensure_dir, save_model_and_tokenizer


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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_path = os.path.join(output_dir, "best_model")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

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
            f"train_loss={avg_loss:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_model_and_tokenizer(model, tokenizer, best_path)

    total_time = time.time() - start_time

    pd.DataFrame(
        [{
            "best_val_f1": best_f1,
            "training_time_seconds": total_time,
        }]
    ).to_csv(os.path.join(output_dir, "training_summary.csv"), index=False)

    print(f"\nTraining finished in {total_time:.2f} seconds")
    print(f"Best validation F1: {best_f1:.4f}")

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

    all_labels = []
    all_predictions = []

    for batch in loader:
        labels = batch["labels"].cpu().numpy().tolist()

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()

        all_labels.extend(labels)
        all_predictions.extend(predictions)

    metrics = compute_metrics(all_labels, all_predictions)

    if print_output:
        print_metrics(title, all_labels, all_predictions)

    return metrics, all_predictions


def evaluate_with_dataframe(
    model,
    dataframe,
    dataset,
    batch_size,
    device,
    output_dir,
    title,
):
    metrics, predictions = evaluate_classifier(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        title=title,
        print_output=True,
    )

    result_df = dataframe.copy()
    result_df["prediction"] = predictions

    ensure_dir(output_dir)

    result_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    per_df = per_dataset_metrics(result_df)
    per_df.to_csv(os.path.join(output_dir, "per_dataset_metrics.csv"), index=False)

    print("\nPer-dataset metrics:")
    print(per_df)

    return metrics