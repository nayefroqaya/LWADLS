import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling
from tqdm import tqdm

from dataset import MLMDataset
from utils import ensure_dir


def build_balanced_sampler(dataframe):
    dataset_counts = dataframe["DatasetName"].value_counts().to_dict()

    weights = dataframe["DatasetName"].apply(
        lambda name: 1.0 / dataset_counts[name]
    ).tolist()

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    return sampler


def remove_metadata_from_batch(batch):
    if "DatasetName" in batch:
        del batch["DatasetName"]
    return batch


def run_mlm_pretraining(
    train_dataframe,
    tokenizer,
    mlm_config,
    model_name,
    max_length,
    output_dir,
    device,
):
    ensure_dir(output_dir)

    batch_size = mlm_config["batch_size"]
    epochs = mlm_config["epochs"]
    lr = float(mlm_config["lr"])
    mlm_probability = float(mlm_config["mlm_probability"])
    balance_datasets = bool(mlm_config.get("balance_datasets", False))

    dataset = MLMDataset(train_dataframe, tokenizer, max_length=max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    sampler = build_balanced_sampler(train_dataframe) if balance_datasets else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=data_collator,
    )

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start_time = time.time()

    print(f"\nMLM output directory: {output_dir}")
    print(f"MLM model base: {model_name}")
    print(f"Balanced MLM sampling: {balance_datasets}")
    print("MLM datasets:", train_dataframe["DatasetName"].value_counts().to_dict())

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            loader,
            desc=f"MLM epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )

        for batch in progress_bar:
            batch = remove_metadata_from_batch(batch)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(len(loader), 1)
        print(f"MLM epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f}")

    total_time = time.time() - start_time

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    pd.DataFrame(
        [
            {
                "mlm_time_seconds": total_time,
                "mlm_epochs": epochs,
                "mlm_probability": mlm_probability,
                "balance_datasets": balance_datasets,
                "datasets": ",".join(train_dataframe["DatasetName"].unique()),
            }
        ]
    ).to_csv(os.path.join(output_dir, "mlm_summary.csv"), index=False)

    print(f"\nMLM saved to: {output_dir}")

    return output_dir