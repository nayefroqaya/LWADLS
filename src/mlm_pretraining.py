import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling

from dataset import MLMDataset
from utils import ensure_dir


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

    dataset = MLMDataset(train_dataframe, tokenizer, max_length=max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"MLM epoch {epoch + 1}/{epochs} | loss={total_loss / len(loader):.4f}")

    total_time = time.time() - start_time

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    pd.DataFrame(
        [{
            "mlm_time_seconds": total_time,
            "mlm_epochs": epochs,
            "mlm_probability": mlm_probability,
        }]
    ).to_csv(os.path.join(output_dir, "mlm_summary.csv"), index=False)

    print(f"\nMLM saved to: {output_dir}")

    return output_dir