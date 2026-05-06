from typing import List, Dict, Any

import torch
from torch.optim import AdamW
from tqdm import tqdm

from .losses import center_loss


def train_normality(
    model,
    loader,
    device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    center_loss_weight: float,
    gradient_clip_norm: float | None = None,
    stage_name: str = "train",
):
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        total_loss = 0.0
        total_mlm = 0.0
        total_center = 0.0
        n = 0

        progress = tqdm(loader, desc=f"{stage_name} epoch {epoch}/{epochs}")

        for batch in progress:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            mlm_loss = out["mlm_loss"]
            c_loss = center_loss(out["embedding"])
            loss = mlm_loss + float(center_loss_weight) * c_loss

            loss.backward()

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))

            optimizer.step()

            bs = input_ids.size(0)
            total_loss += loss.item() * bs
            total_mlm += mlm_loss.item() * bs
            total_center += c_loss.item() * bs
            n += bs

            progress.set_postfix(loss=total_loss / max(n, 1))

        history.append({
            "epoch": epoch,
            "loss": total_loss / max(n, 1),
            "mlm_loss": total_mlm / max(n, 1),
            "center_loss": total_center / max(n, 1),
        })

    return history
