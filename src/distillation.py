import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from trainer import evaluate_classifier
from utils import ensure_dir, save_model_and_tokenizer


def train_distilled_student(
    teacher,
    student,
    tokenizer,
    train_dataset,
    val_dataset,
    model_config,
    distill_config,
    output_dir,
    device,
):
    ensure_dir(output_dir)

    batch_size = model_config["batch_size"]
    epochs = distill_config["epochs"]
    lr = float(distill_config["lr"])

    temperature = float(distill_config["temperature"])
    alpha = float(distill_config["alpha"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    teacher.to(device)
    student.to(device)

    teacher.eval()

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    best_f1 = -1.0
    best_path = os.path.join(output_dir, "best_model")

    start_time = time.time()

    for epoch in range(epochs):
        student.train()

        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            with torch.no_grad():
                teacher_logits = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits

            student_logits = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits

            loss_ce = F.cross_entropy(student_logits, labels)

            loss_kd = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)

            loss = alpha * loss_ce + (1.0 - alpha) * loss_kd

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_kd += loss_kd.item()

        val_metrics, _ = evaluate_classifier(
            model=student,
            dataset=val_dataset,
            batch_size=batch_size,
            device=device,
            title=f"Distillation validation epoch {epoch + 1}",
            print_output=False,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={total_loss / len(train_loader):.4f} | "
            f"ce={total_ce / len(train_loader):.4f} | "
            f"kd={total_kd / len(train_loader):.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_model_and_tokenizer(student, tokenizer, best_path)

    total_time = time.time() - start_time

    pd.DataFrame(
        [{
            "best_val_f1": best_f1,
            "distillation_time_seconds": total_time,
        }]
    ).to_csv(os.path.join(output_dir, "distillation_summary.csv"), index=False)

    print(f"\nDistillation finished in {total_time:.2f} seconds")
    print(f"Best validation F1: {best_f1:.4f}")

    return best_path