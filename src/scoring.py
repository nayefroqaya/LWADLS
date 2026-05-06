from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


@torch.no_grad()
def score_loader(model, loader, center, device, alpha_mlm: float, beta_center: float):
    model.eval()

    rows = []
    center = center.to(device)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_mlm = batch["labels"].to(device)
        labels_cls = batch["labels_cls"].cpu().numpy()

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_mlm,
        )

        # Batch-level MLM loss.
        # Later you can replace this with per-sample token loss if needed.
        mlm_loss = float(out["mlm_loss"].detach().cpu())

        emb = out["embedding"]
        dist = ((emb - center) ** 2).sum(dim=1).detach().cpu().numpy()

        batch_mlm = np.full(
            shape=(input_ids.size(0),),
            fill_value=mlm_loss,
            dtype=np.float32,
        )

        batch_scores = alpha_mlm * batch_mlm + beta_center * dist

        for i in range(input_ids.size(0)):
            rows.append(
                {
                    "score": float(batch_scores[i]),
                    "mlm_loss": float(batch_mlm[i]),
                    "center_distance": float(dist[i]),
                    "label": int(labels_cls[i]),
                    "dataset": batch["dataset_names"][i],
                    "group_id": batch["group_ids"][i],
                    "sequence": batch["sequence_texts"][i],
                }
            )

    return pd.DataFrame(rows)


def calibrate_threshold(
    scores_normal,
    method: str = "percentile",
    percentile: float = 95.0,
    fixed_threshold: float = 0.5,
):
    scores_normal = np.asarray(scores_normal, dtype=np.float64)

    if method == "fixed":
        return float(fixed_threshold)

    if len(scores_normal) == 0:
        raise ValueError("Cannot calibrate threshold with empty normal scores.")

    if method == "percentile":
        return float(np.percentile(scores_normal, percentile))

    if method == "mean_std":
        return float(scores_normal.mean() + 3.0 * scores_normal.std())

    raise ValueError(f"Unknown threshold method: {method}")


def evaluate_scores(score_df: pd.DataFrame, threshold: float, normal_label: int = 0):
    """
    Evaluates prediction results.

    Class mapping:
        0 = normal
        1 = anomaly

    Confusion matrix format:
        [[TN, FP],
         [FN, TP]]
    """

    y_true = (score_df["label"].to_numpy() != int(normal_label)).astype(int)
    y_pred = (score_df["score"].to_numpy() > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    acc = accuracy_score(y_true, y_pred)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["normal", "anomaly"],
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["normal", "anomaly"],
        zero_division=0,
        output_dict=False,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "threshold": float(threshold),

        # Binary anomaly-focused metrics.
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),

        # Dataset statistics.
        "num_samples": int(len(score_df)),
        "num_anomalies": int(y_true.sum()),
        "num_normals": int((y_true == 0).sum()),

        # Per-class classification report.
        "classification_report": report_dict,
        "classification_report_text": report_text,

        # Confusion matrix.
        "confusion_matrix": {
            "labels": ["normal", "anomaly"],
            "matrix": cm.tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }

    try:
        metrics["auc"] = float(
            roc_auc_score(y_true, score_df["score"].to_numpy())
        )
    except Exception:
        metrics["auc"] = None

    return metrics


def save_classification_report_files(
    metrics: Dict[str, Any],
    output_dir,
    prefix: str = "test",
):
    """
    Saves:
        test_classification_report.txt
        test_classification_report.csv
        test_confusion_matrix.csv
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save text report.
    report_text = metrics.get("classification_report_text", "")
    report_txt_path = output_dir / f"{prefix}_classification_report.txt"
    report_txt_path.write_text(report_text, encoding="utf-8")

    # Save CSV report.
    report_dict = metrics.get("classification_report", {})
    if report_dict:
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(output_dir / f"{prefix}_classification_report.csv")

    # Save confusion matrix.
    cm_info = metrics.get("confusion_matrix", {})
    matrix = cm_info.get("matrix")
    labels = cm_info.get("labels", ["normal", "anomaly"])

    if matrix is not None:
        cm_df = pd.DataFrame(
            matrix,
            index=[f"true_{x}" for x in labels],
            columns=[f"pred_{x}" for x in labels],
        )
        cm_df.to_csv(output_dir / f"{prefix}_confusion_matrix.csv")