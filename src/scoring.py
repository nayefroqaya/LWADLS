from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


@torch.no_grad()
def score_loader(
    model,
    loader,
    center,
    device,
    alpha_mlm: float,
    beta_center: float,
    desc: str = "Scoring",
):
    model.eval()

    rows = []
    center = center.to(device)

    progress = tqdm(loader, desc=desc, unit="batch")

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_mlm = batch["labels"].to(device)
        labels_cls = batch["labels_cls"].cpu().numpy()

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_mlm,
        )

        mlm_loss = float(out["mlm_loss"].detach().cpu())

        emb = out["embedding"]
        dist = ((emb - center) ** 2).sum(dim=1).detach().cpu().numpy()

        batch_mlm = np.full(
            shape=(input_ids.size(0),),
            fill_value=mlm_loss,
            dtype=np.float32,
        )

        batch_scores = alpha_mlm * batch_mlm + beta_center * dist

        progress.set_postfix(
            mlm_loss=f"{mlm_loss:.4f}",
            mean_score=f"{float(np.mean(batch_scores)):.4f}",
        )

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

    score_df = pd.DataFrame(rows)

    print(
        f"[Scoring completed] samples={len(score_df):,}, "
        f"mean_score={score_df['score'].mean():.4f}, "
        f"min_score={score_df['score'].min():.4f}, "
        f"max_score={score_df['score'].max():.4f}"
    )

    return score_df


def calibrate_threshold(
    scores_normal,
    method: str = "percentile",
    percentile: float = 95.0,
    fixed_threshold: float = 0.5,
):
    print("[Threshold calibration]")
    print(f"method={method}")

    scores_normal = np.asarray(scores_normal, dtype=np.float64)

    if method == "fixed":
        print(f"fixed_threshold={fixed_threshold}")
        return float(fixed_threshold)

    if len(scores_normal) == 0:
        raise ValueError("Cannot calibrate threshold with empty normal scores.")

    if method == "percentile":
        threshold = float(np.percentile(scores_normal, percentile))
        print(f"percentile={percentile}, threshold={threshold:.6f}")
        return threshold

    if method == "mean_std":
        threshold = float(scores_normal.mean() + 3.0 * scores_normal.std())
        print(
            f"mean={scores_normal.mean():.6f}, "
            f"std={scores_normal.std():.6f}, "
            f"threshold={threshold:.6f}"
        )
        return threshold

    raise ValueError(f"Unknown threshold method: {method}")


def evaluate_scores(score_df: pd.DataFrame, threshold: float, normal_label: int = 0):
    print("[Evaluation]")
    print(f"threshold={threshold:.6f}")

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
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "num_samples": int(len(score_df)),
        "num_anomalies": int(y_true.sum()),
        "num_normals": int((y_true == 0).sum()),
        "classification_report": report_dict,
        "classification_report_text": report_text,
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

    print("[Classification Report]")
    print(report_text)

    print("[Confusion Matrix]")
    print(cm)

    return metrics


def save_classification_report_files(
    metrics: Dict[str, Any],
    output_dir,
    prefix: str = "test",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Saving reports] output_dir={output_dir}")

    report_text = metrics.get("classification_report_text", "")
    report_txt_path = output_dir / f"{prefix}_classification_report.txt"
    report_txt_path.write_text(report_text, encoding="utf-8")

    report_dict = metrics.get("classification_report", {})
    if report_dict:
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(output_dir / f"{prefix}_classification_report.csv")

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

    print("[Reports saved]")