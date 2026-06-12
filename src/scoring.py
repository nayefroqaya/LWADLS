from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix, )
from tqdm import tqdm


def compute_center_distance(embeddings: torch.Tensor, center: torch.Tensor, ):
    """
    Supports:
        center shape [dim]       -> single center
        center shape [k, dim]    -> multi-prototype centers

    Returns:
        distance per sample
    """

    if center.dim() == 1:
        return ((embeddings - center) ** 2).sum(dim=1)

    if center.dim() == 2:
        dist = torch.cdist(embeddings, center, p=2) ** 2
        return dist.min(dim=1).values

    raise ValueError(f"Unsupported center shape: {tuple(center.shape)}")


@torch.no_grad()
def score_loader(model, loader, center, device, alpha_mlm: float, beta_center: float, desc: str = "Scoring", ):
    """
    Score sequences using:

        score = alpha_mlm * per_sample_mlm_loss
              + beta_center * nearest_center_distance

    If multiple centers are provided, distance is computed to nearest prototype.
    """

    model.eval()

    rows = []
    center = center.to(device)

    progress = tqdm(loader, desc=desc, unit="batch")

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_mlm = batch["labels"].to(device)
        labels_cls = batch["labels_cls"].cpu().numpy()

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_mlm, )

        emb = out["embedding"]

        if out.get("per_sample_mlm_loss") is not None:
            mlm_losses = out["per_sample_mlm_loss"].detach().cpu().numpy()
        else:
            batch_loss = float(out["mlm_loss"].detach().cpu())
            mlm_losses = np.full(shape=(input_ids.size(0),), fill_value=batch_loss, dtype=np.float32, )

        dist = compute_center_distance(embeddings=emb, center=center, ).detach().cpu().numpy()

        batch_scores = alpha_mlm * mlm_losses + beta_center * dist

        progress.set_postfix(mean_mlm=f"{float(np.mean(mlm_losses)):.4f}", mean_dist=f"{float(np.mean(dist)):.4f}",
            mean_score=f"{float(np.mean(batch_scores)):.4f}", )

        for i in range(input_ids.size(0)):
            rows.append(
                {"score": float(batch_scores[i]), "mlm_loss": float(mlm_losses[i]), "center_distance": float(dist[i]),
                    "label": int(labels_cls[i]), "dataset": batch["dataset_names"][i],
                    "group_id": batch["group_ids"][i], "sequence": batch["sequence_texts"][i], })

    score_df = pd.DataFrame(rows)

    print(f"[Scoring completed] samples={len(score_df):,}, "
          f"mean_score={score_df['score'].mean():.6f}, "
          f"min_score={score_df['score'].min():.6f}, "
          f"max_score={score_df['score'].max():.6f}")

    return score_df


def calibrate_threshold(scores_normal, method: str = "percentile", percentile: float = 95.0,
        fixed_threshold: float = 0.5, ):
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
        print(f"mean={scores_normal.mean():.6f}, "
              f"std={scores_normal.std():.6f}, "
              f"threshold={threshold:.6f}")
        return threshold

    raise ValueError(f"Unknown threshold method: {method}")


def evaluate_scores(score_df: pd.DataFrame, threshold: float, normal_label: int = 0, ):
    print("[Evaluation]")
    print(f"threshold={threshold:.6f}")

    y_true = (score_df["label"].to_numpy() != int(normal_label)).astype(int)
    y_pred = (score_df["score"].to_numpy() > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0, )

    acc = accuracy_score(y_true, y_pred)

    report_dict = classification_report(y_true, y_pred, labels=[0, 1], target_names=["normal", "anomaly"],
        zero_division=0, output_dict=True, digits=3,

    )

    report_text = classification_report(y_true, y_pred, labels=[0, 1], target_names=["normal", "anomaly"],
        zero_division=0, output_dict=False, digits=3

    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {"threshold": float(threshold), "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "accuracy": float(acc), "num_samples": int(len(score_df)), "num_anomalies": int(y_true.sum()),
        "num_normals": int((y_true == 0).sum()), "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": {"labels": ["normal", "anomaly"], "matrix": cm.tolist(), "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]), }, }

    try:
        metrics["auc"] = float(roc_auc_score(y_true, score_df["score"].to_numpy()))
    except Exception:
        metrics["auc"] = None

    print("[Classification Report]")
    print(report_text)

    print("[Confusion Matrix]")
    print(cm)

    return metrics


def save_classification_report_files(metrics: Dict[str, Any], output_dir, prefix: str = "test", ):
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
        cm_df = pd.DataFrame(matrix, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels], )
        cm_df.to_csv(output_dir / f"{prefix}_confusion_matrix.csv")

    print("[Reports saved]")


def run_posthoc_grid_search(score_df: pd.DataFrame, cfg: Dict[str, Any], output_dir, normal_label: int = 0, ):
    post_cfg = cfg.get("posthoc_calibration", {})

    if not post_cfg.get("enabled", False):
        print("[Post-hoc calibration] disabled")
        return None, score_df

    print("=" * 80)
    print("[Post-hoc calibration] started")
    print("=" * 80)

    required_cols = ["mlm_loss", "center_distance", "label"]
    missing = [c for c in required_cols if c not in score_df.columns]

    if missing:
        raise ValueError(f"Post-hoc calibration requires columns {required_cols}. "
                         f"Missing: {missing}")

    alpha_values = post_cfg.get("alpha_mlm_values", [0.5, 0.6, 0.7, 0.8, 0.9], )

    percentile_values = post_cfg.get("percentile_values", [85, 87, 88, 89, 90, 91, 92, 93, 95], )

    selection_metric = post_cfg.get("selection_metric", "macro_f1")

    valid_metrics = {"macro_f1", "anomaly_f1", "weighted_f1", "accuracy", "normal_f1", "precision_at_recall", }

    if selection_metric not in valid_metrics:
        raise ValueError(f"Invalid selection_metric={selection_metric}. "
                         f"Use one of {valid_metrics}")

    min_anomaly_recall = float(post_cfg.get("min_anomaly_recall", 0.97))

    y_true = (score_df["label"].to_numpy() != int(normal_label)).astype(int)

    rows = []
    best = None

    for alpha_mlm in tqdm(alpha_values, desc="Post-hoc alpha search", unit="alpha"):
        beta_center = round(1.0 - float(alpha_mlm), 6)

        raw_scores = (float(alpha_mlm) * score_df["mlm_loss"].to_numpy() + beta_center * score_df[
            "center_distance"].to_numpy())

        for percentile in percentile_values:
            threshold = float(np.percentile(raw_scores, percentile))
            y_pred = (raw_scores > threshold).astype(int)

            report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["normal", "anomaly"],
                output_dict=True, zero_division=0, digits=3

            )

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            try:
                auc = float(roc_auc_score(y_true, raw_scores))
            except Exception:
                auc = None

            row = {"alpha_mlm": float(alpha_mlm), "beta_center": float(beta_center), "percentile": float(percentile),
                "threshold": float(threshold), "accuracy": float(accuracy_score(y_true, y_pred)), "auc": auc,
                "normal_precision": float(report["normal"]["precision"]),
                "normal_recall": float(report["normal"]["recall"]), "normal_f1": float(report["normal"]["f1-score"]),
                "anomaly_precision": float(report["anomaly"]["precision"]),
                "anomaly_recall": float(report["anomaly"]["recall"]),
                "anomaly_f1": float(report["anomaly"]["f1-score"]),
                "macro_precision": float(report["macro avg"]["precision"]),
                "macro_recall": float(report["macro avg"]["recall"]),
                "macro_f1": float(report["macro avg"]["f1-score"]),
                "weighted_precision": float(report["weighted avg"]["precision"]),
                "weighted_recall": float(report["weighted avg"]["recall"]),
                "weighted_f1": float(report["weighted avg"]["f1-score"]), "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]), }

            rows.append(row)

            if selection_metric == "precision_at_recall":
                candidate_valid = row["anomaly_recall"] >= min_anomaly_recall

                if candidate_valid:
                    if (best is None or not best.get("valid", False) or row["anomaly_precision"] > best["row"][
                        "anomaly_precision"] or (
                            row["anomaly_precision"] == best["row"]["anomaly_precision"] and row["anomaly_f1"] >
                            best["row"]["anomaly_f1"])):
                        best = {"row": row, "scores": raw_scores, "y_pred": y_pred, "report": report, "cm": cm,
                            "valid": True, }
                else:
                    if best is None:
                        best = {"row": row, "scores": raw_scores, "y_pred": y_pred, "report": report, "cm": cm,
                            "valid": False, }
                    elif (not best.get("valid", False) and row["anomaly_f1"] > best["row"]["anomaly_f1"]):
                        best = {"row": row, "scores": raw_scores, "y_pred": y_pred, "report": report, "cm": cm,
                            "valid": False, }

            else:
                if best is None or row[selection_metric] > best["row"][selection_metric]:
                    best = {"row": row, "scores": raw_scores, "y_pred": y_pred, "report": report, "cm": cm,
                        "valid": True, }

    sort_metric = ("anomaly_precision" if selection_metric == "precision_at_recall" else selection_metric)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(sort_metric, ascending=False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_results_file = post_cfg.get("grid_results_file", "posthoc_grid_search_results.csv", )

    best_report_file = post_cfg.get("best_report_file", "posthoc_best_report.txt", )

    best_predictions_file = post_cfg.get("best_predictions_file", "posthoc_best_predictions.csv", )

    if post_cfg.get("save_grid_results", True):
        results_df.to_csv(output_dir / grid_results_file, index=False)

    best_row = best["row"]

    best_report_text = classification_report(y_true, best["y_pred"], labels=[0, 1], target_names=["normal", "anomaly"],
        zero_division=0, digits=3

    )

    best_output = f"""
Post-hoc calibration best setting
=================================
selection_metric: {selection_metric}
min_anomaly_recall: {min_anomaly_recall}
valid_recall_constraint: {best.get("valid", False)}

alpha_mlm: {best_row["alpha_mlm"]}
beta_center: {best_row["beta_center"]}
percentile: {best_row["percentile"]}
threshold: {best_row["threshold"]}

Overall
=======
accuracy: {best_row["accuracy"]}
auc: {best_row["auc"]}
macro_f1: {best_row["macro_f1"]}
weighted_f1: {best_row["weighted_f1"]}

Normal class
============
precision: {best_row["normal_precision"]}
recall: {best_row["normal_recall"]}
f1: {best_row["normal_f1"]}

Anomaly class
=============
precision: {best_row["anomaly_precision"]}
recall: {best_row["anomaly_recall"]}
f1: {best_row["anomaly_f1"]}

Classification report
=====================
{best_report_text}

Confusion matrix
================
Format:
[[TN, FP],
 [FN, TP]]

{best["cm"]}
"""

    (output_dir / best_report_file).write_text(best_output.strip() + "\n", encoding="utf-8", )

    calibrated_df = score_df.copy()
    calibrated_df["posthoc_score"] = best["scores"]
    calibrated_df["posthoc_prediction"] = best["y_pred"]
    calibrated_df["posthoc_alpha_mlm"] = best_row["alpha_mlm"]
    calibrated_df["posthoc_beta_center"] = best_row["beta_center"]
    calibrated_df["posthoc_percentile"] = best_row["percentile"]
    calibrated_df["posthoc_threshold"] = best_row["threshold"]
    calibrated_df["posthoc_selection_metric"] = selection_metric
    calibrated_df["posthoc_min_anomaly_recall"] = min_anomaly_recall
    calibrated_df["posthoc_valid_recall_constraint"] = best.get("valid", False)

    calibrated_df.to_csv(output_dir / best_predictions_file, index=False)

    print("[Post-hoc calibration] classification report:")
    print(best_report_text)

    print("[Best post-hoc values used]")
    print(f"alpha_mlm   : {best_row['alpha_mlm']}")
    print(f"beta_center : {best_row['beta_center']}")
    print(f"percentile  : {best_row['percentile']}")
    print(f"threshold   : {best_row['threshold']}")
    print(f"selection_metric : {selection_metric}")

    if selection_metric == "precision_at_recall":
        print(f"min_anomaly_recall : {min_anomaly_recall}")
        print(f"valid_recall_constraint : {best.get('valid', False)}")

    print("[Post-hoc calibration] confusion matrix:")
    print(best["cm"])

    print("[Post-hoc calibration] saved:")
    print(output_dir / grid_results_file)
    print(output_dir / best_report_file)
    print(output_dir / best_predictions_file)

    print("=" * 80)
    print("[Post-hoc calibration] completed")
    print("=" * 80)

    return best_row, calibrated_df
