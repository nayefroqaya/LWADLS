from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)


# ======================================================
# CONFIG
# ======================================================

# Update this path if your output folder is different.
PREDICTIONS_FILE = Path(
    "../outputs/AdaLogSLM_BGL_HDFS_to_TH_1G/predictions.csv"
)

OUTPUT_FILE = Path(
    "../outputs/AdaLogSLM_BGL_HDFS_to_TH_1G/grid_search_results.csv"
)

BEST_REPORT_FILE = Path(
    "../outputs/AdaLogSLM_BGL_HDFS_to_TH_1G/best_grid_search_report.txt"
)

BEST_PREDICTIONS_FILE = Path(
    "../outputs/AdaLogSLM_BGL_HDFS_to_TH_1G/best_grid_search_predictions.csv"
)

NORMAL_LABEL = 0

# ======================================================
# GRID SEARCH SPACE
# ======================================================

# alpha_mlm + beta_center should sum to 1.0.
ALPHA_VALUES = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

# Percentile threshold values.
PERCENTILES = [
    80,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
]

# Choose how to select the best setting.
#
# Recommended if you want balanced performance:
#   "macro_f1"
#
# Recommended if anomaly is most important:
#   "anomaly_f1"
#
# Other options:
#   "accuracy"
#   "weighted_f1"
#   "normal_f1"
SELECTION_METRIC = "macro_f1"


# ======================================================
# FUNCTIONS
# ======================================================

def compute_scores(
    df: pd.DataFrame,
    alpha_mlm: float,
    beta_center: float,
) -> np.ndarray:
    """
    Recompute anomaly score from saved prediction columns.

    score = alpha_mlm * mlm_loss + beta_center * center_distance
    """

    return (
        alpha_mlm * df["mlm_loss"].to_numpy()
        + beta_center * df["center_distance"].to_numpy()
    )


def evaluate_setting(
    df: pd.DataFrame,
    alpha_mlm: float,
    beta_center: float,
    percentile: float,
):
    y_true = (df["label"].to_numpy() != NORMAL_LABEL).astype(int)

    scores = compute_scores(
        df=df,
        alpha_mlm=alpha_mlm,
        beta_center=beta_center,
    )

    threshold = float(np.percentile(scores, percentile))

    y_pred = (scores > threshold).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["normal", "anomaly"],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    try:
        auc = float(roc_auc_score(y_true, scores))
    except Exception:
        auc = None

    row = {
        "alpha_mlm": float(alpha_mlm),
        "beta_center": float(beta_center),
        "percentile": float(percentile),
        "threshold": float(threshold),

        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": auc,

        "normal_precision": float(report["normal"]["precision"]),
        "normal_recall": float(report["normal"]["recall"]),
        "normal_f1": float(report["normal"]["f1-score"]),
        "normal_support": int(report["normal"]["support"]),

        "anomaly_precision": float(report["anomaly"]["precision"]),
        "anomaly_recall": float(report["anomaly"]["recall"]),
        "anomaly_f1": float(report["anomaly"]["f1-score"]),
        "anomaly_support": int(report["anomaly"]["support"]),

        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),

        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),

        # Confusion matrix:
        # [[TN, FP],
        #  [FN, TP]]
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    return row, report, cm, scores, threshold, y_pred


def print_top_results(results_df: pd.DataFrame, top_k: int = 20):
    cols = [
        "alpha_mlm",
        "beta_center",
        "percentile",
        "accuracy",
        "normal_f1",
        "anomaly_precision",
        "anomaly_recall",
        "anomaly_f1",
        "macro_f1",
        "weighted_f1",
        "fp",
        "fn",
    ]

    print("\nTop settings:")
    print("-" * 100)
    print(
        results_df[cols]
        .head(top_k)
        .to_string(index=False)
    )


def save_best_predictions(
    df: pd.DataFrame,
    best_scores: np.ndarray,
    best_y_pred: np.ndarray,
    best_threshold: float,
    best_alpha: float,
    best_beta: float,
    best_percentile: float,
):
    output_df = df.copy()
    output_df["grid_score"] = best_scores
    output_df["grid_prediction"] = best_y_pred
    output_df["grid_threshold"] = best_threshold
    output_df["grid_alpha_mlm"] = best_alpha
    output_df["grid_beta_center"] = best_beta
    output_df["grid_percentile"] = best_percentile

    BEST_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(BEST_PREDICTIONS_FILE, index=False)


# ======================================================
# MAIN
# ======================================================

def main():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_cols = [
        "mlm_loss",
        "center_distance",
        "label",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in predictions.csv: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    print("=" * 100)
    print("Grid search on saved predictions")
    print(f"Input file       : {PREDICTIONS_FILE}")
    print(f"Rows             : {len(df):,}")
    print(f"Selection metric : {SELECTION_METRIC}")
    print("=" * 100)

    rows = []
    best = None

    for alpha in ALPHA_VALUES:
        beta = round(1.0 - alpha, 4)

        for percentile in PERCENTILES:
            row, report, cm, scores, threshold, y_pred = evaluate_setting(
                df=df,
                alpha_mlm=alpha,
                beta_center=beta,
                percentile=percentile,
            )

            rows.append(row)

            if best is None or row[SELECTION_METRIC] > best["row"][SELECTION_METRIC]:
                best = {
                    "row": row,
                    "report": report,
                    "cm": cm,
                    "scores": scores,
                    "threshold": threshold,
                    "y_pred": y_pred,
                }

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        SELECTION_METRIC,
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    best_row = best["row"]

    print("\nBest setting:")
    print("=" * 100)
    for key, value in best_row.items():
        print(f"{key}: {value}")

    print_top_results(results_df, top_k=20)

    y_true = (df["label"].to_numpy() != NORMAL_LABEL).astype(int)

    report_text = classification_report(
        y_true,
        best["y_pred"],
        labels=[0, 1],
        target_names=["normal", "anomaly"],
        zero_division=0,
    )

    cm_text = str(best["cm"])

    report_output = f"""
Best grid-search setting
========================
selection_metric: {SELECTION_METRIC}

alpha_mlm: {best_row["alpha_mlm"]}
beta_center: {best_row["beta_center"]}
percentile: {best_row["percentile"]}
threshold: {best_row["threshold"]}

Overall metrics
===============
accuracy: {best_row["accuracy"]}
auc: {best_row["auc"]}
macro_f1: {best_row["macro_f1"]}
weighted_f1: {best_row["weighted_f1"]}

Normal class
============
precision: {best_row["normal_precision"]}
recall: {best_row["normal_recall"]}
f1: {best_row["normal_f1"]}
support: {best_row["normal_support"]}

Anomaly class
=============
precision: {best_row["anomaly_precision"]}
recall: {best_row["anomaly_recall"]}
f1: {best_row["anomaly_f1"]}
support: {best_row["anomaly_support"]}

Classification report
=====================
{report_text}

Confusion matrix
================
Format:
[[TN, FP],
 [FN, TP]]

{cm_text}
"""

    BEST_REPORT_FILE.write_text(
        report_output.strip() + "\n",
        encoding="utf-8",
    )

    save_best_predictions(
        df=df,
        best_scores=best["scores"],
        best_y_pred=best["y_pred"],
        best_threshold=best["threshold"],
        best_alpha=best_row["alpha_mlm"],
        best_beta=best_row["beta_center"],
        best_percentile=best_row["percentile"],
    )

    print("\nBest classification report:")
    print("=" * 100)
    print(report_text)

    print("\nBest confusion matrix:")
    print("=" * 100)
    print(best["cm"])

    print("\nSaved files:")
    print("=" * 100)
    print(f"Grid results     : {OUTPUT_FILE}")
    print(f"Best report      : {BEST_REPORT_FILE}")
    print(f"Best predictions : {BEST_PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()