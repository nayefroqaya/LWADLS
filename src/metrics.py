import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    accuracy = accuracy_score(y_true, y_pred)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def print_metrics(title, y_true, y_pred):
    metrics = compute_metrics(y_true, y_pred)

    print(f"\n{title}")
    print("-" * 50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return metrics


def per_dataset_metrics(result_df: pd.DataFrame):
    rows = []

    for dataset_name, group in result_df.groupby("DatasetName"):
        m = compute_metrics(group["label"], group["prediction"])
        m["DatasetName"] = dataset_name
        rows.append(m)

    return pd.DataFrame(rows)