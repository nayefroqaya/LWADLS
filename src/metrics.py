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


def get_classification_report_df(y_true, y_pred):
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Class 0 - Normal", "Class 1 - Anomaly"],
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report_dict).transpose()


def print_metrics(title, y_true, y_pred):
    metrics = compute_metrics(y_true, y_pred)

    print(f"\n{title}")
    print("-" * 60)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Class 0 - Normal", "Class 1 - Anomaly"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    return metrics


def per_dataset_metrics(result_df: pd.DataFrame):
    rows = []

    for dataset_name, group in result_df.groupby("DatasetName"):
        metrics = compute_metrics(group["label"], group["prediction"])
        metrics["DatasetName"] = dataset_name
        rows.append(metrics)

    return pd.DataFrame(rows)


def per_dataset_classification_reports(result_df: pd.DataFrame):
    reports = {}

    for dataset_name, group in result_df.groupby("DatasetName"):
        reports[dataset_name] = get_classification_report_df(
            group["label"],
            group["prediction"],
        )

    return reports