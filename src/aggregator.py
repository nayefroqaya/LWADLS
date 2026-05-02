import pandas as pd


def normalize_label(value, normal_values, anomaly_values) -> int:
    if value in anomaly_values:
        return 1
    if value in normal_values:
        return 0

    try:
        return int(value)
    except Exception:
        raise ValueError(f"Unknown label value: {value}")


def aggregate_by_block(
    df: pd.DataFrame,
    timestamp_col: str,
    template_col: str,
    block_col: str,
    label_col: str,
    normal_values,
    anomaly_values,
) -> pd.DataFrame:
    df = df.copy()

    df[label_col] = df[label_col].apply(
        lambda x: normalize_label(x, normal_values, anomaly_values)
    )

    df[template_col] = df[template_col].fillna("").astype(str)

    sequences = []

    group_cols = ["datasets", block_col]

    grouped = df.groupby(group_cols, sort=False)

    for (dataset_name, block_id), group in grouped:
        group = group.sort_values(timestamp_col)

        templates = group[template_col].tolist()
        sequence_text = " [SEP] ".join(templates)

        sequence_label = 1 if (group[label_col] == 1).any() else 0

        sequences.append(
            {
                "datasets": dataset_name,
                "Node_block_id": block_id,
                "text": sequence_text,
                "label": sequence_label,
                "sequence_length": len(templates),
            }
        )

    return pd.DataFrame(sequences)


def print_sequence_stats(name: str, df: pd.DataFrame):
    print(f"\n{name} sequence stats")
    print("-" * 50)
    print("Samples:", len(df))
    print("Normal:", int((df["label"] == 0).sum()))
    print("Anomaly:", int((df["label"] == 1).sum()))
    print("Datasets:", df["datasets"].value_counts().to_dict())
    print("Avg sequence length:", round(df["sequence_length"].mean(), 2))