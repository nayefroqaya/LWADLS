import re
import pandas as pd


def normalize_log_text(text: str, config: dict) -> str:
    norm_cfg = config.get("normalization", {})

    if not norm_cfg.get("enabled", False):
        return str(text)

    text = str(text)

    if norm_cfg.get("lowercase", True):
        text = text.lower()

    if norm_cfg.get("replace_ips", True):
        text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>", text)

    if norm_cfg.get("replace_paths", True):
        text = re.sub(r"(/[\w\-.]+)+", "<PATH>", text)

    if norm_cfg.get("replace_hex", True):
        text = re.sub(r"\b0x[a-fA-F0-9]+\b", "<HEX>", text)
        text = re.sub(r"\b[a-fA-F0-9]{8,}\b", "<HEX>", text)

    if norm_cfg.get("replace_block_ids", True):
        text = re.sub(r"blk_[\-\d]+", "<BLOCK>", text)
        text = re.sub(r"block[_\-]?\d+", "<BLOCK>", text)

    if norm_cfg.get("replace_numbers", True):
        text = re.sub(r"\b\d+\b", "<NUM>", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


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
    config=None,
) -> pd.DataFrame:
    df = df.copy()

    df[label_col] = df[label_col].apply(
        lambda x: normalize_label(x, normal_values, anomaly_values)
    )

    df[template_col] = df[template_col].fillna("").astype(str)

    if config is not None:
        df[template_col] = df[template_col].apply(
            lambda x: normalize_log_text(x, config)
        )

    sequences = []

    grouped = df.groupby(["DatasetName", block_col], sort=False)

    for (dataset_name, block_id), group in grouped:
        group = group.sort_values(timestamp_col)

        templates = group[template_col].tolist()
        sequence_text = " [SEP] ".join(templates)

        sequence_label = 1 if (group[label_col] == 1).any() else 0

        sequences.append(
            {
                "DatasetName": dataset_name,
                "Node_block_id": block_id,
                "text": sequence_text,
                "label": sequence_label,
                "sequence_length": len(templates),
            }
        )

    return pd.DataFrame(sequences)


def print_sequence_stats(name: str, df: pd.DataFrame):
    print(f"\n{name} sequence stats")
    print("-" * 60)
    print("Samples:", len(df))
    print("Normal:", int((df["label"] == 0).sum()))
    print("Anomaly:", int((df["label"] == 1).sum()))
    print("Datasets:", df["DatasetName"].value_counts().to_dict())
    print("Avg sequence length:", round(df["sequence_length"].mean(), 2))