import os
import pandas as pd


REQUIRED_COLUMNS = [
    "Timestamp",
    "Date",
    "Time",
    "Content",
    "EventId",
    "EventTemplate",
    "processed_EventTemplate",
    "Node_block_id",
    "Label",
]


def load_split_file(data_dir: str, filename: str, split: str) -> pd.DataFrame:
    path = os.path.join(data_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_pickle(path)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} is missing columns: {missing_cols}")

    df = df[REQUIRED_COLUMNS].copy()
    df["Split"] = split

    if "DatasetName" not in df.columns:
        df["DatasetName"] = "unknown"

    return df


def load_all_splits(config):
    data_dir = config["data_dir"]
    split_files = config["split_files"]

    train_df = load_split_file(data_dir, split_files["train"], "train")
    val_df = load_split_file(data_dir, split_files["val"], "val")
    test_df = load_split_file(data_dir, split_files["test"], "test")

    return train_df, val_df, test_df