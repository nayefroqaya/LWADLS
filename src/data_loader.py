import os
import pandas as pd
from pathlib import Path
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

def load_split_from_datasets(config, split_name):
    data_dir = Path(config.get("data_dir", config.get("data_root")))
    split_files = config.get("split_files", config.get("file_names"))

    dataset_key = f"{split_name}_datasets"
    dataset_names = config.get(dataset_key, [])

    if not dataset_names:
        raise ValueError(f"No datasets defined for '{dataset_key}' in config.")

    split_file = split_files[split_name]

    all_dfs = []

    for dataset_name in dataset_names:
        file_path = data_dir / dataset_name / split_file

        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        if file_path.suffix == ".pkl":
            df = pd.read_pickle(file_path)
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix == ".json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        df["dataset_name"] = dataset_name
        df["split"] = split_name

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def load_all_splits(config):
    train_df = load_split_from_datasets(config, "train")
    val_df = load_split_from_datasets(config, "val")
    test_df = load_split_from_datasets(config, "test")

    return train_df, val_df, test_df