import os
import pandas as pd
from pathlib import Path
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

    dataset_path_template = config.get(
        "dataset_path_template",
        "{dataset}"
    )

    dataset_key = f"{split_name}_datasets"
    dataset_names = config.get(dataset_key, [])

    if not dataset_names:
        raise ValueError(f"No datasets defined for '{dataset_key}' in config.")

    split_file = split_files[split_name]
    all_dfs = []

    for dataset_name in dataset_names:
        dataset_folder = dataset_path_template.format(dataset=dataset_name)
        file_path = data_dir / dataset_folder / split_file

        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        df = pd.read_pickle(file_path)

        df["dataset_name"] = dataset_name
        df["split"] = split_name

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def load_all_splits(config):
    train_df = load_split_from_datasets(config, "train")
    val_df = load_split_from_datasets(config, "val")
    test_df = load_split_from_datasets(config, "test")

    return train_df, val_df, test_df