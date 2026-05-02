import os
import pandas as pd
from pathlib import Path
import pandas as pd
from pathlib import Path
import pandas as pd

from pathlib import Path
import pandas as pd


def _resolve_path(path_str):
    """
    Convert a string path to a Path object.
    Supports relative and absolute paths.
    """
    path = Path(path_str).expanduser()
    return path


def _load_file(file_path):
    """
    Load dataset file based on extension.
    Supports .pkl, .pickle, .csv, .json, .jsonl, .parquet.
    """
    suffix = file_path.suffix.lower()

    if suffix in [".pkl", ".pickle"]:
        return pd.read_pickle(file_path)

    if suffix == ".csv":
        return pd.read_csv(file_path)

    if suffix == ".json":
        return pd.read_json(file_path)

    if suffix == ".jsonl":
        return pd.read_json(file_path, lines=True)

    if suffix == ".parquet":
        return pd.read_parquet(file_path)

    raise ValueError(f"Unsupported file format: {file_path}")


def _validate_required_columns(df, config, file_path):
    """
    Check that required columns exist in the dataframe.
    """
    columns_cfg = config.get("columns", {})

    required_columns = [
        columns_cfg.get("timestamp", "Timestamp"),
        columns_cfg.get("template", "processed_EventTemplate"),
        columns_cfg.get("block_id", "Node_block_id"),
        columns_cfg.get("label", "Label"),
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required columns in {file_path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def load_split_from_datasets(config, split_name):
    """
    Load one split, for example train/val/test, from one or more datasets.

    Example YAML:

    data_dir: "../datasets"
    dataset_path_template: "{dataset}/1_{dataset}_Splitted_Datasets"

    train_datasets: ["BGL", "HDFS"]
    val_datasets: ["BGL", "HDFS"]
    test_datasets: ["BGL"]

    split_files:
      train: "train.pkl"
      val: "val.pkl"
      test: "test.pkl"
    """

    data_dir_value = config.get("data_dir", config.get("data_root"))

    if data_dir_value is None:
        raise KeyError(
            "Missing data directory in config. Add one of these:\n"
            "data_dir: ../datasets\n"
            "or\n"
            "data_root: ../datasets"
        )

    data_dir = _resolve_path(data_dir_value)

    split_files = config.get("split_files", config.get("file_names"))

    if split_files is None:
        raise KeyError(
            "Missing split file names in config. Add one of these:\n"
            "split_files:\n"
            "  train: train.pkl\n"
            "  val: val.pkl\n"
            "  test: test.pkl"
        )

    if split_name not in split_files:
        raise KeyError(
            f"Missing file name for split '{split_name}' in split_files/file_names."
        )

    dataset_key = f"{split_name}_datasets"
    dataset_names = config.get(dataset_key, [])

    if not dataset_names:
        raise ValueError(f"No datasets defined for '{dataset_key}' in config.")

    dataset_path_template = config.get("dataset_path_template", "{dataset}")

    split_file = split_files[split_name]
    all_dfs = []

    dataset_col = config.get("columns", {}).get("dataset", "DatasetName")

    for dataset_name in dataset_names:
        dataset_folder = dataset_path_template.format(dataset=dataset_name)
        file_path = data_dir / dataset_folder / split_file

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}\n"
                f"Check data_dir, dataset_path_template, and split_files in config."
            )

        df = _load_file(file_path)

        _validate_required_columns(df, config, file_path)

        # Important for multi-dataset runs.
        # This prevents same block IDs from different datasets being grouped together.
        df[dataset_col] = dataset_name

        # Extra helper columns. These are useful for debugging and analysis.
        df["dataset_name"] = dataset_name
        df["split"] = split_name

        all_dfs.append(df)

        print(f"Loaded {split_name}: {dataset_name} -> {file_path} | rows={len(df)}")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    print(
        f"Combined {split_name}: "
        f"{len(dataset_names)} dataset(s), rows={len(combined_df)}"
    )

    return combined_df


def load_all_splits(config):
    """
    Load train, val, and test splits.
    Each split can contain one or more datasets.
    """

    train_df = load_split_from_datasets(config, "train")
    val_df = load_split_from_datasets(config, "val")
    test_df = load_split_from_datasets(config, "test")

    return train_df, val_df, test_df