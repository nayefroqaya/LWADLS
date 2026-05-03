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


def build_dataset_split_path(config, dataset_name, split):
    data_dir = config.get("data_dir", config.get("data_root"))
    dataset_path_template = config.get(
        "dataset_path_template",
        "{dataset}/1_{dataset}_Splitted_Datasets",
    )
    split_files = config.get("split_files", config.get("file_names"))

    if data_dir is None:
        raise KeyError("Missing 'data_dir' or 'data_root' in config.")

    if split_files is None:
        raise KeyError("Missing 'split_files' or 'file_names' in config.")

    if split not in split_files:
        raise KeyError(f"Missing split file for '{split}' in config.")

    dataset_subfolder = dataset_path_template.format(dataset=dataset_name)
    filename = split_files[split]

    return os.path.join(data_dir, dataset_subfolder, filename)


def load_split_from_dataset_folders(config, dataset_names, split):
    dfs = []

    for dataset_name in dataset_names:
        path = build_dataset_split_path(config, dataset_name, split)

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_pickle(path)

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{path} missing columns: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )

        df = df[REQUIRED_COLUMNS].copy()

        # Needed so aggregator does not mix same Node_block_id from different datasets.
        df["DatasetName"] = dataset_name

        # Helper/debug columns.
        df["dataset_name"] = dataset_name
        df["Split"] = split
        df["split"] = split

        dfs.append(df)

        print(f"Loaded {dataset_name} {split}: {path} | rows={len(df)}")

    if not dfs:
        raise ValueError(f"No datasets loaded for split={split}")

    combined_df = pd.concat(dfs, ignore_index=True)

    print(
        f"Combined {split}: "
        f"{len(dataset_names)} dataset(s), rows={len(combined_df)}"
    )

    return combined_df


def load_train_val_test_from_config(config):
    train_df = load_split_from_dataset_folders(
        config=config,
        dataset_names=config["train_datasets"],
        split="train",
    )

    val_df = load_split_from_dataset_folders(
        config=config,
        dataset_names=config["val_datasets"],
        split="val",
    )

    test_df = load_split_from_dataset_folders(
        config=config,
        dataset_names=config["test_datasets"],
        split="test",
    )

    return train_df, val_df, test_df


def load_all_splits(config):
    """
    Compatibility function for run_mlm.py and run_experiment.py.

    Your scripts expect:
        from data_loader import load_all_splits

    So this function calls your existing loader.
    """
    return load_train_val_test_from_config(config)