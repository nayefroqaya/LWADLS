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
    data_dir = config["data_dir"]
    dataset_path_template = config["dataset_path_template"]
    split_files = config["split_files"]

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
            raise ValueError(f"{path} missing columns: {missing_cols}")

        df = df[REQUIRED_COLUMNS].copy()
        df["DatasetName"] = dataset_name
        df["Split"] = split

        dfs.append(df)

        print(f"Loaded {dataset_name} {split}: {path} | rows={len(df)}")

    if not dfs:
        raise ValueError(f"No datasets loaded for split={split}")

    return pd.concat(dfs, ignore_index=True)


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


def get_prediction_datasets_from_config(config):
    prediction_stage = config.get("prediction_stage", {})

    predict_split = prediction_stage.get("predict_split", "test")
    predict_datasets = prediction_stage.get("predict_datasets", config["test_datasets"])

    return predict_split, predict_datasets