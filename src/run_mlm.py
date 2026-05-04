import argparse
import os

from data_loader import load_split_from_dataset_folders
from aggregator import aggregate_by_block, print_sequence_stats
from models import get_tokenizer
from mlm_pretraining import run_mlm_pretraining
from utils import load_config, get_device, set_seed, ensure_dir, clean_name


def get_mlm_output_dir(config, stage):
    train_name = clean_name(config["train_datasets"])
    target_name = clean_name(config.get("prediction_stage", {}).get("predict_datasets", []))

    if stage == "dapt":
        return os.path.join(
            config["output_dir"],
            f"mlm_dapt__train-{train_name}",
        )

    if stage == "tapt":
        return os.path.join(
            config["output_dir"],
            f"mlm_tapt__train-{train_name}__target-{target_name}",
        )

    raise ValueError(f"Unknown MLM stage: {stage}")


def load_mlm_data(config, stage):
    if stage == "dapt":
        datasets = config["mlm"]["dapt"]["datasets"]
        split = "train"

    elif stage == "tapt":
        datasets = config["mlm"]["tapt"]["datasets"]
        split = "train"

    else:
        raise ValueError(f"Unknown MLM stage: {stage}")

    raw_df = load_split_from_dataset_folders(
        config=config,
        dataset_names=datasets,
        split=split,
    )

    columns = config["columns"]
    labels = config["labels"]

    seq_df = aggregate_by_block(
        raw_df,
        timestamp_col=columns["timestamp"],
        template_col=columns["template"],
        block_col=columns["block_id"],
        label_col=columns["label"],
        normal_values=labels["normal_values"],
        anomaly_values=labels["anomaly_values"],
        config=config,
    )

    return seq_df


def main(config_path, stage):
    set_seed(42)

    config = load_config(config_path)
    device = get_device()

    print(f"Using device: {device}")
    print(f"MLM stage: {stage}")

    if not config.get("mlm", {}).get("enabled", False):
        print("MLM is disabled in YAML.")
        return

    if stage == "dapt" and not config["mlm"]["dapt"].get("enabled", False):
        print("DAPT is disabled in YAML.")
        return

    if stage == "tapt" and not config["mlm"]["tapt"].get("enabled", False):
        print("TAPT is disabled in YAML.")
        return

    seq_df = load_mlm_data(config, stage)
    print_sequence_stats(f"MLM {stage.upper()} data", seq_df)

    model_name = config["mlm"]["model_name"]

    if stage == "tapt":
        dapt_dir = get_mlm_output_dir(config, "dapt")
        if os.path.exists(dapt_dir):
            model_name = dapt_dir
            print(f"Using DAPT model as TAPT base: {model_name}")

    tokenizer = get_tokenizer(model_name)
    output_dir = get_mlm_output_dir(config, stage)

    ensure_dir(output_dir)

    run_mlm_pretraining(
        train_dataframe=seq_df,
        tokenizer=tokenizer,
        mlm_config=config["mlm"],
        model_name=model_name,
        max_length=config["model"]["max_length"],
        output_dir=output_dir,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["dapt", "tapt"],
    )

    args = parser.parse_args()

    main(args.config, args.stage)