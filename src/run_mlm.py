import argparse
import os

from data_loader import load_all_splits
from aggregator import aggregate_by_block, print_sequence_stats
from models import get_tokenizer
from mlm_pretraining import run_mlm_pretraining
from utils import load_config, get_device, set_seed, make_output_dir


def main(config_path):
    set_seed(42)

    config = load_config(config_path)
    device = get_device()

    print(f"Using device: {device}")

    train_raw, _, _ = load_all_splits(config)

    columns = config["columns"]
    labels = config["labels"]

    train_seq = aggregate_by_block(
        train_raw,
        timestamp_col=columns["timestamp"],
        template_col=columns["template"],
        block_col=columns["block_id"],
        label_col=columns["label"],
        normal_values=labels["normal_values"],
        anomaly_values=labels["anomaly_values"],
    )

    print_sequence_stats("MLM train", train_seq)

    model_name = config["mlm"]["model_name"]
    tokenizer = get_tokenizer(model_name)

    output_dir = make_output_dir(config, "mlm_teacher")

    run_mlm_pretraining(
        train_dataframe=train_seq,
        tokenizer=tokenizer,
        mlm_config=config["mlm"],
        model_name=model_name,
        max_length=config["model"]["max_length"],
        output_dir=output_dir,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    args = parser.parse_args()

    main(args.config)