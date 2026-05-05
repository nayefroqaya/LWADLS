import argparse
import os

from data_loader import load_split_from_dataset_folders
from aggregator import aggregate_by_block, print_sequence_stats
from models import get_tokenizer
from mlm_pretraining import run_mlm_pretraining
from utils import load_config, get_device, set_seed, ensure_dir, clean_name


# --------------------------------------------------
# Resolve MLM base model (AUTO → student model)
# --------------------------------------------------
def get_mlm_base_model_name(config):
    model_name = config.get("mlm", {}).get("model_name", "auto")

    if model_name is None or str(model_name).lower() == "auto":
        model_name = config["model"]["student_name"]

    return model_name


# --------------------------------------------------
# Output directory
# --------------------------------------------------
def get_mlm_output_dir(config, stage):
    train_name = clean_name(config["train_datasets"])
    target_name = clean_name(
        config.get("prediction_stage", {}).get("predict_datasets", [])
    )

    model_name = clean_name(get_mlm_base_model_name(config))

    if stage == "dapt":
        return os.path.join(
            config["output_dir"],
            f"mlm_dapt__model-{model_name}__train-{train_name}",
        )

    if stage == "tapt":
        return os.path.join(
            config["output_dir"],
            f"mlm_tapt__model-{model_name}__train-{train_name}__target-{target_name}",
        )

    raise ValueError(f"Unknown MLM stage: {stage}")


# --------------------------------------------------
# Load MLM data
# --------------------------------------------------
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


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main(config_path, stage):
    set_seed(42)

    config = load_config(config_path)
    device = get_device()

    print("=" * 80)
    print("MLM PRETRAINING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Stage: {stage}")

    if not config.get("mlm", {}).get("enabled", False):
        print("❌ MLM is disabled in YAML.")
        return

    if stage == "dapt" and not config["mlm"].get("dapt", {}).get("enabled", False):
        print("❌ DAPT is disabled.")
        return

    if stage == "tapt" and not config["mlm"].get("tapt", {}).get("enabled", False):
        print("❌ TAPT is disabled.")
        return

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    seq_df = load_mlm_data(config, stage)
    print_sequence_stats(f"MLM {stage.upper()} data", seq_df)

    # --------------------------------------------------
    # Resolve model
    # --------------------------------------------------
    model_name = get_mlm_base_model_name(config)

    # TAPT uses DAPT model if exists
    if stage == "tapt":
        dapt_dir = get_mlm_output_dir(config, "dapt")

        if os.path.exists(dapt_dir):
            print(f"✔ Using DAPT model for TAPT: {dapt_dir}")
            model_name = dapt_dir
        else:
            print(f"⚠ DAPT model not found → using base model: {model_name}")

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------
    tokenizer = get_tokenizer(model_name)

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    output_dir = get_mlm_output_dir(config, stage)
    ensure_dir(output_dir)

    print("\nMLM configuration:")
    print("-" * 60)
    print(f"Base model: {model_name}")
    print(f"Output dir: {output_dir}")
    print(f"Datasets:   {len(seq_df)} samples")

    # --------------------------------------------------
    # Run MLM
    # --------------------------------------------------
    run_mlm_pretraining(
        train_dataframe=seq_df,
        tokenizer=tokenizer,
        mlm_config=config["mlm"],
        model_name=model_name,
        max_length=config["model"]["max_length"],
        output_dir=output_dir,
        device=device,
    )

    print("\n✔ MLM completed successfully.")


# --------------------------------------------------
# CLI
# --------------------------------------------------
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