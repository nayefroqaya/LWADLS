import argparse
from transformers import AutoModelForSequenceClassification

from data_loader import load_train_val_test_from_config
from aggregator import aggregate_by_block, print_sequence_stats
from dataset import LogSequenceDataset
from models import get_student_model, get_student_tokenizer
from trainer import train_classifier, evaluate_classifier
from utils import (
    load_config,
    get_device,
    set_seed,
    make_output_dir,
)


def prepare_data(config):
    print("\nLoading datasets...")

    train_raw, val_raw, test_raw = load_train_val_test_from_config(config)

    columns = config["columns"]
    labels = config["labels"]

    print("\nAggregating logs into sequences...")

    train_seq = aggregate_by_block(
        train_raw,
        timestamp_col=columns["timestamp"],
        template_col=columns["template"],
        block_col=columns["block_id"],
        label_col=columns["label"],
        normal_values=labels["normal_values"],
        anomaly_values=labels["anomaly_values"],
        config=config,
    )

    val_seq = aggregate_by_block(
        val_raw,
        timestamp_col=columns["timestamp"],
        template_col=columns["template"],
        block_col=columns["block_id"],
        label_col=columns["label"],
        normal_values=labels["normal_values"],
        anomaly_values=labels["anomaly_values"],
        config=config,
    )

    test_seq = aggregate_by_block(
        test_raw,
        timestamp_col=columns["timestamp"],
        template_col=columns["template"],
        block_col=columns["block_id"],
        label_col=columns["label"],
        normal_values=labels["normal_values"],
        anomaly_values=labels["anomaly_values"],
        config=config,
    )

    print_sequence_stats("Train", train_seq)
    print_sequence_stats("Validation", val_seq)
    print_sequence_stats("Test (internal)", test_seq)

    return train_seq, val_seq, test_seq


def build_datasets(train_seq, val_seq, test_seq, tokenizer, max_length):
    train_dataset = LogSequenceDataset(train_seq, tokenizer, max_length)
    val_dataset = LogSequenceDataset(val_seq, tokenizer, max_length)
    test_dataset = LogSequenceDataset(test_seq, tokenizer, max_length)

    return train_dataset, val_dataset, test_dataset


def build_training_config(config):
    training_config = config["model"].copy()
    training_config["training"] = config.get("training", {})
    return training_config


def main(config_path):
    set_seed(42)

    config = load_config(config_path)
    device = get_device()

    print(f"\nUsing device: {device}")
    print("Mode: student-only")

    train_seq, val_seq, test_seq = prepare_data(config)

    output_dir = make_output_dir(config, "student")
    training_config = build_training_config(config)

    print("\nTraining student model...")

    tokenizer = get_student_tokenizer(config)

    train_dataset, val_dataset, test_dataset = build_datasets(
        train_seq=train_seq,
        val_seq=val_seq,
        test_seq=test_seq,
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
    )

    model = get_student_model(config)

    best_path = train_classifier(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
        output_dir=output_dir,
        device=device,
    )

    best_model = AutoModelForSequenceClassification.from_pretrained(best_path)

    print("\nEvaluating student on internal test set...")
    evaluate_classifier(
        model=best_model,
        dataset=test_dataset,
        batch_size=config["model"]["batch_size"],
        device=device,
        title="Student internal test",
        print_output=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML file.",
    )

    args = parser.parse_args()

    main(args.config)