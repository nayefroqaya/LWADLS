import argparse
import torch

from data_loader import load_train_val_test_from_config
from aggregator import aggregate_by_block, print_sequence_stats
from dataset import LogSequenceDataset
from models import get_teacher_model, get_student_model, get_tokenizer
from training import train_classifier, evaluate_classifier
from distillation import train_student_with_distillation
from metrics import print_metrics
from utils import (
    load_config,
    get_device,
    set_seed,
    make_output_dir,
    save_model_and_tokenizer,
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
    )

    val_seq = aggregate_by_block(
        val_raw,
        timestamp_col=columns["timestamp"],
        template_col=columns["template"],
        block_col=columns["block_id"],
        label_col=columns["label"],
        normal_values=labels["normal_values"],
        anomaly_values=labels["anomaly_values"],
    )

    test_seq = aggregate_by_block(
        test_raw,
        timestamp_col=columns["timestamp"],
        template_col=columns["template"],
        block_col=columns["block_id"],
        label_col=columns["label"],
        normal_values=labels["normal_values"],
        anomaly_values=labels["anomaly_values"],
    )

    print_sequence_stats("Train", train_seq)
    print_sequence_stats("Validation", val_seq)
    print_sequence_stats("Test (internal)", test_seq)

    return train_seq, val_seq, test_seq


def main(config_path, mode):
    set_seed(42)

    config = load_config(config_path)
    device = get_device()

    print(f"\nUsing device: {device}")
    print(f"Mode: {mode}")

    train_seq, val_seq, test_seq = prepare_data(config)

    tokenizer = get_tokenizer(config["model"]["teacher_name"])

    train_dataset = LogSequenceDataset(
        train_seq, tokenizer, config["model"]["max_length"]
    )
    val_dataset = LogSequenceDataset(
        val_seq, tokenizer, config["model"]["max_length"]
    )
    test_dataset = LogSequenceDataset(
        test_seq, tokenizer, config["model"]["max_length"]
    )

    output_dir = make_output_dir(config, mode)

    # =========================================================
    # 1. Teacher training
    # =========================================================
    if mode == "teacher":
        print("\nTraining teacher model...")

        model = get_teacher_model(config)

        model = train_classifier(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_config=config["model"],
            output_dir=output_dir,
            device=device,
        )

        print("\nEvaluating teacher on internal test set...")
        y_true, y_pred = evaluate_classifier(
            model, test_dataset, config["model"]["batch_size"], device
        )

        print_metrics("Teacher Test Metrics", y_true, y_pred)

        save_model_and_tokenizer(model, tokenizer, f"{output_dir}/best_model")

    # =========================================================
    # 2. Student without distillation
    # =========================================================
    elif mode == "student":
        print("\nTraining student (no distillation)...")

        model = get_student_model(config)

        model = train_classifier(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_config=config["model"],
            output_dir=output_dir,
            device=device,
        )

        print("\nEvaluating student on internal test set...")
        y_true, y_pred = evaluate_classifier(
            model, test_dataset, config["model"]["batch_size"], device
        )

        print_metrics("Student Test Metrics", y_true, y_pred)

        save_model_and_tokenizer(model, tokenizer, f"{output_dir}/best_model")

    # =========================================================
    # 3. Student with distillation
    # =========================================================
    elif mode == "distill":
        print("\nTraining student with distillation...")

        teacher_dir = make_output_dir(config, "teacher") + "/best_model"

        teacher_model = get_teacher_model(config)
        teacher_model.load_state_dict(
            torch.load(f"{teacher_dir}/pytorch_model.bin", map_location=device)
        )

        student_model = get_student_model(config)

        student_model = train_student_with_distillation(
            student_model=student_model,
            teacher_model=teacher_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            distill_config=config["distillation"],
            output_dir=output_dir,
            device=device,
        )

        print("\nEvaluating distilled student...")
        y_true, y_pred = evaluate_classifier(
            student_model,
            test_dataset,
            config["model"]["batch_size"],
            device,
        )

        print_metrics("Distilled Student Test Metrics", y_true, y_pred)

        save_model_and_tokenizer(student_model, tokenizer, f"{output_dir}/best_model")

    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["teacher", "student", "distill"],
    )

    args = parser.parse_args()

    main(args.config, args.mode)