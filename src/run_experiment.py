import argparse
from transformers import AutoModelForSequenceClassification

from data_loader import load_train_val_test_from_config
from aggregator import aggregate_by_block, print_sequence_stats
from dataset import LogSequenceDataset
from models import get_teacher_model, get_student_model, get_tokenizer
from trainer import train_classifier, evaluate_classifier
from distillation import train_distilled_student
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


def build_datasets(train_seq, val_seq, test_seq, tokenizer, max_length):
    train_dataset = LogSequenceDataset(train_seq, tokenizer, max_length)
    val_dataset = LogSequenceDataset(val_seq, tokenizer, max_length)
    test_dataset = LogSequenceDataset(test_seq, tokenizer, max_length)

    return train_dataset, val_dataset, test_dataset


def main(config_path, mode):
    set_seed(42)

    config = load_config(config_path)
    device = get_device()

    print(f"\nUsing device: {device}")
    print(f"Mode: {mode}")

    train_seq, val_seq, test_seq = prepare_data(config)

    output_dir = make_output_dir(config, mode)

    if mode == "teacher":
        print("\nTraining teacher model...")

        teacher_model_name = config["model"]["teacher_name"]
        tokenizer = get_tokenizer(teacher_model_name)

        train_dataset, val_dataset, test_dataset = build_datasets(
            train_seq=train_seq,
            val_seq=val_seq,
            test_seq=test_seq,
            tokenizer=tokenizer,
            max_length=config["model"]["max_length"],
        )

        model = get_teacher_model(config)

        best_path = train_classifier(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config["model"],
            output_dir=output_dir,
            device=device,
        )

        best_model = AutoModelForSequenceClassification.from_pretrained(best_path)

        print("\nEvaluating teacher on internal test set...")
        evaluate_classifier(
            model=best_model,
            dataset=test_dataset,
            batch_size=config["model"]["batch_size"],
            device=device,
            title="Teacher internal test",
            print_output=True,
        )

    elif mode == "student":
        print("\nTraining student without distillation...")

        student_model_name = config["model"]["student_name"]
        tokenizer = get_tokenizer(student_model_name)

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
            config=config["model"],
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

    elif mode == "distill":
        print("\nTraining student with distillation...")

        teacher_dir = make_output_dir(config, "teacher")
        teacher_path = f"{teacher_dir}/best_model"

        print(f"Loading trained teacher from: {teacher_path}")

        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            teacher_path
        )

        student_model_name = config["model"]["student_name"]
        tokenizer = get_tokenizer(student_model_name)

        train_dataset, val_dataset, test_dataset = build_datasets(
            train_seq=train_seq,
            val_seq=val_seq,
            test_seq=test_seq,
            tokenizer=tokenizer,
            max_length=config["model"]["max_length"],
        )

        student_model = get_student_model(config)

        best_path = train_distilled_student(
            teacher=teacher_model,
            student=student_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_config=config["model"],
            distill_config=config["distillation"],
            output_dir=output_dir,
            device=device,
        )

        best_model = AutoModelForSequenceClassification.from_pretrained(best_path)

        print("\nEvaluating distilled student on internal test set...")
        evaluate_classifier(
            model=best_model,
            dataset=test_dataset,
            batch_size=config["model"]["batch_size"],
            device=device,
            title="Distilled student internal test",
            print_output=True,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML file.",
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["teacher", "student", "distill"],
        help="Training mode.",
    )

    args = parser.parse_args()

    main(args.config, args.mode)