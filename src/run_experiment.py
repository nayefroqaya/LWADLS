import os
import argparse
from transformers import AutoModelForSequenceClassification

from data_loader import load_all_splits
from aggregator import aggregate_by_block, print_sequence_stats
from dataset import LogSequenceDataset
from models import get_tokenizer, get_sequence_classifier
from trainer import train_classifier, evaluate_with_dataframe
from distillation import train_distilled_student
from utils import (
    load_config,
    get_device,
    set_seed,
    ensure_dir,
    make_output_dir,
)


def prepare_data(config):
    train_raw, val_raw, test_raw = load_all_splits(config)

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
    print_sequence_stats("Test", test_seq)

    return train_seq, val_seq, test_seq


def build_datasets(train_seq, val_seq, test_seq, tokenizer, max_length):
    train_dataset = LogSequenceDataset(train_seq, tokenizer, max_length)
    val_dataset = LogSequenceDataset(val_seq, tokenizer, max_length)
    test_dataset = LogSequenceDataset(test_seq, tokenizer, max_length)

    return train_dataset, val_dataset, test_dataset


def run_teacher(config, device):
    output_dir = make_output_dir(config, "teacher")
    ensure_dir(output_dir)

    train_seq, val_seq, test_seq = prepare_data(config)

    model_name = config["model"]["teacher_name"]

    mlm_dir = make_output_dir(config, "mlm_teacher")
    if config.get("mlm", {}).get("enabled", False) and os.path.exists(mlm_dir):
        print(f"Using MLM-pretrained model: {mlm_dir}")
        model_name = mlm_dir

    tokenizer = get_tokenizer(model_name)

    train_dataset, val_dataset, test_dataset = build_datasets(
        train_seq,
        val_seq,
        test_seq,
        tokenizer,
        config["model"]["max_length"],
    )

    model = get_sequence_classifier(model_name)

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

    evaluate_with_dataframe(
        model=best_model,
        dataframe=test_seq,
        dataset=test_dataset,
        batch_size=config["model"]["batch_size"],
        device=device,
        output_dir=os.path.join(output_dir, "test_results"),
        title="Teacher test evaluation",
    )


def run_student(config, device):
    output_dir = make_output_dir(config, "student_no_distill")
    ensure_dir(output_dir)

    train_seq, val_seq, test_seq = prepare_data(config)

    model_name = config["model"]["student_name"]
    tokenizer = get_tokenizer(model_name)

    train_dataset, val_dataset, test_dataset = build_datasets(
        train_seq,
        val_seq,
        test_seq,
        tokenizer,
        config["model"]["max_length"],
    )

    model = get_sequence_classifier(model_name)

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

    evaluate_with_dataframe(
        model=best_model,
        dataframe=test_seq,
        dataset=test_dataset,
        batch_size=config["model"]["batch_size"],
        device=device,
        output_dir=os.path.join(output_dir, "test_results"),
        title="Student without distillation test evaluation",
    )


def run_distillation(config, device):
    output_dir = make_output_dir(config, "student_distilled")
    ensure_dir(output_dir)

    teacher_dir = make_output_dir(config, "teacher")
    teacher_path = os.path.join(teacher_dir, "best_model")

    if not os.path.exists(teacher_path):
        raise FileNotFoundError(
            f"Trained teacher not found: {teacher_path}. "
            f"Run teacher first."
        )

    train_seq, val_seq, test_seq = prepare_data(config)

    student_name = config["model"]["student_name"]
    tokenizer = get_tokenizer(student_name)

    train_dataset, val_dataset, test_dataset = build_datasets(
        train_seq,
        val_seq,
        test_seq,
        tokenizer,
        config["model"]["max_length"],
    )

    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path)
    student = get_sequence_classifier(student_name)

    best_path = train_distilled_student(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_config=config["model"],
        distill_config=config["distillation"],
        output_dir=output_dir,
        device=device,
    )

    best_model = AutoModelForSequenceClassification.from_pretrained(best_path)

    evaluate_with_dataframe(
        model=best_model,
        dataframe=test_seq,
        dataset=test_dataset,
        batch_size=config["model"]["batch_size"],
        device=device,
        output_dir=os.path.join(output_dir, "test_results"),
        title="Distilled student test evaluation",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["teacher", "student", "distill"],
    )

    args = parser.parse_args()

    set_seed(42)

    config = load_config(args.config)
    device = get_device()

    print(f"Using device: {device}")
    print(f"Running mode: {args.mode}")

    if args.mode == "teacher":
        run_teacher(config, device)

    elif args.mode == "student":
        run_student(config, device)

    elif args.mode == "distill":
        run_distillation(config, device)


if __name__ == "__main__":
    main()