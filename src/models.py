import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import clean_name


def get_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(model_name_or_path)


def get_sequence_classifier(model_name_or_path: str, num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
    )


def get_dapt_dir(config):
    train_name = clean_name(config["train_datasets"])
    return os.path.join(
        config["output_dir"],
        f"mlm_dapt__train-{train_name}",
    )


def get_tapt_dir(config):
    train_name = clean_name(config["train_datasets"])
    target_name = clean_name(config.get("prediction_stage", {}).get("predict_datasets", []))

    return os.path.join(
        config["output_dir"],
        f"mlm_tapt__train-{train_name}__target-{target_name}",
    )


def get_teacher_base_model_name(config):
    if config.get("mlm", {}).get("enabled", False):
        tapt_dir = get_tapt_dir(config)
        dapt_dir = get_dapt_dir(config)

        if config["mlm"].get("tapt", {}).get("enabled", False) and os.path.exists(tapt_dir):
            print(f"Using TAPT model for teacher: {tapt_dir}")
            return tapt_dir

        if config["mlm"].get("dapt", {}).get("enabled", False) and os.path.exists(dapt_dir):
            print(f"Using DAPT model for teacher: {dapt_dir}")
            return dapt_dir

    return config["model"]["teacher_name"]


def get_teacher_model(config):
    model_name = get_teacher_base_model_name(config)

    return get_sequence_classifier(
        model_name_or_path=model_name,
        num_labels=2,
    )


def get_teacher_tokenizer(config):
    model_name = get_teacher_base_model_name(config)
    return get_tokenizer(model_name)


def get_student_model(config):
    model_name = config["model"]["student_name"]

    return get_sequence_classifier(
        model_name_or_path=model_name,
        num_labels=2,
    )