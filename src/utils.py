import os
import yaml
import random
import numpy as np
import torch


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_name(items):
    if isinstance(items, str):
        return items.replace(" ", "")
    return "_".join([str(x).replace(" ", "") for x in items])


def make_experiment_name(config, mode):
    train_name = clean_name(config["train_datasets"])
    val_name = clean_name(config["val_datasets"])
    test_name = clean_name(config["test_datasets"])

    return f"{mode}__train-{train_name}__val-{val_name}__test-{test_name}"


def make_output_dir(config, mode):
    exp_name = make_experiment_name(config, mode)
    return os.path.join(config["output_dir"], exp_name)


def save_model_and_tokenizer(model, tokenizer, output_dir: str):
    ensure_dir(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model to: {output_dir}")