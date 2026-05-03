from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(model_name_or_path)


def get_sequence_classifier(model_name_or_path: str, num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
    )


# =========================
# Teacher model
# =========================
def get_teacher_model(config):
    model_name = config["model"]["teacher_name"]

    return get_sequence_classifier(
        model_name_or_path=model_name,
        num_labels=2,
    )


# =========================
# Student model
# =========================
def get_student_model(config):
    model_name = config["model"]["student_name"]

    return get_sequence_classifier(
        model_name_or_path=model_name,
        num_labels=2,
    )