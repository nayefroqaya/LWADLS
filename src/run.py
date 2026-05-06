from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from .config import load_config, ensure_output_dir, save_resolved_config
from .utils import set_seed, get_device, save_json
from .data import (
    load_sequences_for_dataset,
    concat_dataset_splits,
    build_loader_from_sequences,
)
from .model import LogSLMNC
from .train import train_normality
from .losses import compute_center
from .scoring import (
    score_loader,
    calibrate_threshold,
    evaluate_scores,
    save_classification_report_files,
    run_posthoc_grid_search,
)


def make_model_and_tokenizer(cfg, device):
    model_cfg = cfg["model"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["student_name"])

    model = LogSLMNC(
        backbone_name=model_cfg["student_name"],
        projection_dim=model_cfg["projection_dim"],
        dropout=model_cfg["dropout"],
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    ).to(device)

    return model, tokenizer


def make_loader(cfg, seq_df, tokenizer, shuffle: bool, balanced: bool = False):
    return build_loader_from_sequences(
        seq_df=seq_df,
        tokenizer=tokenizer,
        max_length=cfg["model"]["max_length"],
        batch_size=cfg["training"]["batch_size"],
        mlm_probability=cfg["mlm"]["mlm_probability"],
        shuffle=shuffle,
        use_dataset_balanced_sampler=balanced,
    )


def run_in_domain(cfg, config_path: str):
    out_dir = ensure_output_dir(cfg)

    if cfg.get("outputs", {}).get("save_config", True):
        save_resolved_config(config_path, out_dir)

    set_seed(cfg["experiment"].get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    mode_cfg = cfg["datasets"]["in_domain"]
    dataset_name = mode_cfg["dataset_name"]

    print("=" * 80)
    print("[Mode] in_domain")
    print(f"[Dataset] {dataset_name}")
    print("=" * 80)

    train_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["train_split"],
        normal_only=mode_cfg.get("train_normal_only", True),
    )

    val_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["val_split"],
        normal_only=mode_cfg.get("val_normal_only", True),
    )

    test_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["test_split"],
        normal_only=mode_cfg.get("test_normal_only", False),
    )

    print(
        f"[Sequences] train={len(train_df)} "
        f"val={len(val_df)} "
        f"test={len(test_df)}"
    )

    model, tokenizer = make_model_and_tokenizer(cfg, device)

    train_loader = make_loader(
        cfg,
        train_df,
        tokenizer,
        shuffle=True,
        balanced=cfg["training"].get("use_dataset_balanced_sampler", False),
    )

    val_loader = make_loader(
        cfg,
        val_df,
        tokenizer,
        shuffle=False,
    )

    test_loader = make_loader(
        cfg,
        test_df,
        tokenizer,
        shuffle=False,
    )

    history = train_normality(
        model=model,
        loader=train_loader,
        device=device,
        epochs=cfg["training"]["in_domain_epochs"],
        learning_rate=cfg["training"]["in_domain_lr"],
        weight_decay=cfg["training"]["weight_decay"],
        center_loss_weight=cfg["training"]["center_loss_weight"],
        gradient_clip_norm=cfg["training"].get("gradient_clip_norm"),
        stage_name="in-domain",
    )

    center = compute_center(
        model,
        val_loader,
        device,
        desc="Computing in-domain normal center",
    )

    val_scores = score_loader(
        model,
        val_loader,
        center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Scoring validation normal data",
    )

    threshold = calibrate_threshold(
        val_scores["score"],
        method=cfg["threshold"]["method"],
        percentile=cfg["threshold"].get("percentile", 95),
        fixed_threshold=cfg["prediction_stage"].get("anomaly_threshold", 0.5),
    )

    test_scores = score_loader(
        model,
        test_loader,
        center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Predicting test data",
    )

    metrics = evaluate_scores(
        test_scores,
        threshold,
        normal_label=0,
    )

    save_classification_report_files(
        metrics,
        out_dir,
        prefix="test",
    )

    if cfg.get("outputs", {}).get("save_predictions", True) or cfg.get(
        "evaluation", {}
    ).get("save_predictions", True):
        print("[Saving predictions]")
        test_scores["prediction"] = (test_scores["score"] > threshold).astype(int)
        test_scores.to_csv(
            out_dir / cfg["outputs"].get("predictions_file", "predictions.csv"),
            index=False,
        )
        print("[Predictions saved]")

    # Post-hoc calibration / grid search without retraining.
    posthoc_best, posthoc_predictions = run_posthoc_grid_search(
        score_df=test_scores,
        cfg=cfg,
        output_dir=out_dir,
        normal_label=0,
    )

    if cfg.get("outputs", {}).get("save_model", True):
        print("[Saving model]")
        torch.save(model.state_dict(), out_dir / "model.pt")

    if cfg.get("outputs", {}).get("save_normal_center", True):
        print("[Saving normal center]")
        torch.save(center.cpu(), out_dir / "normal_center.pt")

    if cfg.get("outputs", {}).get("save_tokenizer", True):
        print("[Saving tokenizer]")
        tokenizer.save_pretrained(out_dir / "tokenizer")

    save_json(
        {
            "mode": "in_domain",
            "history": history,
            "metrics": metrics,
            "posthoc_best": posthoc_best,
        },
        out_dir / cfg["outputs"].get("results_file", "results.json"),
    )

    print("=" * 80)
    print("[Classification Report]")
    print(metrics["classification_report_text"])
    print("[Metrics]", metrics)

    if posthoc_best is not None:
        print("[Post-hoc best]")
        print(posthoc_best)

    print("=" * 80)


def run_fewshot_target_adaptation(cfg, config_path: str):
    out_dir = ensure_output_dir(cfg)

    if cfg.get("outputs", {}).get("save_config", True):
        save_resolved_config(config_path, out_dir)

    set_seed(cfg["experiment"].get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    mode_cfg = cfg["datasets"]["fewshot_target_adaptation"]
    source_datasets = mode_cfg["source_datasets"]
    target_dataset = mode_cfg["target_dataset"]

    print("=" * 80)
    print("[Mode] fewshot_target_adaptation")
    print(f"[Source datasets] {source_datasets}")
    print(f"[Target dataset] {target_dataset}")
    print("=" * 80)

    source_train_df = concat_dataset_splits(
        cfg,
        datasets=source_datasets,
        split=mode_cfg["source_train_split"],
        normal_only=True,
    )

    target_adapt_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_adapt_split"],
        normal_only=mode_cfg.get("target_adapt_normal_only", True),
        target_normal_ratio=mode_cfg.get("target_normal_ratio"),
        target_normal_max_samples=mode_cfg.get("target_normal_max_samples"),
    )

    target_val_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_val_split"],
        normal_only=mode_cfg.get("target_val_normal_only", True),
    )

    target_test_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_test_split"],
        normal_only=mode_cfg.get("target_test_normal_only", False),
    )

    print(
        f"[Sequences] source_train={len(source_train_df)} "
        f"target_adapt={len(target_adapt_df)} "
        f"target_val={len(target_val_df)} "
        f"target_test={len(target_test_df)}"
    )

    model, tokenizer = make_model_and_tokenizer(cfg, device)

    source_loader = make_loader(
        cfg,
        source_train_df,
        tokenizer,
        shuffle=True,
        balanced=cfg["training"].get("balance_source_datasets", True),
    )

    target_adapt_loader = make_loader(
        cfg,
        target_adapt_df,
        tokenizer,
        shuffle=True,
    )

    target_val_loader = make_loader(
        cfg,
        target_val_df,
        tokenizer,
        shuffle=False,
    )

    target_test_loader = make_loader(
        cfg,
        target_test_df,
        tokenizer,
        shuffle=False,
    )

    source_history = train_normality(
        model=model,
        loader=source_loader,
        device=device,
        epochs=cfg["training"]["source_epochs"],
        learning_rate=cfg["training"]["source_lr"],
        weight_decay=cfg["training"]["weight_decay"],
        center_loss_weight=cfg["training"]["center_loss_weight"],
        gradient_clip_norm=cfg["training"].get("gradient_clip_norm"),
        stage_name="source",
    )

    target_history = train_normality(
        model=model,
        loader=target_adapt_loader,
        device=device,
        epochs=cfg["training"]["target_adapt_epochs"],
        learning_rate=cfg["training"]["target_adapt_lr"],
        weight_decay=cfg["training"]["weight_decay"],
        center_loss_weight=cfg["training"]["center_loss_weight"],
        gradient_clip_norm=cfg["training"].get("gradient_clip_norm"),
        stage_name="target-adapt",
    )

    target_center = compute_center(
        model,
        target_val_loader,
        device,
        desc="Computing target normal center",
    )

    val_scores = score_loader(
        model,
        target_val_loader,
        target_center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Scoring target validation normal data",
    )

    threshold = calibrate_threshold(
        val_scores["score"],
        method=cfg["threshold"]["method"],
        percentile=cfg["threshold"].get("percentile", 95),
        fixed_threshold=cfg["prediction_stage"].get("anomaly_threshold", 0.5),
    )

    test_scores = score_loader(
        model,
        target_test_loader,
        target_center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Predicting target test data",
    )

    metrics = evaluate_scores(
        test_scores,
        threshold,
        normal_label=0,
    )

    save_classification_report_files(
        metrics,
        out_dir,
        prefix="test",
    )

    if cfg.get("outputs", {}).get("save_predictions", True) or cfg.get(
        "evaluation", {}
    ).get("save_predictions", True):
        print("[Saving predictions]")
        test_scores["prediction"] = (test_scores["score"] > threshold).astype(int)
        test_scores.to_csv(
            out_dir / cfg["outputs"].get("predictions_file", "predictions.csv"),
            index=False,
        )
        print("[Predictions saved]")

    # Post-hoc calibration / grid search without retraining.
    posthoc_best, posthoc_predictions = run_posthoc_grid_search(
        score_df=test_scores,
        cfg=cfg,
        output_dir=out_dir,
        normal_label=0,
    )

    if cfg.get("outputs", {}).get("save_model", True):
        print("[Saving model]")
        torch.save(model.state_dict(), out_dir / "model.pt")

    if cfg.get("outputs", {}).get("save_normal_center", True):
        print("[Saving target normal center]")
        torch.save(target_center.cpu(), out_dir / "target_normal_center.pt")

    if cfg.get("outputs", {}).get("save_tokenizer", True):
        print("[Saving tokenizer]")
        tokenizer.save_pretrained(out_dir / "tokenizer")

    save_json(
        {
            "mode": "fewshot_target_adaptation",
            "source_history": source_history,
            "target_history": target_history,
            "metrics": metrics,
            "posthoc_best": posthoc_best,
            "source_datasets": source_datasets,
            "target_dataset": target_dataset,
            "target_adapt_size": int(len(target_adapt_df)),
        },
        out_dir / cfg["outputs"].get("results_file", "results.json"),
    )

    print("=" * 80)
    print("[Classification Report]")
    print(metrics["classification_report_text"])
    print("[Metrics]", metrics)

    if posthoc_best is not None:
        print("[Post-hoc best]")
        print(posthoc_best)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = cfg["experiment"]["mode"]

    if mode == "in_domain":
        run_in_domain(cfg, args.config)
    elif mode == "fewshot_target_adaptation":
        run_fewshot_target_adaptation(cfg, args.config)
    else:
        raise ValueError(
            f"Unsupported mode: {mode}. "
            "Use 'in_domain' or 'fewshot_target_adaptation'."
        )


if __name__ == "__main__":
    main()