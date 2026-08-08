from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
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


# ======================================================
# COMMON HELPERS
# ======================================================

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


def get_center_config(cfg):
    center_cfg = cfg.get("normal_center", {})

    return {
        "num_prototypes": int(center_cfg.get("num_prototypes", 1)),
        "prototype_method": center_cfg.get("prototype_method", "kmeans"),
        "seed": int(cfg["experiment"].get("seed", 42)),
    }


def load_saved_model_tokenizer_center(
    cfg,
    out_dir: Path,
    device,
    center_filename: str,
):
    model_path = out_dir / "model.pt"
    center_path = out_dir / center_filename
    tokenizer_path = out_dir / "tokenizer"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}\n"
            "Run stage: train first."
        )

    if not center_path.exists():
        raise FileNotFoundError(
            f"Saved center/prototypes not found: {center_path}\n"
            "Run stage: train first."
        )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Saved tokenizer not found: {tokenizer_path}\n"
            "Run stage: train first."
        )

    print(f"[Loading model] {model_path}")
    print(f"[Loading center/prototypes] {center_path}")
    print(f"[Loading tokenizer] {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = LogSLMNC(
        backbone_name=cfg["model"]["student_name"],
        projection_dim=cfg["model"]["projection_dim"],
        dropout=cfg["model"]["dropout"],
        freeze_backbone=cfg["model"].get("freeze_backbone", False),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    center = torch.load(center_path, map_location="cpu")

    return model, tokenizer, center

def apply_fixed_posthoc_parameters(score_df: pd.DataFrame, posthoc_best: dict) -> pd.DataFrame:
    """
    Apply already-selected post-hoc parameters to a score dataframe.

    IMPORTANT:
    This function does NOT search or optimize anything. It simply applies the
    alpha/beta/threshold values that were selected earlier on the validation set.
    Therefore it is safe to use on the held-out test set.
    """
    if posthoc_best is None:
        return score_df.copy()

    alpha_mlm = float(posthoc_best["alpha_mlm"])
    beta_center = float(posthoc_best["beta_center"])
    threshold = float(posthoc_best["threshold"])

    calibrated_df = score_df.copy()

    # Preserve the original score produced with hybrid_scoring values from config.
    calibrated_df["base_score"] = calibrated_df["score"]

    # Apply the FIXED parameters selected on validation data.
    calibrated_df["score"] = (
        alpha_mlm * calibrated_df["mlm_loss"].to_numpy()
        + beta_center * calibrated_df["center_distance"].to_numpy()
    )
    calibrated_df["prediction"] = (calibrated_df["score"] > threshold).astype(int)

    calibrated_df["posthoc_alpha_mlm"] = alpha_mlm
    calibrated_df["posthoc_beta_center"] = beta_center
    calibrated_df["posthoc_percentile"] = float(posthoc_best["percentile"])
    calibrated_df["posthoc_threshold"] = threshold

    return calibrated_df


# ======================================================
# MODE 1: IN-DOMAIN TRAIN ONLY
# ======================================================

def train_in_domain(cfg, config_path: str):
    """
    In-domain training stage only.

    Trains on normal sequences from one dataset and saves:
        model.pt
        normal_center.pt
        tokenizer/
        train_results.json

    It does NOT predict.
    """

    out_dir = ensure_output_dir(cfg)

    if cfg.get("outputs", {}).get("save_config", True):
        save_resolved_config(config_path, out_dir)

    set_seed(cfg["experiment"].get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    mode_cfg = cfg["datasets"]["in_domain"]
    dataset_name = mode_cfg["dataset_name"]

    print("=" * 80)
    print("[Stage] TRAIN ONLY")
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

    print(
        f"[Sequences] train={len(train_df)} "
        f"val={len(val_df)}"
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

    center_cfg = get_center_config(cfg)

    center = compute_center(
        model,
        val_loader,
        device,
        desc="Computing in-domain normal center/prototypes",
        num_prototypes=center_cfg["num_prototypes"],
        prototype_method=center_cfg["prototype_method"],
        seed=center_cfg["seed"],
    )

    print("[Saving trained model]")
    torch.save(model.state_dict(), out_dir / "model.pt")

    print("[Saving normal center/prototypes]")
    torch.save(center.cpu(), out_dir / "normal_center.pt")

    print("[Saving tokenizer]")
    tokenizer.save_pretrained(out_dir / "tokenizer")

    save_json(
        {
            "stage": "train",
            "mode": "in_domain",
            "dataset": dataset_name,
            "history": history,
            "normal_center": center_cfg,
            "saved_model": str(out_dir / "model.pt"),
            "saved_center": str(out_dir / "normal_center.pt"),
            "saved_tokenizer": str(out_dir / "tokenizer"),
        },
        out_dir / "train_results.json",
    )

    print("=" * 80)
    print("[IN-DOMAIN TRAINING FINISHED]")
    print(f"Saved model     : {out_dir / 'model.pt'}")
    print(f"Saved center    : {out_dir / 'normal_center.pt'}")
    print(f"Saved tokenizer : {out_dir / 'tokenizer'}")
    print("=" * 80)


# ======================================================
# MODE 1: IN-DOMAIN PREDICT ONLY
# ======================================================

def predict_in_domain(cfg, config_path: str):
    """
    In-domain prediction stage only.

    Loads:
        model.pt
        normal_center.pt
        tokenizer/

    Then:
        1) uses NORMAL-ONLY validation data for the original threshold calibration
        2) uses FULL validation data (normal + anomaly) for post-hoc parameter search
        3) freezes the selected post-hoc parameters
        4) applies the frozen parameters to the test set for final evaluation

    It does NOT retrain.
    """

    out_dir = ensure_output_dir(cfg)

    set_seed(cfg["experiment"].get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    mode_cfg = cfg["datasets"]["in_domain"]
    dataset_name = mode_cfg["dataset_name"]

    print("=" * 80)
    print("[Stage] PREDICT ONLY")
    print("[Mode] in_domain")
    print(f"[Dataset] {dataset_name}")
    print("=" * 80)

    model, tokenizer, center = load_saved_model_tokenizer_center(
        cfg=cfg,
        out_dir=out_dir,
        device=device,
        center_filename="normal_center.pt",
    )

    # ------------------------------------------------------------------
    # VALIDATION VIEW 1: NORMAL ONLY
    # Used only for normality-based threshold calibration.
    # This preserves the original behavior of the method.
    # ------------------------------------------------------------------
    val_normal_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["val_split"],
        normal_only=True,
    )

    # ------------------------------------------------------------------
    # VALIDATION VIEW 2: FULL VALIDATION SET (NORMAL + ANOMALY)
    # Used only for supervised post-hoc parameter selection.
    # It comes from the SAME validation split, but anomalies are not removed.
    # ------------------------------------------------------------------
    val_full_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["val_split"],
        normal_only=False,
    )

    # Held-out test set: used only after all calibration parameters are fixed.
    test_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["test_split"],
        normal_only=mode_cfg.get("test_normal_only", False),
    )

    print(
        f"[Sequences] val_normal={len(val_normal_df)} "
        f"val_full={len(val_full_df)} "
        f"test={len(test_df)}"
    )

    val_normal_loader = make_loader(
        cfg,
        val_normal_df,
        tokenizer,
        shuffle=False,
    )

    val_full_loader = make_loader(
        cfg,
        val_full_df,
        tokenizer,
        shuffle=False,
    )

    test_loader = make_loader(
        cfg,
        test_df,
        tokenizer,
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # ORIGINAL THRESHOLD CALIBRATION: NORMAL VALIDATION ONLY
    # ------------------------------------------------------------------
    val_normal_scores = score_loader(
        model,
        val_normal_loader,
        center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Scoring in-domain NORMAL validation data",
    )

    threshold = calibrate_threshold(
        val_normal_scores["score"],
        method=cfg["threshold"]["method"],
        percentile=cfg["threshold"].get("percentile", 95),
        fixed_threshold=cfg["prediction_stage"].get("anomaly_threshold", 0.5),
    )

    # ------------------------------------------------------------------
    # POST-HOC CALIBRATION: FULL VALIDATION SET
    # score_loader stores mlm_loss, center_distance, and ground-truth label.
    # run_posthoc_grid_search uses these VALIDATION labels to select the best
    # alpha/beta/percentile/threshold.
    # ------------------------------------------------------------------
    val_calibration_scores = score_loader(
        model,
        val_full_loader,
        center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Scoring FULL validation data for post-hoc calibration",
    )

    # Save validation calibration scores so stage=posthoc_only can also use
    # validation data instead of test data.
    validation_calibration_file = out_dir / "validation_calibration_scores.csv"
    val_calibration_scores.to_csv(validation_calibration_file, index=False)
    print(f"[Validation calibration scores saved] {validation_calibration_file}")

    # Macro-F1/label-based post-hoc selection requires both classes.
    if val_calibration_scores["label"].nunique() < 2:
        raise ValueError(
            "Full validation data must contain both normal and anomaly samples "
            "for label-based post-hoc calibration."
        )

    # ------------------------------------------------------------------
    # PROBLEM IN THE OLD CODE -- KEPT AS COMMENT FOR AUDITABILITY:
    # The old implementation tuned post-hoc parameters on TEST labels:
    #
    # posthoc_best, _ = run_posthoc_grid_search(
    #     score_df=test_scores,
    #     cfg=cfg,
    #     output_dir=out_dir,
    #     normal_label=0,
    # )
    #
    # This caused test-label information leakage.
    # ------------------------------------------------------------------

    # NEW: select post-hoc parameters exclusively on FULL VALIDATION data.
    posthoc_best, _ = run_posthoc_grid_search(
        score_df=val_calibration_scores,
        cfg=cfg,
        output_dir=out_dir,
        normal_label=0,
    )

    # ------------------------------------------------------------------
    # TEST: NO PARAMETER SEARCH HERE
    # First keep the original/base evaluation for comparison.
    # ------------------------------------------------------------------
    test_scores = score_loader(
        model,
        test_loader,
        center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Predicting in-domain held-out TEST data",
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

    print("[Saving base test predictions]")
    test_scores["prediction"] = (test_scores["score"] > threshold).astype(int)
    test_scores.to_csv(
        out_dir / cfg["outputs"].get("predictions_file", "predictions.csv"),
        index=False,
    )
    print("[Base test predictions saved]")

    # Apply the FROZEN validation-selected post-hoc parameters to test.
    # Test labels are used only by evaluate_scores AFTER parameter selection.
    posthoc_test_metrics = None
    if posthoc_best is not None:
        posthoc_test_scores = apply_fixed_posthoc_parameters(test_scores, posthoc_best)

        posthoc_test_metrics = evaluate_scores(
            posthoc_test_scores,
            float(posthoc_best["threshold"]),
            normal_label=0,
        )

        save_classification_report_files(
            posthoc_test_metrics,
            out_dir,
            prefix="posthoc_test",
        )

        posthoc_test_predictions_file = out_dir / "posthoc_test_predictions.csv"
        posthoc_test_scores.to_csv(posthoc_test_predictions_file, index=False)
        print(f"[Post-hoc TEST predictions saved] {posthoc_test_predictions_file}")

    save_json(
        {
            "stage": "predict",
            "mode": "in_domain",
            "dataset": dataset_name,
            "metrics": metrics,
            # Keep the old key for backward compatibility.
            "posthoc_best": posthoc_best,
            "posthoc_best_validation": posthoc_best,
            "posthoc_test_metrics": posthoc_test_metrics,
        },
        out_dir / cfg["outputs"].get("results_file", "results.json"),
    )

    print("=" * 80)
    print("[IN-DOMAIN PREDICTION FINISHED]")
    print("[Base test Classification Report]")
    print(metrics["classification_report_text"])

    if posthoc_best is not None:
        print("[Post-hoc best selected on VALIDATION]")
        print(posthoc_best)
        print("[Final post-hoc TEST Classification Report]")
        print(posthoc_test_metrics["classification_report_text"])

    print("=" * 80)


# ======================================================
# MODE 1: IN-DOMAIN FULL PIPELINE
# ======================================================

def run_in_domain(cfg, config_path: str):
    """
    Full in-domain pipeline:
        train
        predict
    """

    train_in_domain(cfg, config_path)
    predict_in_domain(cfg, config_path)


# ======================================================
# MODE 2: FEW-SHOT TARGET ADAPTATION TRAIN ONLY
# ======================================================

def train_fewshot_target_adaptation(cfg, config_path: str):
    """
    Few-shot target adaptation training stage only.

    Trains:
        source normal model on source_datasets
        target adaptation on target normal subset

    Saves:
        model.pt
        target_normal_center.pt
        tokenizer/
        train_results.json

    It does NOT predict.
    """

    out_dir = ensure_output_dir(cfg)

    if cfg.get("outputs", {}).get("save_config", True):
        save_resolved_config(config_path, out_dir)

    set_seed(cfg["experiment"].get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    mode_cfg = cfg["datasets"]["fewshot_target_adaptation"]
    source_datasets = mode_cfg["source_datasets"]
    target_dataset = mode_cfg["target_dataset"]

    print("=" * 80)
    print("[Stage] TRAIN ONLY")
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

    print(
        f"[Sequences] source_train={len(source_train_df)} "
        f"target_adapt={len(target_adapt_df)} "
        f"target_val={len(target_val_df)}"
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

    center_cfg = get_center_config(cfg)

    target_center = compute_center(
        model,
        target_val_loader,
        device,
        desc="Computing target normal center/prototypes",
        num_prototypes=center_cfg["num_prototypes"],
        prototype_method=center_cfg["prototype_method"],
        seed=center_cfg["seed"],
    )

    print("[Saving trained model]")
    torch.save(model.state_dict(), out_dir / "model.pt")

    print("[Saving target normal center/prototypes]")
    torch.save(target_center.cpu(), out_dir / "target_normal_center.pt")

    print("[Saving tokenizer]")
    tokenizer.save_pretrained(out_dir / "tokenizer")

    save_json(
        {
            "stage": "train",
            "mode": "fewshot_target_adaptation",
            "source_history": source_history,
            "target_history": target_history,
            "source_datasets": source_datasets,
            "target_dataset": target_dataset,
            "target_adapt_size": int(len(target_adapt_df)),
            "normal_center": center_cfg,
            "saved_model": str(out_dir / "model.pt"),
            "saved_center": str(out_dir / "target_normal_center.pt"),
            "saved_tokenizer": str(out_dir / "tokenizer"),
        },
        out_dir / "train_results.json",
    )

    print("=" * 80)
    print("[FEW-SHOT TARGET ADAPTATION TRAINING FINISHED]")
    print(f"Saved model     : {out_dir / 'model.pt'}")
    print(f"Saved center    : {out_dir / 'target_normal_center.pt'}")
    print(f"Saved tokenizer : {out_dir / 'tokenizer'}")
    print("=" * 80)


# ======================================================
# MODE 2: FEW-SHOT TARGET ADAPTATION PREDICT ONLY
# ======================================================

def predict_fewshot_target_adaptation(cfg, config_path: str):
    """
    Few-shot target adaptation prediction stage only.

    Loads:
        model.pt
        target_normal_center.pt
        tokenizer/

    Then:
        1) uses NORMAL-ONLY target validation for the original threshold
        2) uses FULL target validation (normal + anomaly) for post-hoc search
        3) freezes the selected post-hoc parameters
        4) applies them once to the held-out target test set

    It does NOT retrain.
    """

    out_dir = ensure_output_dir(cfg)

    set_seed(cfg["experiment"].get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    mode_cfg = cfg["datasets"]["fewshot_target_adaptation"]
    target_dataset = mode_cfg["target_dataset"]

    print("=" * 80)
    print("[Stage] PREDICT ONLY")
    print("[Mode] fewshot_target_adaptation")
    print(f"[Target dataset] {target_dataset}")
    print("=" * 80)

    model, tokenizer, target_center = load_saved_model_tokenizer_center(
        cfg=cfg,
        out_dir=out_dir,
        device=device,
        center_filename="target_normal_center.pt",
    )

    # Validation view 1: normal only, for original normality threshold.
    target_val_normal_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_val_split"],
        normal_only=True,
    )

    # Validation view 2: full target validation, for post-hoc parameter search.
    target_val_full_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_val_split"],
        normal_only=False,
    )

    target_test_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_test_split"],
        normal_only=mode_cfg.get("target_test_normal_only", False),
    )

    print(
        f"[Sequences] target_val_normal={len(target_val_normal_df)} "
        f"target_val_full={len(target_val_full_df)} "
        f"target_test={len(target_test_df)}"
    )

    target_val_normal_loader = make_loader(
        cfg,
        target_val_normal_df,
        tokenizer,
        shuffle=False,
    )

    target_val_full_loader = make_loader(
        cfg,
        target_val_full_df,
        tokenizer,
        shuffle=False,
    )

    target_test_loader = make_loader(
        cfg,
        target_test_df,
        tokenizer,
        shuffle=False,
    )

    # Original normal-only threshold calibration.
    val_normal_scores = score_loader(
        model,
        target_val_normal_loader,
        target_center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Scoring target NORMAL validation data",
    )

    threshold = calibrate_threshold(
        val_normal_scores["score"],
        method=cfg["threshold"]["method"],
        percentile=cfg["threshold"].get("percentile", 95),
        fixed_threshold=cfg["prediction_stage"].get("anomaly_threshold", 0.5),
    )

    # Full validation scores used for post-hoc parameter selection.
    val_calibration_scores = score_loader(
        model,
        target_val_full_loader,
        target_center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Scoring FULL target validation data for post-hoc calibration",
    )

    validation_calibration_file = out_dir / "validation_calibration_scores.csv"
    val_calibration_scores.to_csv(validation_calibration_file, index=False)
    print(f"[Validation calibration scores saved] {validation_calibration_file}")

    # Macro-F1/label-based post-hoc selection requires both classes.
    if val_calibration_scores["label"].nunique() < 2:
        raise ValueError(
            "Full validation data must contain both normal and anomaly samples "
            "for label-based post-hoc calibration."
        )

    # ------------------------------------------------------------------
    # PROBLEM IN THE OLD CODE -- KEPT AS COMMENT FOR AUDITABILITY:
    #
    # posthoc_best, _ = run_posthoc_grid_search(
    #     score_df=test_scores,
    #     cfg=cfg,
    #     output_dir=out_dir,
    #     normal_label=0,
    # )
    #
    # This selected hyperparameters using TEST labels.
    # ------------------------------------------------------------------

    # NEW: tune only on full target validation data.
    posthoc_best, _ = run_posthoc_grid_search(
        score_df=val_calibration_scores,
        cfg=cfg,
        output_dir=out_dir,
        normal_label=0,
    )

    # Held-out test set: no search/optimization.
    test_scores = score_loader(
        model,
        target_test_loader,
        target_center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Predicting held-out target TEST data",
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

    print("[Saving base test predictions]")
    test_scores["prediction"] = (test_scores["score"] > threshold).astype(int)
    test_scores.to_csv(
        out_dir / cfg["outputs"].get("predictions_file", "predictions.csv"),
        index=False,
    )
    print("[Base test predictions saved]")

    posthoc_test_metrics = None
    if posthoc_best is not None:
        posthoc_test_scores = apply_fixed_posthoc_parameters(test_scores, posthoc_best)

        posthoc_test_metrics = evaluate_scores(
            posthoc_test_scores,
            float(posthoc_best["threshold"]),
            normal_label=0,
        )

        save_classification_report_files(
            posthoc_test_metrics,
            out_dir,
            prefix="posthoc_test",
        )

        posthoc_test_predictions_file = out_dir / "posthoc_test_predictions.csv"
        posthoc_test_scores.to_csv(posthoc_test_predictions_file, index=False)
        print(f"[Post-hoc TEST predictions saved] {posthoc_test_predictions_file}")

    save_json(
        {
            "stage": "predict",
            "mode": "fewshot_target_adaptation",
            "metrics": metrics,
            # Keep the old key for backward compatibility.
            "posthoc_best": posthoc_best,
            "posthoc_best_validation": posthoc_best,
            "posthoc_test_metrics": posthoc_test_metrics,
            "target_dataset": target_dataset,
        },
        out_dir / cfg["outputs"].get("results_file", "results.json"),
    )

    print("=" * 80)
    print("[FEW-SHOT TARGET ADAPTATION PREDICTION FINISHED]")
    print("[Base test Classification Report]")
    print(metrics["classification_report_text"])

    if posthoc_best is not None:
        print("[Post-hoc best selected on VALIDATION]")
        print(posthoc_best)
        print("[Final post-hoc TEST Classification Report]")
        print(posthoc_test_metrics["classification_report_text"])

    print("=" * 80)


# ======================================================
# MODE 2: FEW-SHOT TARGET ADAPTATION FULL PIPELINE
# ======================================================

def run_fewshot_target_adaptation(cfg, config_path: str):
    """
    Full few-shot target adaptation pipeline:
        train
        predict
    """

    train_fewshot_target_adaptation(cfg, config_path)
    predict_fewshot_target_adaptation(cfg, config_path)


# ======================================================
# SHARED POSTHOC ONLY
# ======================================================

def run_posthoc_only(cfg):
    """
    Post-hoc calibration only.

    IMPORTANT:
    The grid search is performed on validation_calibration_scores.csv, which is
    created by the predict stage from the FULL validation split.

    predictions.csv contains held-out test scores and is NEVER used for
    post-hoc parameter search. After parameters are selected on validation, they
    are applied once to predictions.csv for final test evaluation.
    """

    out_dir = ensure_output_dir(cfg)

    validation_calibration_file = out_dir / "validation_calibration_scores.csv"
    predictions_file = out_dir / cfg["outputs"].get(
        "predictions_file",
        "predictions.csv",
    )

    if not validation_calibration_file.exists():
        raise FileNotFoundError(
            f"Validation calibration file not found: {validation_calibration_file}\n"
            "Run stage: predict first with the revised code so full validation scores are saved."
        )

    print("=" * 80)
    print("[Stage] POSTHOC ONLY")
    print(f"[Loading VALIDATION calibration scores] {validation_calibration_file}")
    print("=" * 80)

    validation_score_df = pd.read_csv(validation_calibration_file)

    # ------------------------------------------------------------------
    # PROBLEM IN THE OLD CODE -- KEPT AS COMMENT FOR AUDITABILITY:
    # The old code loaded predictions.csv (TEST data) and optimized on it:
    #
    # score_df = pd.read_csv(predictions_file)
    # posthoc_best, _ = run_posthoc_grid_search(
    #     score_df=score_df,
    #     cfg=cfg,
    #     output_dir=out_dir,
    #     normal_label=0,
    # )
    # ------------------------------------------------------------------

    # NEW: optimize only on the FULL VALIDATION scores.
    posthoc_best, _ = run_posthoc_grid_search(
        score_df=validation_score_df,
        cfg=cfg,
        output_dir=out_dir,
        normal_label=0,
    )

    posthoc_test_metrics = None

    # If test predictions are available, apply the already-fixed parameters
    # to test. No test label is used to choose alpha/beta/threshold.
    if posthoc_best is not None and predictions_file.exists():
        print(f"[Loading held-out TEST scores] {predictions_file}")
        test_score_df = pd.read_csv(predictions_file)
        posthoc_test_scores = apply_fixed_posthoc_parameters(test_score_df, posthoc_best)

        posthoc_test_metrics = evaluate_scores(
            posthoc_test_scores,
            float(posthoc_best["threshold"]),
            normal_label=0,
        )

        save_classification_report_files(
            posthoc_test_metrics,
            out_dir,
            prefix="posthoc_test",
        )

        posthoc_test_predictions_file = out_dir / "posthoc_test_predictions.csv"
        posthoc_test_scores.to_csv(posthoc_test_predictions_file, index=False)
        print(f"[Post-hoc TEST predictions saved] {posthoc_test_predictions_file}")

    save_json(
        {
            "stage": "posthoc_only",
            "mode": cfg["experiment"].get("mode"),
            # Keep the old key for backward compatibility.
            "posthoc_best": posthoc_best,
            "posthoc_best_validation": posthoc_best,
            "posthoc_test_metrics": posthoc_test_metrics,
        },
        out_dir / "posthoc_only_results.json",
    )

    print("=" * 80)
    print("[POSTHOC FINISHED]")
    print("[Post-hoc best selected on VALIDATION]")
    print(posthoc_best)
    if posthoc_test_metrics is not None:
        print("[Final post-hoc TEST Classification Report]")
        print(posthoc_test_metrics["classification_report_text"])
    print("=" * 80)


# ======================================================
# TERMINAL ENTRY POINT
# ======================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = cfg["experiment"]["mode"]
    stage = cfg["experiment"].get("stage", "train_predict")

    if stage == "posthoc_only":
        run_posthoc_only(cfg)
        return

    if mode == "in_domain":
        if stage == "train":
            train_in_domain(cfg, args.config)

        elif stage == "predict":
            predict_in_domain(cfg, args.config)

        elif stage == "train_predict":
            run_in_domain(cfg, args.config)

        else:
            raise ValueError(
                f"Unsupported stage for in_domain: {stage}. "
                "Use 'train', 'predict', 'posthoc_only', or 'train_predict'."
            )

    elif mode == "fewshot_target_adaptation":
        if stage == "train":
            train_fewshot_target_adaptation(cfg, args.config)

        elif stage == "predict":
            predict_fewshot_target_adaptation(cfg, args.config)

        elif stage == "train_predict":
            run_fewshot_target_adaptation(cfg, args.config)

        else:
            raise ValueError(
                f"Unsupported stage for fewshot_target_adaptation: {stage}. "
                "Use 'train', 'predict', 'posthoc_only', or 'train_predict'."
            )

    else:
        raise ValueError(
            f"Unsupported mode: {mode}. "
            "Use 'in_domain' or 'fewshot_target_adaptation'."
        )


if __name__ == "__main__":
    main()