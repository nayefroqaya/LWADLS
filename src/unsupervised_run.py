from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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

def calibrate_conformal_normal_only(normal_scores, alpha: float = 0.01) -> dict:
    """
    Calibrate an anomaly threshold using NORMAL validation scores only.

    This is a split-conformal style upper-tail calibration for anomaly scores,
    where larger scores indicate more anomalous samples.

    Parameters
    ----------
    normal_scores:
        Scores from normal validation instances only.
    alpha:
        Global target false-alarm / miscoverage level. Use the SAME value for
        every dataset (default: 0.01). Do not tune alpha on test labels.

    Returns
    -------
    dict with threshold and calibration metadata.
    """
    scores = np.asarray(normal_scores, dtype=np.float64)

    if scores.size == 0:
        raise ValueError("Normal-only post-hoc calibration received no validation scores.")

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"conformal_alpha must be in (0, 1), got {alpha}")

    scores = np.sort(scores)
    n = int(scores.size)

    # Finite-sample conformal rank for the (1-alpha) upper quantile.
    # k is 1-based and is capped at n when the calibration set is small.
    k = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    k = min(max(k, 1), n)

    threshold = float(scores[k - 1])
    quantile_level = float(k / n)

    print("=" * 80)
    print("[NORMAL-ONLY POST-HOC CALIBRATION]")
    print("No anomalous validation labels are used.")
    print(f"normal validation samples : {n}")
    print(f"conformal alpha           : {float(alpha):.6f}")
    print(f"conformal rank k          : {k}/{n}")
    print(f"empirical quantile level  : {quantile_level:.6f}")
    print(f"post-hoc threshold        : {threshold:.12f}")
    print("=" * 80)

    return {
        "threshold": threshold,
        "conformal_alpha": float(alpha),
        "quantile_level": quantile_level,
        "num_normal_validation": n,
    }


def build_normal_only_posthoc_params(cfg, val_normal_scores: pd.DataFrame):
    """
    Build post-hoc parameters using ONLY normal validation instances.

    alpha_mlm and beta_center remain fixed from the configuration. The only
    calibrated decision parameter is the threshold. This avoids using anomaly
    labels for model/score/threshold selection.
    """
    post_cfg = cfg.get("posthoc_calibration", {})

    if not post_cfg.get("enabled", False):
        print("[Post-hoc calibration] disabled")
        return None

    required_cols = ["score", "mlm_loss", "center_distance"]
    missing = [c for c in required_cols if c not in val_normal_scores.columns]
    if missing:
        raise ValueError(
            f"Normal-only post-hoc calibration requires columns {required_cols}. Missing: {missing}"
        )

    # Safety check: calibration must contain NORMAL validation instances only.
    if "label" in val_normal_scores.columns:
        labels = set(val_normal_scores["label"].astype(int).unique().tolist())
        if labels - {0}:
            raise ValueError(
                "Normal-only post-hoc calibration received anomalous validation samples. "
                "Expected label=0 only."
            )

    alpha_mlm = float(cfg["hybrid_scoring"]["alpha_mlm"])
    beta_center = float(cfg["hybrid_scoring"]["beta_center"])

    # One global value for all datasets. If omitted from YAML, 0.01 is used.
    conformal_alpha = float(post_cfg.get("conformal_alpha", 0.01))

    calibration = calibrate_conformal_normal_only(
        val_normal_scores["score"].to_numpy(),
        alpha=conformal_alpha,
    )

    params = {
        "method": "normal_only_conformal",
        "alpha_mlm": alpha_mlm,
        "beta_center": beta_center,
        **calibration,
    }

    print("[Post-hoc parameters fixed BEFORE test evaluation]")
    print(params)
    return params


def apply_fixed_posthoc_parameters(score_df: pd.DataFrame, posthoc_params: dict) -> pd.DataFrame:
    """
    Apply already-fixed post-hoc parameters to a score dataframe.

    IMPORTANT:
    This function performs NO search, optimization, or calibration. It simply
    applies parameters already fixed from NORMAL validation data, so it is safe
    to apply to the held-out test set.
    """
    if posthoc_params is None:
        return score_df.copy()

    alpha_mlm = float(posthoc_params["alpha_mlm"])
    beta_center = float(posthoc_params["beta_center"])
    threshold = float(posthoc_params["threshold"])

    calibrated_df = score_df.copy()
    calibrated_df["base_score"] = calibrated_df["score"]

    calibrated_df["score"] = (
        alpha_mlm * calibrated_df["mlm_loss"].to_numpy()
        + beta_center * calibrated_df["center_distance"].to_numpy()
    )
    calibrated_df["prediction"] = (calibrated_df["score"] > threshold).astype(int)

    calibrated_df["posthoc_method"] = posthoc_params.get("method", "normal_only_conformal")
    calibrated_df["posthoc_alpha_mlm"] = alpha_mlm
    calibrated_df["posthoc_beta_center"] = beta_center
    calibrated_df["posthoc_threshold"] = threshold
    calibrated_df["posthoc_conformal_alpha"] = posthoc_params.get("conformal_alpha")
    calibrated_df["posthoc_quantile_level"] = posthoc_params.get("quantile_level")

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

    Calibration protocol:
        1) load NORMAL validation instances only
        2) compute base threshold from normal validation scores
        3) compute normal-only conformal post-hoc threshold from the SAME
           normal validation scores
        4) freeze all parameters
        5) only then load and evaluate the held-out test set

    Test labels are used only in evaluate_scores() for final reporting.
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

    # ==============================================================
    # NORMAL VALIDATION ONLY
    # ==============================================================
    # IMPORTANT: normal_only=True is explicit. No anomalous validation
    # instance is allowed into either base or post-hoc calibration.
    #
    # LEGACY FULL-VALIDATION CALIBRATION -- DISABLED / DO NOT USE:
    # val_full_df = load_sequences_for_dataset(
    #     cfg, dataset=dataset_name, split=mode_cfg["val_split"], normal_only=False
    # )
    # The line above is intentionally commented because anomalous validation
    # samples must NOT be used for post-hoc calibration.
    val_normal_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["val_split"],
        normal_only=True,
    )

    print(f"[Sequences] val_normal={len(val_normal_df)}")

    val_normal_loader = make_loader(
        cfg,
        val_normal_df,
        tokenizer,
        shuffle=False,
    )

    val_normal_scores = score_loader(
        model,
        val_normal_loader,
        center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Scoring in-domain NORMAL validation data",
    )

    # Save ONLY normal validation scores for reproducible posthoc_only runs.
    validation_normal_file = out_dir / "validation_normal_scores.csv"
    val_normal_scores.to_csv(validation_normal_file, index=False)
    print(f"[Normal validation scores saved] {validation_normal_file}")

    # Base threshold: original normal-only rule from YAML.
    base_threshold = calibrate_threshold(
        val_normal_scores["score"],
        method=cfg["threshold"]["method"],
        percentile=cfg["threshold"].get("percentile", 95),
        fixed_threshold=cfg["prediction_stage"].get("anomaly_threshold", 0.5),
    )

    # ==============================================================
    # NORMAL-ONLY POST-HOC CALIBRATION
    # ==============================================================
    # Calibration below uses ONLY normal validation scores.

    posthoc_params = build_normal_only_posthoc_params(cfg, val_normal_scores)

    # ==============================================================
    # HELD-OUT TEST -- LOADED ONLY AFTER CALIBRATION IS FIXED
    # ==============================================================
    test_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["test_split"],
        normal_only=mode_cfg.get("test_normal_only", False),
    )

    print(f"[Sequences] held-out test={len(test_df)}")

    test_loader = make_loader(
        cfg,
        test_df,
        tokenizer,
        shuffle=False,
    )

    test_scores = score_loader(
        model,
        test_loader,
        center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Predicting in-domain held-out TEST data",
    )

    # Base test result: original threshold calibrated from normal validation.
    metrics = evaluate_scores(
        test_scores,
        base_threshold,
        normal_label=0,
    )

    save_classification_report_files(metrics, out_dir, prefix="test")

    print("[Saving base test predictions]")
    test_scores["prediction"] = (test_scores["score"] > base_threshold).astype(int)
    test_scores.to_csv(
        out_dir / cfg["outputs"].get("predictions_file", "predictions.csv"),
        index=False,
    )
    print("[Base test predictions saved]")

    # Final post-hoc test result: threshold was fixed using NORMAL validation only.
    posthoc_test_metrics = None
    if posthoc_params is not None:
        posthoc_test_scores = apply_fixed_posthoc_parameters(test_scores, posthoc_params)

        posthoc_test_metrics = evaluate_scores(
            posthoc_test_scores,
            float(posthoc_params["threshold"]),
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
            "posthoc_params": posthoc_params,
            "posthoc_test_metrics": posthoc_test_metrics,
        },
        out_dir / cfg["outputs"].get("results_file", "results.json"),
    )

    print("=" * 80)
    print("[IN-DOMAIN PREDICTION FINISHED]")
    print("[Base test Classification Report]")
    print(metrics["classification_report_text"])

    if posthoc_params is not None:
        print("[Post-hoc parameters selected from NORMAL VALIDATION ONLY]")
        print(posthoc_params)
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

    Cross-dataset calibration protocol:
        1) use TARGET NORMAL validation instances only
        2) compute target-specific thresholds from those normal scores
        3) freeze all parameters
        4) only then load/evaluate the held-out TARGET test set

    No anomalous target validation label and no target test label is used for
    calibration or parameter selection.
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

    # ==============================================================
    # TARGET NORMAL VALIDATION ONLY
    # ==============================================================
    # LEGACY FULL-TARGET-VALIDATION CALIBRATION -- DISABLED / DO NOT USE:
    # target_val_full_df = load_sequences_for_dataset(
    #     cfg, dataset=target_dataset, split=mode_cfg["target_val_split"], normal_only=False
    # )
    # The line above is intentionally commented because anomalous target
    # validation samples must NOT be used for post-hoc calibration.
    target_val_normal_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_val_split"],
        normal_only=True,
    )

    print(f"[Sequences] target_val_normal={len(target_val_normal_df)}")

    target_val_normal_loader = make_loader(
        cfg,
        target_val_normal_df,
        tokenizer,
        shuffle=False,
    )

    val_normal_scores = score_loader(
        model,
        target_val_normal_loader,
        target_center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Scoring target NORMAL validation data",
    )

    validation_normal_file = out_dir / "validation_normal_scores.csv"
    val_normal_scores.to_csv(validation_normal_file, index=False)
    print(f"[Normal target validation scores saved] {validation_normal_file}")

    base_threshold = calibrate_threshold(
        val_normal_scores["score"],
        method=cfg["threshold"]["method"],
        percentile=cfg["threshold"].get("percentile", 95),
        fixed_threshold=cfg["prediction_stage"].get("anomaly_threshold", 0.5),
    )

    # Calibration below uses ONLY normal target-validation scores.
    posthoc_params = build_normal_only_posthoc_params(cfg, val_normal_scores)

    # ==============================================================
    # HELD-OUT TARGET TEST -- ONLY AFTER CALIBRATION IS FIXED
    # ==============================================================
    target_test_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_test_split"],
        normal_only=mode_cfg.get("target_test_normal_only", False),
    )

    print(f"[Sequences] held-out target_test={len(target_test_df)}")

    target_test_loader = make_loader(
        cfg,
        target_test_df,
        tokenizer,
        shuffle=False,
    )

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
        base_threshold,
        normal_label=0,
    )

    save_classification_report_files(metrics, out_dir, prefix="test")

    print("[Saving base test predictions]")
    test_scores["prediction"] = (test_scores["score"] > base_threshold).astype(int)
    test_scores.to_csv(
        out_dir / cfg["outputs"].get("predictions_file", "predictions.csv"),
        index=False,
    )
    print("[Base test predictions saved]")

    posthoc_test_metrics = None
    if posthoc_params is not None:
        posthoc_test_scores = apply_fixed_posthoc_parameters(test_scores, posthoc_params)

        posthoc_test_metrics = evaluate_scores(
            posthoc_test_scores,
            float(posthoc_params["threshold"]),
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
            "posthoc_params": posthoc_params,
            "posthoc_test_metrics": posthoc_test_metrics,
            "target_dataset": target_dataset,
        },
        out_dir / cfg["outputs"].get("results_file", "results.json"),
    )

    print("=" * 80)
    print("[FEW-SHOT TARGET ADAPTATION PREDICTION FINISHED]")
    print("[Base test Classification Report]")
    print(metrics["classification_report_text"])

    if posthoc_params is not None:
        print("[Post-hoc parameters selected from TARGET NORMAL VALIDATION ONLY]")
        print(posthoc_params)
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
    Re-run post-hoc calibration without re-running the model.

    Calibration uses validation_normal_scores.csv, which contains NORMAL
    validation scores only. predictions.csv is held-out test data and is loaded
    only after the normal-only threshold is fixed.
    """
    out_dir = ensure_output_dir(cfg)

    validation_normal_file = out_dir / "validation_normal_scores.csv"
    predictions_file = out_dir / cfg["outputs"].get(
        "predictions_file",
        "predictions.csv",
    )

    if not validation_normal_file.exists():
        raise FileNotFoundError(
            f"Normal validation score file not found: {validation_normal_file}\n"
            "Run stage: predict first with the revised normal-only calibration code."
        )

    print("=" * 80)
    print("[Stage] POSTHOC ONLY -- NORMAL VALIDATION ONLY")
    print(f"[Loading NORMAL validation scores] {validation_normal_file}")
    print("=" * 80)

    val_normal_scores = pd.read_csv(validation_normal_file)

    # Safety check happens inside build_normal_only_posthoc_params().
    posthoc_params = build_normal_only_posthoc_params(cfg, val_normal_scores)

    posthoc_test_metrics = None

    if posthoc_params is not None and predictions_file.exists():
        print(f"[Loading held-out TEST scores AFTER calibration] {predictions_file}")
        test_score_df = pd.read_csv(predictions_file)

        posthoc_test_scores = apply_fixed_posthoc_parameters(
            test_score_df,
            posthoc_params,
        )

        posthoc_test_metrics = evaluate_scores(
            posthoc_test_scores,
            float(posthoc_params["threshold"]),
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
            "posthoc_params": posthoc_params,
            "posthoc_test_metrics": posthoc_test_metrics,
        },
        out_dir / "posthoc_only_results.json",
    )

    print("=" * 80)
    print("[POSTHOC FINISHED]")
    print("[Post-hoc parameters selected from NORMAL VALIDATION ONLY]")
    print(posthoc_params)
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