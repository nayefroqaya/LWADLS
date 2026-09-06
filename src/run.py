from __future__ import annotations

import argparse
import json
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


def build_limited_calibration_set(
    seq_df: pd.DataFrame,
    normal_fraction: float,
    anomaly_fraction: float,
    seed: int,
):
    """
    Build the labeled validation subset used by post-hoc calibration.

    Calibration uses:
        - only a fixed fraction of NORMAL validation sequences
        - only a fixed fraction of ANOMALOUS validation sequences

    With normal_fraction=0.20 and anomaly_fraction=0.20, the calibration
    subset preserves (approximately) the original validation class ratio
    while exposing only 20% of each class to supervised post-hoc calibration.

    IMPORTANT:
        The FULL normal validation set is still used elsewhere for:
            - normal center/prototype construction
            - original/base threshold calibration

        This helper affects ONLY the supervised post-hoc calibration subset.

    Notes
    -----
    * Sampling is performed at SEQUENCE level because load_sequences_for_dataset()
      already returns sequence-level rows with binary labels:
          0 = normal
          1 = anomaly
    * The random seed makes both sampled subsets reproducible.
    """
    if "label" not in seq_df.columns:
        raise ValueError(
            "Validation sequence dataframe must contain a 'label' column."
        )

    normal_fraction = float(normal_fraction)
    anomaly_fraction = float(anomaly_fraction)

    if not 0.0 < normal_fraction <= 1.0:
        raise ValueError(
            f"normal_fraction must be in (0, 1], got {normal_fraction}."
        )

    if not 0.0 < anomaly_fraction <= 1.0:
        raise ValueError(
            f"anomaly_fraction must be in (0, 1], got {anomaly_fraction}."
        )

    normal_df = seq_df[seq_df["label"].astype(int) == 0].copy()
    anomaly_df = seq_df[seq_df["label"].astype(int) != 0].copy()

    if normal_df.empty:
        raise ValueError("No normal validation sequences are available.")

    if anomaly_df.empty:
        raise ValueError(
            "No anomalous validation sequences are available. "
            "Label-based post-hoc calibration requires at least one anomaly."
        )

    # Approximately the requested fraction of each class.
    # max(1, ...) keeps calibration usable for very small validation sets.
    num_normal_used = max(
        1,
        int(round(len(normal_df) * normal_fraction)),
    )
    num_anomaly_used = max(
        1,
        int(round(len(anomaly_df) * anomaly_fraction)),
    )

    num_normal_used = min(num_normal_used, len(normal_df))
    num_anomaly_used = min(num_anomaly_used, len(anomaly_df))

    normal_sample_df = normal_df.sample(
        n=num_normal_used,
        replace=False,
        random_state=int(seed),
    ).copy()

    # Use a deterministic but different seed for anomalies so the two
    # class-specific draws are independently reproducible.
    anomaly_sample_df = anomaly_df.sample(
        n=num_anomaly_used,
        replace=False,
        random_state=int(seed) + 1,
    ).copy()

    calibration_df = pd.concat(
        [normal_sample_df, anomaly_sample_df],
        ignore_index=True,
    )

    # Deterministic shuffle so normal/anomaly rows are not stored in blocks.
    calibration_df = calibration_df.sample(
        frac=1.0,
        replace=False,
        random_state=int(seed),
    ).reset_index(drop=True)

    original_anomaly_rate = float(len(anomaly_df) / len(seq_df))
    calibration_anomaly_rate = float(num_anomaly_used / len(calibration_df))

    info = {
        "normal_fraction_requested": normal_fraction,
        "anomaly_fraction_requested": anomaly_fraction,
        "seed": int(seed),
        "normal_validation_total": int(len(normal_df)),
        "anomaly_validation_total": int(len(anomaly_df)),
        "normal_used_for_calibration": int(num_normal_used),
        "anomaly_used_for_calibration": int(num_anomaly_used),
        "normal_excluded_from_calibration": int(
            len(normal_df) - num_normal_used
        ),
        "anomaly_excluded_from_calibration": int(
            len(anomaly_df) - num_anomaly_used
        ),
        "calibration_total": int(len(calibration_df)),
        "realized_normal_fraction_of_available_normals": float(
            num_normal_used / len(normal_df)
        ),
        "realized_anomaly_fraction_of_available_anomalies": float(
            num_anomaly_used / len(anomaly_df)
        ),
        "original_validation_anomaly_rate": original_anomaly_rate,
        "calibration_subset_anomaly_rate": calibration_anomaly_rate,
    }

    print("=" * 80)
    print("[LIMITED BALANCED-FRACTION VALIDATION CALIBRATION SET]")
    print(f"Total normal validation sequences    : {len(normal_df):,}")
    print(f"Total anomaly validation sequences   : {len(anomaly_df):,}")
    print(
        f"Normal fraction requested            : "
        f"{100.0 * normal_fraction:.1f}%"
    )
    print(
        f"Anomaly fraction requested           : "
        f"{100.0 * anomaly_fraction:.1f}%"
    )
    print(f"Normal sequences used                : {num_normal_used:,}")
    print(f"Anomaly sequences used               : {num_anomaly_used:,}")
    print(
        f"Normal sequences excluded            : "
        f"{len(normal_df) - num_normal_used:,}"
    )
    print(
        f"Anomaly sequences excluded           : "
        f"{len(anomaly_df) - num_anomaly_used:,}"
    )
    print(f"Calibration subset size              : {len(calibration_df):,}")
    print(
        f"Original validation anomaly rate     : "
        f"{100.0 * original_anomaly_rate:.3f}%"
    )
    print(
        f"Calibration subset anomaly rate      : "
        f"{100.0 * calibration_anomaly_rate:.3f}%"
    )
    print(f"Sampling seed                        : {int(seed)}")
    print("=" * 80)

    return calibration_df, info



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

    Calibration protocol:
        1) NORMAL-ONLY validation data -> original/base threshold.
        2) Labeled calibration subset -> 20% of validation normals + 20%
           of validation anomalies by default (configurable via
           posthoc_calibration.normal_fraction and anomaly_fraction).
        3) Post-hoc alpha/beta/percentile/threshold are selected ONLY from
           that limited validation subset.
        4) Parameters are frozen.
        5) The held-out test set is loaded/scored only after calibration
           parameters are fixed.

    It does NOT retrain.
    """

    out_dir = ensure_output_dir(cfg)

    set_seed(cfg["experiment"].get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    mode_cfg = cfg["datasets"]["in_domain"]
    dataset_name = mode_cfg["dataset_name"]
    seed = int(cfg["experiment"].get("seed", 42))

    # Default: expose only 20% of each validation class to post-hoc calibration.
    calibration_cfg = cfg.get("posthoc_calibration", {})
    normal_fraction = float(calibration_cfg.get("normal_fraction", 0.10))
    anomaly_fraction = float(calibration_cfg.get("anomaly_fraction", 0.10))

    print("=" * 80)
    print("[Stage] PREDICT ONLY")
    print("[Mode] in_domain")
    print(f"[Dataset] {dataset_name}")
    print(
        f"[Calibration normal budget] "
        f"{100.0 * normal_fraction:.1f}% of validation normals"
    )
    print(
        f"[Calibration anomaly budget] "
        f"{100.0 * anomaly_fraction:.1f}% of validation anomalies"
    )
    print("=" * 80)

    model, tokenizer, center = load_saved_model_tokenizer_center(
        cfg=cfg,
        out_dir=out_dir,
        device=device,
        center_filename="normal_center.pt",
    )

    # ------------------------------------------------------------------
    # VALIDATION VIEW 1: NORMAL ONLY
    # Used for the original/base normality threshold.
    # ------------------------------------------------------------------
    val_normal_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["val_split"],
        normal_only=True,
    )

    # ------------------------------------------------------------------
    # VALIDATION POOL FOR LIMITED SUPERVISED CALIBRATION
    # Load the benchmark validation pool, then expose only:
    #     normal_fraction of normal sequences
    #     anomaly_fraction of anomaly sequences
    #
    # With both set to 0.20, post-hoc calibration uses 20% of each class,
    # approximately preserving the original validation class ratio.
    # The FULL normal validation view above remains unchanged for the base
    # threshold and normal-reference computations.
    # ------------------------------------------------------------------
    val_pool_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["val_split"],
        normal_only=False,
    )

    val_calibration_df, calibration_info = (
        build_limited_calibration_set(
            val_pool_df,
            normal_fraction=normal_fraction,
            anomaly_fraction=anomaly_fraction,
            seed=seed,
        )
    )

    print(
        f"[Sequences before TEST] "
        f"val_normal={len(val_normal_df)} "
        f"calibration_subset={len(val_calibration_df)}"
    )

    val_normal_loader = make_loader(
        cfg,
        val_normal_df,
        tokenizer,
        shuffle=False,
    )

    val_calibration_loader = make_loader(
        cfg,
        val_calibration_df,
        tokenizer,
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # BASE THRESHOLD: NORMAL VALIDATION ONLY
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
        fixed_threshold=cfg["prediction_stage"].get(
            "anomaly_threshold",
            0.5,
        ),
    )

    # ------------------------------------------------------------------
    # POST-HOC CALIBRATION:
    # 20% validation normals + 20% validation anomalies by default.
    # ------------------------------------------------------------------
    val_calibration_scores = score_loader(
        model,
        val_calibration_loader,
        center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc=(
            "Scoring LIMITED labeled validation subset "
            "(sampled normals + sampled anomalies)"
        ),
    )

    # Safety checks: the post-hoc search must see both classes, and the anomaly
    # count must match the sampled calibration subset rather than the full pool.
    if val_calibration_scores["label"].nunique() < 2:
        raise ValueError(
            "Limited validation calibration subset must contain both "
            "normal and anomaly samples."
        )

    scored_normal_count = int(
        (val_calibration_scores["label"].astype(int) == 0).sum()
    )
    scored_anomaly_count = int(
        (val_calibration_scores["label"].astype(int) != 0).sum()
    )

    if scored_normal_count != calibration_info["normal_used_for_calibration"]:
        raise RuntimeError(
            "Unexpected normal count after scoring calibration subset: "
            f"expected={calibration_info['normal_used_for_calibration']}, "
            f"got={scored_normal_count}."
        )

    if scored_anomaly_count != calibration_info["anomaly_used_for_calibration"]:
        raise RuntimeError(
            "Unexpected anomaly count after scoring calibration subset: "
            f"expected={calibration_info['anomaly_used_for_calibration']}, "
            f"got={scored_anomaly_count}."
        )

    # Save ONLY the limited calibration subset scores.
    # stage=posthoc_only will therefore also use the same sampled class fractions.
    validation_calibration_file = (
        out_dir / "validation_calibration_scores.csv"
    )
    val_calibration_scores.to_csv(
        validation_calibration_file,
        index=False,
    )
    print(
        f"[Limited validation calibration scores saved] "
        f"{validation_calibration_file}"
    )

    calibration_sampling_file = (
        out_dir / "validation_calibration_sampling.json"
    )
    save_json(calibration_info, calibration_sampling_file)
    print(
        f"[Calibration sampling metadata saved] "
        f"{calibration_sampling_file}"
    )

    # Save an audit table showing which validation sequences were exposed.
    audit_cols = [
        c
        for c in ["dataset", "group_id", "label", "num_events"]
        if c in val_calibration_df.columns
    ]
    if audit_cols:
        calibration_subset_file = (
            out_dir / "validation_calibration_subset.csv"
        )
        val_calibration_df[audit_cols].to_csv(
            calibration_subset_file,
            index=False,
        )
        print(
            f"[Calibration subset audit saved] "
            f"{calibration_subset_file}"
        )

    # Select post-hoc parameters using ONLY the limited validation subset.
    posthoc_best, _ = run_posthoc_grid_search(
        score_df=val_calibration_scores,
        cfg=cfg,
        output_dir=out_dir,
        normal_label=0,
    )

    # ------------------------------------------------------------------
    # TEST: load only AFTER calibration is complete and frozen.
    # No test score/label participates in parameter selection.
    # ------------------------------------------------------------------
    test_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["test_split"],
        normal_only=mode_cfg.get("test_normal_only", False),
    )

    print(f"[Held-out TEST sequences] {len(test_df)}")

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

    # Base result retained for comparison.
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
    test_scores["prediction"] = (
        test_scores["score"] > threshold
    ).astype(int)
    test_scores.to_csv(
        out_dir
        / cfg["outputs"].get(
            "predictions_file",
            "predictions.csv",
        ),
        index=False,
    )
    print("[Base test predictions saved]")

    # Apply only the FROZEN validation-selected post-hoc parameters to test.
    posthoc_test_metrics = None

    if posthoc_best is not None:
        posthoc_test_scores = apply_fixed_posthoc_parameters(
            test_scores,
            posthoc_best,
        )

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

        posthoc_test_predictions_file = (
            out_dir / "posthoc_test_predictions.csv"
        )
        posthoc_test_scores.to_csv(
            posthoc_test_predictions_file,
            index=False,
        )
        print(
            f"[Post-hoc TEST predictions saved] "
            f"{posthoc_test_predictions_file}"
        )

    save_json(
        {
            "stage": "predict",
            "mode": "in_domain",
            "dataset": dataset_name,
            "metrics": metrics,
            "calibration_sampling": calibration_info,
            "posthoc_best": posthoc_best,
            "posthoc_best_validation": posthoc_best,
            "posthoc_test_metrics": posthoc_test_metrics,
        },
        out_dir
        / cfg["outputs"].get(
            "results_file",
            "results.json",
        ),
    )

    print("=" * 80)
    print("[IN-DOMAIN PREDICTION FINISHED]")
    print("[Base test Classification Report]")
    print(metrics["classification_report_text"])

    print("[Validation calibration sampling: sampled normals + sampled anomalies]")
    print(calibration_info)

    if posthoc_best is not None:
        print(
            "[Post-hoc best selected on LIMITED VALIDATION "
            "(sampled normals + sampled anomalies)]"
        )
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

    Calibration protocol on the TARGET validation split:
        1) NORMAL-ONLY target validation -> original/base threshold.
        2) Labeled calibration subset -> 20% of target validation normals +
           20% of target validation anomalies by default.
        3) Post-hoc parameters are selected only from that limited subset.
        4) Parameters are frozen.
        5) Held-out target test is loaded/scored only after calibration.

    It does NOT retrain.
    """

    out_dir = ensure_output_dir(cfg)

    set_seed(cfg["experiment"].get("seed", 42))
    device = get_device(cfg["training"].get("device", "auto"))

    mode_cfg = cfg["datasets"]["fewshot_target_adaptation"]
    target_dataset = mode_cfg["target_dataset"]
    seed = int(cfg["experiment"].get("seed", 42))

    calibration_cfg = cfg.get("posthoc_calibration", {})
    normal_fraction = float(calibration_cfg.get("normal_fraction", 0.20))
    anomaly_fraction = float(calibration_cfg.get("anomaly_fraction", 0.20))

    print("=" * 80)
    print("[Stage] PREDICT ONLY")
    print("[Mode] fewshot_target_adaptation")
    print(f"[Target dataset] {target_dataset}")
    print(
        f"[Target calibration normal budget] "
        f"{100.0 * normal_fraction:.1f}% of validation normals"
    )
    print(
        f"[Target calibration anomaly budget] "
        f"{100.0 * anomaly_fraction:.1f}% of validation anomalies"
    )
    print("=" * 80)

    model, tokenizer, target_center = load_saved_model_tokenizer_center(
        cfg=cfg,
        out_dir=out_dir,
        device=device,
        center_filename="target_normal_center.pt",
    )

    # ------------------------------------------------------------------
    # TARGET VALIDATION VIEW 1: NORMAL ONLY
    # ------------------------------------------------------------------
    target_val_normal_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_val_split"],
        normal_only=True,
    )

    # ------------------------------------------------------------------
    # TARGET VALIDATION POOL:
    # Build a calibration subset containing only normal_fraction of target
    # validation normals and anomaly_fraction of target validation anomalies.
    # With both set to 0.20, the original target-validation class ratio is
    # approximately preserved.
    # ------------------------------------------------------------------
    target_val_pool_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_val_split"],
        normal_only=False,
    )

    target_val_calibration_df, calibration_info = (
        build_limited_calibration_set(
            target_val_pool_df,
            normal_fraction=normal_fraction,
            anomaly_fraction=anomaly_fraction,
            seed=seed,
        )
    )

    print(
        f"[Sequences before TARGET TEST] "
        f"target_val_normal={len(target_val_normal_df)} "
        f"target_calibration_subset={len(target_val_calibration_df)}"
    )

    target_val_normal_loader = make_loader(
        cfg,
        target_val_normal_df,
        tokenizer,
        shuffle=False,
    )

    target_val_calibration_loader = make_loader(
        cfg,
        target_val_calibration_df,
        tokenizer,
        shuffle=False,
    )

    # Base threshold from target NORMAL validation only.
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
        fixed_threshold=cfg["prediction_stage"].get(
            "anomaly_threshold",
            0.5,
        ),
    )

    # Limited supervised target-validation calibration.
    val_calibration_scores = score_loader(
        model,
        target_val_calibration_loader,
        target_center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc=(
            "Scoring LIMITED target validation subset "
            "(sampled normals + sampled anomalies)"
        ),
    )

    if val_calibration_scores["label"].nunique() < 2:
        raise ValueError(
            "Limited target-validation calibration subset must contain "
            "both normal and anomaly samples."
        )

    scored_normal_count = int(
        (val_calibration_scores["label"].astype(int) == 0).sum()
    )
    scored_anomaly_count = int(
        (val_calibration_scores["label"].astype(int) != 0).sum()
    )

    if scored_normal_count != calibration_info["normal_used_for_calibration"]:
        raise RuntimeError(
            "Unexpected target normal count after scoring calibration subset: "
            f"expected={calibration_info['normal_used_for_calibration']}, "
            f"got={scored_normal_count}."
        )

    if scored_anomaly_count != calibration_info["anomaly_used_for_calibration"]:
        raise RuntimeError(
            "Unexpected target anomaly count after scoring calibration subset: "
            f"expected={calibration_info['anomaly_used_for_calibration']}, "
            f"got={scored_anomaly_count}."
        )

    # Save ONLY the limited target validation calibration scores.
    validation_calibration_file = (
        out_dir / "validation_calibration_scores.csv"
    )
    val_calibration_scores.to_csv(
        validation_calibration_file,
        index=False,
    )
    print(
        f"[Limited target validation calibration scores saved] "
        f"{validation_calibration_file}"
    )

    calibration_sampling_file = (
        out_dir / "validation_calibration_sampling.json"
    )
    save_json(calibration_info, calibration_sampling_file)
    print(
        f"[Target calibration sampling metadata saved] "
        f"{calibration_sampling_file}"
    )

    audit_cols = [
        c
        for c in ["dataset", "group_id", "label", "num_events"]
        if c in target_val_calibration_df.columns
    ]
    if audit_cols:
        calibration_subset_file = (
            out_dir / "validation_calibration_subset.csv"
        )
        target_val_calibration_df[audit_cols].to_csv(
            calibration_subset_file,
            index=False,
        )
        print(
            f"[Target calibration subset audit saved] "
            f"{calibration_subset_file}"
        )

    posthoc_best, _ = run_posthoc_grid_search(
        score_df=val_calibration_scores,
        cfg=cfg,
        output_dir=out_dir,
        normal_label=0,
    )

    # ------------------------------------------------------------------
    # HELD-OUT TARGET TEST:
    # loaded only after calibration parameters are frozen.
    # ------------------------------------------------------------------
    target_test_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_test_split"],
        normal_only=mode_cfg.get("target_test_normal_only", False),
    )

    print(f"[Held-out target TEST sequences] {len(target_test_df)}")

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
        threshold,
        normal_label=0,
    )

    save_classification_report_files(
        metrics,
        out_dir,
        prefix="test",
    )

    print("[Saving base target test predictions]")
    test_scores["prediction"] = (
        test_scores["score"] > threshold
    ).astype(int)
    test_scores.to_csv(
        out_dir
        / cfg["outputs"].get(
            "predictions_file",
            "predictions.csv",
        ),
        index=False,
    )
    print("[Base target test predictions saved]")

    posthoc_test_metrics = None

    if posthoc_best is not None:
        posthoc_test_scores = apply_fixed_posthoc_parameters(
            test_scores,
            posthoc_best,
        )

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

        posthoc_test_predictions_file = (
            out_dir / "posthoc_test_predictions.csv"
        )
        posthoc_test_scores.to_csv(
            posthoc_test_predictions_file,
            index=False,
        )
        print(
            f"[Post-hoc target TEST predictions saved] "
            f"{posthoc_test_predictions_file}"
        )

    save_json(
        {
            "stage": "predict",
            "mode": "fewshot_target_adaptation",
            "metrics": metrics,
            "calibration_sampling": calibration_info,
            "posthoc_best": posthoc_best,
            "posthoc_best_validation": posthoc_best,
            "posthoc_test_metrics": posthoc_test_metrics,
            "target_dataset": target_dataset,
        },
        out_dir
        / cfg["outputs"].get(
            "results_file",
            "results.json",
        ),
    )

    print("=" * 80)
    print("[FEW-SHOT TARGET ADAPTATION PREDICTION FINISHED]")
    print("[Base target test Classification Report]")
    print(metrics["classification_report_text"])

    print("[Target validation calibration sampling: sampled normals + sampled anomalies]")
    print(calibration_info)

    if posthoc_best is not None:
        print(
            "[Post-hoc best selected on LIMITED TARGET VALIDATION "
            "(sampled normals + sampled anomalies)]"
        )
        print(posthoc_best)
        print("[Final post-hoc TARGET TEST Classification Report]")
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
    created by the predict stage from the LIMITED labeled validation subset:
        - only the configured fraction of validation normals (default 20%)
        - only the configured fraction of validation anomalies (default 20%)

    predictions.csv contains held-out test scores and is NEVER used for
    post-hoc parameter search. After parameters are selected on validation, they
    are applied once to predictions.csv for final test evaluation.
    """

    out_dir = ensure_output_dir(cfg)

    validation_calibration_file = out_dir / "validation_calibration_scores.csv"
    calibration_sampling_file = (
        out_dir / "validation_calibration_sampling.json"
    )
    predictions_file = out_dir / cfg["outputs"].get(
        "predictions_file",
        "predictions.csv",
    )

    if not validation_calibration_file.exists():
        raise FileNotFoundError(
            f"Validation calibration file not found: {validation_calibration_file}\n"
            "Run stage: predict first so the limited validation calibration scores are saved."
        )

    if not calibration_sampling_file.exists():
        raise FileNotFoundError(
            f"Calibration sampling metadata not found: {calibration_sampling_file}\n"
            "Refusing to run posthoc_only because the existing calibration CSV "
            "may come from an older run that used a different calibration subset. "
            "Run stage: predict first with LIMITED_BALANCED_FRACTION_VALIDATION_V2."
        )

    print("=" * 80)
    print("[Stage] POSTHOC ONLY")
    print(f"[Loading LIMITED VALIDATION calibration scores] {validation_calibration_file}")
    print("=" * 80)

    with open(calibration_sampling_file, "r", encoding="utf-8") as f:
        calibration_info = json.load(f)

    print("[Saved calibration sampling]")
    print(calibration_info)

    validation_score_df = pd.read_csv(validation_calibration_file)

    expected_normals = int(
        calibration_info["normal_used_for_calibration"]
    )
    expected_anomalies = int(
        calibration_info["anomaly_used_for_calibration"]
    )
    actual_normals = int(
        (validation_score_df["label"].astype(int) == 0).sum()
    )
    actual_anomalies = int(
        (validation_score_df["label"].astype(int) != 0).sum()
    )

    if actual_normals != expected_normals:
        raise RuntimeError(
            "Saved calibration scores do not match sampling metadata: "
            f"expected normals={expected_normals}, "
            f"found={actual_normals}."
        )

    if actual_anomalies != expected_anomalies:
        raise RuntimeError(
            "Saved calibration scores do not match sampling metadata: "
            f"expected anomalies={expected_anomalies}, "
            f"found={actual_anomalies}."
        )

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

    # Optimize only on the saved LIMITED validation calibration scores.
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
            "calibration_sampling": calibration_info,
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

    calibration_cfg = cfg.get("posthoc_calibration", {})
    calibration_normal_fraction = float(
        calibration_cfg.get("normal_fraction", 0.20)
    )
    calibration_anomaly_fraction = float(
        calibration_cfg.get("anomaly_fraction", 0.20)
    )
    print("[Calibration protocol] LIMITED_BALANCED_FRACTION_VALIDATION_V2")
    print(
        f"[Calibration normal fraction] "
        f"{100.0 * calibration_normal_fraction:.1f}%"
    )
    print(
        f"[Calibration anomaly fraction] "
        f"{100.0 * calibration_anomaly_fraction:.1f}%"
    )

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