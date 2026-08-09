from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.mixture import GaussianMixture

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

CALIBRATION_IMPLEMENTATION = "UNLABELED_GMM_V2"


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

def _find_gmm_posterior_boundary(
    gmm: GaussianMixture,
    normal_component: int,
    anomaly_component: int,
    low: float,
    high: float,
    grid_size: int = 10000,
):
    """
    Find a score threshold where the posterior probabilities of the low-score
    (normal-like) and high-score (anomaly-like) GMM components are equal.

    Returns None if no posterior crossing exists between the component means.
    """
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None

    grid_size = max(int(grid_size), 1000)
    grid = np.linspace(float(low), float(high), grid_size, dtype=np.float64)
    probs = gmm.predict_proba(grid.reshape(-1, 1))

    diff = probs[:, normal_component] - probs[:, anomaly_component]
    crossing_idx = np.where(np.signbit(diff[:-1]) != np.signbit(diff[1:]))[0]

    if crossing_idx.size == 0:
        return None

    # If multiple crossings exist, use the crossing whose posteriors are
    # closest to 0.5/0.5.
    candidates = []
    for idx in crossing_idx:
        local_idx = idx if abs(diff[idx]) <= abs(diff[idx + 1]) else idx + 1
        p_normal = float(probs[local_idx, normal_component])
        p_anomaly = float(probs[local_idx, anomaly_component])
        candidates.append(
            (
                abs(p_normal - p_anomaly),
                float(grid[local_idx]),
                p_normal,
                p_anomaly,
            )
        )

    candidates.sort(key=lambda x: x[0])
    _, threshold, p_normal, p_anomaly = candidates[0]
    return {
        "threshold": threshold,
        "posterior_normal": p_normal,
        "posterior_anomaly": p_anomaly,
    }


def calibrate_unlabeled_gmm(
    unlabeled_scores,
    base_threshold: float,
    *,
    seed: int = 42,
    min_bic_improvement: float = 10.0,
    min_component_weight: float = 0.01,
    reg_covar: float = 1e-6,
    n_init: int = 5,
    grid_size: int = 10000,
) -> dict:
    """
    Label-free post-hoc calibration from the FULL validation score distribution.

    The validation split may physically contain normal and anomalous instances,
    but their labels are NOT provided to this function. The function sees only
    the final LogSLM anomaly score.

    Procedure
    ---------
    1. Fit a 1-component GMM and a 2-component GMM to validation scores.
    2. Use BIC to decide whether the 2-component model provides sufficiently
       stronger evidence of a mixture.
    3. Treat the lower-mean component as normal-like and the higher-mean
       component as anomaly-like (because larger LogSLM scores are more
       anomalous).
    4. If the mixture is reliable, use the posterior crossing as threshold.
    5. Otherwise, fall back to the original NORMAL-validation base threshold.

    No validation label and no test information is used.
    """
    scores = np.asarray(unlabeled_scores, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]

    if scores.size < 10:
        return {
            "method": "unlabeled_gmm_fallback_base",
            "threshold": float(base_threshold),
            "used_gmm": False,
            "fallback_reason": "too_few_validation_scores",
            "num_unlabeled_validation": int(scores.size),
        }

    X = scores.reshape(-1, 1)

    gmm1 = GaussianMixture(
        n_components=1,
        covariance_type="full",
        random_state=int(seed),
        reg_covar=float(reg_covar),
        n_init=max(int(n_init), 1),
    ).fit(X)

    gmm2 = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=int(seed),
        reg_covar=float(reg_covar),
        n_init=max(int(n_init), 1),
    ).fit(X)

    bic1 = float(gmm1.bic(X))
    bic2 = float(gmm2.bic(X))
    bic_improvement = bic1 - bic2  # positive => 2-GMM has lower/better BIC

    means = gmm2.means_.reshape(-1)
    weights = gmm2.weights_.reshape(-1)
    variances = np.asarray(gmm2.covariances_).reshape(2, -1)[:, 0]

    normal_component = int(np.argmin(means))
    anomaly_component = int(np.argmax(means))

    normal_mean = float(means[normal_component])
    anomaly_mean = float(means[anomaly_component])
    normal_weight = float(weights[normal_component])
    anomaly_weight = float(weights[anomaly_component])
    normal_std = float(np.sqrt(max(variances[normal_component], 0.0)))
    anomaly_std = float(np.sqrt(max(variances[anomaly_component], 0.0)))

    pooled_std = max(np.sqrt((normal_std ** 2 + anomaly_std ** 2) / 2.0), 1e-12)
    component_separation = float((anomaly_mean - normal_mean) / pooled_std)

    boundary = _find_gmm_posterior_boundary(
        gmm2,
        normal_component=normal_component,
        anomaly_component=anomaly_component,
        low=normal_mean,
        high=anomaly_mean,
        grid_size=grid_size,
    )

    fallback_reasons = []

    if bic_improvement < float(min_bic_improvement):
        fallback_reasons.append(
            f"BIC improvement {bic_improvement:.6f} < {float(min_bic_improvement):.6f}"
        )

    if min(normal_weight, anomaly_weight) < float(min_component_weight):
        fallback_reasons.append(
            "one GMM component is smaller than the configured minimum weight"
        )

    if anomaly_mean <= normal_mean:
        fallback_reasons.append("component means are not ordered")

    if boundary is None:
        fallback_reasons.append("no posterior crossing between component means")

    use_gmm = len(fallback_reasons) == 0

    if use_gmm:
        threshold = float(boundary["threshold"])
        method = "unlabeled_gmm_mixture"
        fallback_reason = None
    else:
        threshold = float(base_threshold)
        method = "unlabeled_gmm_fallback_base"
        fallback_reason = "; ".join(fallback_reasons)

    params = {
        "method": method,
        "threshold": threshold,
        "used_gmm": bool(use_gmm),
        "base_threshold": float(base_threshold),
        "num_unlabeled_validation": int(scores.size),
        "bic_1_component": bic1,
        "bic_2_component": bic2,
        "bic_improvement": bic_improvement,
        "min_bic_improvement": float(min_bic_improvement),
        "normal_component_mean": normal_mean,
        "anomaly_component_mean": anomaly_mean,
        "normal_component_std": normal_std,
        "anomaly_component_std": anomaly_std,
        "normal_component_weight": normal_weight,
        "anomaly_component_weight": anomaly_weight,
        "estimated_contamination": anomaly_weight,
        "component_separation": component_separation,
        "min_component_weight": float(min_component_weight),
        "posterior_normal_at_threshold": (
            None if boundary is None else float(boundary["posterior_normal"])
        ),
        "posterior_anomaly_at_threshold": (
            None if boundary is None else float(boundary["posterior_anomaly"])
        ),
        "fallback_reason": fallback_reason,
    }

    print("=" * 80)
    print("[UNLABELED FULL-VALIDATION GMM POST-HOC CALIBRATION]")
    print("Validation labels are NOT used by the calibration function.")
    print(f"validation scores          : {scores.size}")
    print(f"base normal-only threshold : {float(base_threshold):.12f}")
    print(f"BIC 1-component            : {bic1:.6f}")
    print(f"BIC 2-component            : {bic2:.6f}")
    print(f"BIC improvement (1 - 2)    : {bic_improvement:.6f}")
    print(f"normal-like mean/weight    : {normal_mean:.12f} / {normal_weight:.6f}")
    print(f"anomaly-like mean/weight   : {anomaly_mean:.12f} / {anomaly_weight:.6f}")
    print(f"component separation       : {component_separation:.6f}")
    print(f"GMM accepted               : {use_gmm}")
    if fallback_reason is not None:
        print(f"fallback reason            : {fallback_reason}")
    print(f"FINAL post-hoc threshold   : {threshold:.12f}")
    print("=" * 80)

    return params


def build_unlabeled_posthoc_params(
    cfg,
    val_unlabeled_scores: pd.DataFrame,
    base_threshold: float,
):
    """
    Build post-hoc parameters from FULL validation scores with labels removed.

    alpha_mlm and beta_center remain fixed from the configuration. Calibration
    sees only the already-computed final LogSLM score, never validation labels.
    """
    post_cfg = cfg.get("posthoc_calibration", {})

    if not post_cfg.get("enabled", False):
        print("[Post-hoc calibration] disabled")
        return None

    # Strong safety barrier: the calibration dataframe must not contain labels.
    if "label" in val_unlabeled_scores.columns:
        raise ValueError(
            "Unlabeled post-hoc calibration received a 'label' column. "
            "Drop validation labels before calling build_unlabeled_posthoc_params()."
        )

    required_cols = ["score", "mlm_loss", "center_distance"]
    missing = [c for c in required_cols if c not in val_unlabeled_scores.columns]
    if missing:
        raise ValueError(
            f"Unlabeled post-hoc calibration requires columns {required_cols}. Missing: {missing}"
        )

    alpha_mlm = float(cfg["hybrid_scoring"]["alpha_mlm"])
    beta_center = float(cfg["hybrid_scoring"]["beta_center"])

    gmm_params = calibrate_unlabeled_gmm(
        val_unlabeled_scores["score"].to_numpy(),
        base_threshold=float(base_threshold),
        seed=int(cfg["experiment"].get("seed", 42)),
        min_bic_improvement=float(post_cfg.get("gmm_min_bic_improvement", 10.0)),
        min_component_weight=float(post_cfg.get("gmm_min_component_weight", 0.01)),
        reg_covar=float(post_cfg.get("gmm_reg_covar", 1e-6)),
        n_init=int(post_cfg.get("gmm_n_init", 5)),
        grid_size=int(post_cfg.get("gmm_grid_size", 10000)),
    )

    params = {
        "alpha_mlm": alpha_mlm,
        "beta_center": beta_center,
        **gmm_params,
    }

    print("[Post-hoc parameters fixed BEFORE test evaluation]")
    print(params)
    return params


def apply_fixed_posthoc_parameters(score_df: pd.DataFrame, posthoc_params: dict) -> pd.DataFrame:
    """
    Apply parameters already fixed from validation to a score dataframe.

    This function performs NO search, optimization, fitting, or threshold
    calibration on the held-out test set.
    """
    if posthoc_params is None:
        return score_df.copy()

    alpha_mlm = float(posthoc_params["alpha_mlm"])
    beta_center = float(posthoc_params["beta_center"])
    threshold = float(posthoc_params["threshold"])

    calibrated_df = score_df.copy()
    calibrated_df["base_score"] = calibrated_df["score"]

    # Same LogSLM score definition used for validation and test.
    calibrated_df["score"] = (
        alpha_mlm * calibrated_df["mlm_loss"].to_numpy()
        + beta_center * calibrated_df["center_distance"].to_numpy()
    )
    calibrated_df["prediction"] = (calibrated_df["score"] > threshold).astype(int)

    calibrated_df["posthoc_method"] = posthoc_params.get("method")
    calibrated_df["posthoc_alpha_mlm"] = alpha_mlm
    calibrated_df["posthoc_beta_center"] = beta_center
    calibrated_df["posthoc_threshold"] = threshold
    calibrated_df["posthoc_used_gmm"] = posthoc_params.get("used_gmm")
    calibrated_df["posthoc_estimated_contamination"] = posthoc_params.get(
        "estimated_contamination"
    )

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
    print(f"[Calibration implementation] {CALIBRATION_IMPLEMENTATION}")
    """
    In-domain prediction stage only.

    Calibration protocol:
        1) NORMAL validation only -> original/base threshold
        2) FULL validation split -> score all instances, then DROP labels
        3) fit 1-GMM and 2-GMM to the unlabeled final LogSLM score
        4) if the 2-GMM is reliable, use its posterior crossing threshold;
           otherwise fall back to the original normal-only base threshold
        5) freeze all parameters
        6) only then load/evaluate the held-out test set

    Validation labels are never used by the post-hoc calibration function.
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
    # A) NORMAL VALIDATION -> ORIGINAL / SAFE BASE THRESHOLD
    # ==============================================================
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
        desc="Scoring in-domain NORMAL validation data for base threshold",
    )

    validation_normal_file = out_dir / "validation_normal_scores.csv"
    val_normal_scores.to_csv(validation_normal_file, index=False)
    print(f"[Normal validation scores saved] {validation_normal_file}")

    base_threshold = calibrate_threshold(
        val_normal_scores["score"],
        method=cfg["threshold"]["method"],
        percentile=cfg["threshold"].get("percentile", 95),
        fixed_threshold=cfg["prediction_stage"].get("anomaly_threshold", 0.5),
    )

    # ==============================================================
    # B) FULL VALIDATION -> LABEL-FREE DISTRIBUTIONAL CALIBRATION
    # ==============================================================
    # normal_only=False intentionally loads the complete validation split.
    # It may contain both classes, BUT labels are discarded immediately after
    # scoring and are never passed into the GMM calibration function.
    val_unlabeled_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["val_split"],
        normal_only=False,
    )

    print(f"[Sequences] full validation for LABEL-FREE calibration={len(val_unlabeled_df)}")

    val_unlabeled_loader = make_loader(
        cfg,
        val_unlabeled_df,
        tokenizer,
        shuffle=False,
    )

    val_full_scores = score_loader(
        model,
        val_unlabeled_loader,
        center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Scoring FULL validation data with labels ignored",
    )

    # CRITICAL SAFETY STEP: the calibration dataframe contains no labels.
    val_unlabeled_scores = val_full_scores[
        ["mlm_loss", "center_distance", "score"]
    ].copy()

    validation_unlabeled_file = out_dir / "validation_unlabeled_scores.csv"
    val_unlabeled_scores.to_csv(validation_unlabeled_file, index=False)
    print(f"[Unlabeled validation scores saved] {validation_unlabeled_file}")

    posthoc_params = build_unlabeled_posthoc_params(
        cfg,
        val_unlabeled_scores,
        base_threshold=base_threshold,
    )

    # ==============================================================
    # C) HELD-OUT TEST -- LOADED ONLY AFTER CALIBRATION IS FIXED
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

    # Base test result: original normal-only threshold.
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

    # Final label-free post-hoc test result.
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
            "base_threshold": float(base_threshold),
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
        print("[Post-hoc parameters selected from FULL VALIDATION SCORES WITHOUT LABELS]")
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
    print(f"[Calibration implementation] {CALIBRATION_IMPLEMENTATION}")
    """
    Few-shot target adaptation prediction stage only.

    Cross-dataset calibration protocol:
        1) TARGET NORMAL validation -> original/base threshold
        2) FULL TARGET validation -> score all instances and DROP labels
        3) label-free 1-GMM vs 2-GMM calibration on final LogSLM scores
        4) reliable mixture -> posterior-crossing threshold
           unreliable mixture -> fallback to target normal-only base threshold
        5) freeze all parameters
        6) only then load/evaluate held-out TARGET test data
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
    # A) TARGET NORMAL VALIDATION -> BASE THRESHOLD
    # ==============================================================
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
        desc="Scoring target NORMAL validation data for base threshold",
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

    # ==============================================================
    # B) FULL TARGET VALIDATION -> LABEL-FREE GMM CALIBRATION
    # ==============================================================
    target_val_unlabeled_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_val_split"],
        normal_only=False,
    )

    print(
        f"[Sequences] full target validation for LABEL-FREE calibration="
        f"{len(target_val_unlabeled_df)}"
    )

    target_val_unlabeled_loader = make_loader(
        cfg,
        target_val_unlabeled_df,
        tokenizer,
        shuffle=False,
    )

    target_val_full_scores = score_loader(
        model,
        target_val_unlabeled_loader,
        target_center,
        device,
        alpha_mlm=cfg["hybrid_scoring"]["alpha_mlm"],
        beta_center=cfg["hybrid_scoring"]["beta_center"],
        desc="Scoring FULL target validation data with labels ignored",
    )

    target_val_unlabeled_scores = target_val_full_scores[
        ["mlm_loss", "center_distance", "score"]
    ].copy()

    validation_unlabeled_file = out_dir / "validation_unlabeled_scores.csv"
    target_val_unlabeled_scores.to_csv(validation_unlabeled_file, index=False)
    print(f"[Unlabeled target validation scores saved] {validation_unlabeled_file}")

    posthoc_params = build_unlabeled_posthoc_params(
        cfg,
        target_val_unlabeled_scores,
        base_threshold=base_threshold,
    )

    # ==============================================================
    # C) HELD-OUT TARGET TEST -- ONLY AFTER CALIBRATION IS FIXED
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
            "base_threshold": float(base_threshold),
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
        print("[Post-hoc parameters selected from FULL TARGET VALIDATION SCORES WITHOUT LABELS]")
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
    Re-run label-free GMM post-hoc calibration without re-running the model.

    Requires:
        validation_normal_scores.csv    -> recompute original base threshold
        validation_unlabeled_scores.csv -> contains NO labels; GMM calibration
        predictions.csv                 -> held-out test scores, loaded only
                                           after calibration is fixed
    """
    out_dir = ensure_output_dir(cfg)

    validation_normal_file = out_dir / "validation_normal_scores.csv"
    validation_unlabeled_file = out_dir / "validation_unlabeled_scores.csv"
    predictions_file = out_dir / cfg["outputs"].get(
        "predictions_file",
        "predictions.csv",
    )

    if not validation_normal_file.exists():
        raise FileNotFoundError(
            f"Normal validation score file not found: {validation_normal_file}\n"
            "Run stage: predict first."
        )

    if not validation_unlabeled_file.exists():
        raise FileNotFoundError(
            f"Unlabeled validation score file not found: {validation_unlabeled_file}\n"
            "Run stage: predict first with the unlabeled-GMM calibration code."
        )

    print("=" * 80)
    print("[Stage] POSTHOC ONLY -- UNLABELED FULL-VALIDATION GMM")
    print(f"[Loading NORMAL validation scores] {validation_normal_file}")
    print(f"[Loading UNLABELED validation scores] {validation_unlabeled_file}")
    print("=" * 80)

    val_normal_scores = pd.read_csv(validation_normal_file)
    val_unlabeled_scores = pd.read_csv(validation_unlabeled_file)

    # Ensure saved calibration file has no labels.
    if "label" in val_unlabeled_scores.columns:
        raise ValueError(
            "validation_unlabeled_scores.csv must not contain a label column."
        )

    base_threshold = calibrate_threshold(
        val_normal_scores["score"],
        method=cfg["threshold"]["method"],
        percentile=cfg["threshold"].get("percentile", 95),
        fixed_threshold=cfg["prediction_stage"].get("anomaly_threshold", 0.5),
    )

    posthoc_params = build_unlabeled_posthoc_params(
        cfg,
        val_unlabeled_scores,
        base_threshold=base_threshold,
    )

    posthoc_test_metrics = None

    # Held-out TEST is loaded only after the threshold has been fixed.
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
            "base_threshold": float(base_threshold),
            "posthoc_params": posthoc_params,
            "posthoc_test_metrics": posthoc_test_metrics,
        },
        out_dir / "posthoc_only_results.json",
    )

    print("=" * 80)
    print("[POSTHOC FINISHED]")
    print("[Post-hoc parameters selected WITHOUT validation labels]")
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