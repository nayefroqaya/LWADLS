
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

print('Nayeffffffffffffffffffffffff-------------------------------------------------------**************xxxxxxxxxxxxx')
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


# ======================================================
# ROBUST SCORE NORMALIZATION HELPERS
# ======================================================
#
# IMPORTANT DESIGN:
#   * NO anomalous validation labels are used.
#   * NO test data/test scores are used to fit normalization or threshold.
#   * MLM loss and center distance are normalized separately using statistics
#     fitted ONLY on NORMAL validation sequences.
#   * Fixed global weights are then used to combine the normalized components.
#   * The ordinary threshold rule from cfg["threshold"] is calibrated on the
#     robust combined NORMAL-validation score.
#
# The combination weights are fixed globally and dataset-independent.
# ======================================================

ROBUST_SCORING_VERSION = "ROBUST_NORMALIZED_SCORE_V1_CLEAN"

# Fixed globally for every dataset. Do not tune these values on test data.
ROBUST_MLM_WEIGHT = 0.5
ROBUST_CENTER_WEIGHT = 0.5
ROBUST_POSITIVE_ONLY = True
ROBUST_EPSILON = 1e-12


def _fit_robust_location_scale(values, epsilon: float):
    """
    Fit a robust location and scale using NORMAL validation values only.

    Main scale       : 1.4826 * MAD
    Fallback 1       : IQR / 1.349
    Fallback 2       : standard deviation
    Final fallback   : 1.0

    The fallbacks only prevent division by zero for nearly constant features.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("Cannot fit robust normalization on an empty score array.")

    median = float(np.median(x))
    raw_mad = float(np.median(np.abs(x - median)))
    scaled_mad = float(1.4826 * raw_mad)

    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr = float(q75 - q25)
    iqr_scale = float(iqr / 1.349) if iqr > 0.0 else 0.0
    std = float(np.std(x))

    if scaled_mad > epsilon:
        scale = scaled_mad
        scale_source = "scaled_mad"
    elif iqr_scale > epsilon:
        scale = iqr_scale
        scale_source = "iqr"
    elif std > epsilon:
        scale = std
        scale_source = "std"
    else:
        scale = 1.0
        scale_source = "constant_fallback"

    return {
        "median": median,
        "raw_mad": raw_mad,
        "scale": float(scale),
        "scale_source": scale_source,
        "q25": float(q25),
        "q75": float(q75),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "n": int(x.size),
    }


def fit_robust_score_normalizer(val_normal_scores: pd.DataFrame) -> dict:
    """
    STEP 3 of prediction:
    Fit robust normalization parameters using NORMAL VALIDATION ONLY.

    Validation labels are not used for optimization. The optional label check
    below is only a safety assertion that the dataframe really contains normal
    samples exclusively.
    """
    required = {"mlm_loss", "center_distance"}
    missing = required - set(val_normal_scores.columns)
    if missing:
        raise ValueError(
            f"Robust normalization is missing required columns: {sorted(missing)}"
        )

    if "label" in val_normal_scores.columns:
        labels = set(
            val_normal_scores["label"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if labels - {0}:
            raise ValueError(
                "Robust normalization received anomalous validation samples. "
                "Calibration must use NORMAL validation samples only."
            )

    eps = ROBUST_EPSILON

    params = {
        "method": "normal_validation_robust_component_normalization",
        "version": ROBUST_SCORING_VERSION,
        "mlm_weight": ROBUST_MLM_WEIGHT,
        "center_weight": ROBUST_CENTER_WEIGHT,
        "positive_only": ROBUST_POSITIVE_ONLY,
        "epsilon": eps,
        "mlm_loss": _fit_robust_location_scale(
            val_normal_scores["mlm_loss"].to_numpy(),
            eps,
        ),
        "center_distance": _fit_robust_location_scale(
            val_normal_scores["center_distance"].to_numpy(),
            eps,
        ),
    }

    print("=" * 80)
    print("[ROBUST SCORE NORMALIZATION - FIT ON NORMAL VALIDATION ONLY]")
    print(f"[Version] {ROBUST_SCORING_VERSION}")
    print(f"normal validation samples : {len(val_normal_scores)}")
    print(f"MLM weight                : {params['mlm_weight']:.6f}")
    print(f"Center weight             : {params['center_weight']:.6f}")
    print(f"Positive deviations only  : {params['positive_only']}")
    print(
        "MLM median / scale       : "
        f"{params['mlm_loss']['median']:.8f} / "
        f"{params['mlm_loss']['scale']:.8f} "
        f"({params['mlm_loss']['scale_source']})"
    )
    print(
        "Center median / scale    : "
        f"{params['center_distance']['median']:.8f} / "
        f"{params['center_distance']['scale']:.8f} "
        f"({params['center_distance']['scale_source']})"
    )
    print("=" * 80)

    return params


def apply_robust_score_normalization(
    score_df: pd.DataFrame,
    robust_params: dict,
) -> pd.DataFrame:
    """
    Apply FROZEN robust-normalization parameters to validation or test scores.

    No fitting or parameter search happens here.
    """
    required = {"mlm_loss", "center_distance"}
    missing = required - set(score_df.columns)
    if missing:
        raise ValueError(
            f"Cannot apply robust scoring; missing columns: {sorted(missing)}"
        )

    out = score_df.copy()

    mlm_median = float(robust_params["mlm_loss"]["median"])
    mlm_scale = float(robust_params["mlm_loss"]["scale"])
    center_median = float(robust_params["center_distance"]["median"])
    center_scale = float(robust_params["center_distance"]["scale"])

    z_mlm = (out["mlm_loss"].to_numpy(dtype=np.float64) - mlm_median) / mlm_scale
    z_center = (
        out["center_distance"].to_numpy(dtype=np.float64) - center_median
    ) / center_scale

    out["robust_z_mlm"] = z_mlm
    out["robust_z_center"] = z_center

    # Anomaly evidence is one-sided: only deviations ABOVE normal median add
    # anomaly evidence. A very low value in one component therefore cannot
    # cancel a strong anomalous deviation in the other component.
    if bool(robust_params.get("positive_only", True)):
        mlm_component = np.maximum(z_mlm, 0.0)
        center_component = np.maximum(z_center, 0.0)
    else:
        mlm_component = z_mlm
        center_component = z_center

    out["robust_mlm_component"] = mlm_component
    out["robust_center_component"] = center_component

    out["score"] = (
        float(robust_params["mlm_weight"]) * mlm_component
        + float(robust_params["center_weight"]) * center_component
    )

    return out


def calibrate_from_normal_validation(cfg, robust_val_normal_scores: pd.DataFrame) -> float:
    """
    STEP 5 of prediction:
    Calibrate the FINAL threshold only from robust NORMAL-validation scores.
    """
    threshold = calibrate_threshold(
        robust_val_normal_scores["score"],
        method=cfg["threshold"]["method"],
        percentile=cfg["threshold"].get("percentile", 95),
        fixed_threshold=cfg["prediction_stage"].get("anomaly_threshold", 0.5),
    )

    print("=" * 80)
    print("[FINAL THRESHOLD - NORMAL VALIDATION ONLY]")
    print(f"threshold = {threshold:.10f}")
    print("No test data/test score/test label was used to select this threshold.")
    print("=" * 80)

    return float(threshold)

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
        normal_only=True,
    )

    val_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["val_split"],
        normal_only=True,
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


def predict_in_domain(cfg):
    """
    In-domain prediction with robust component normalization.

    DATA USAGE:
      STEP 1: Load trained model + normal center/prototypes.
      STEP 2: Load NORMAL validation only.
      STEP 3: Score normal validation to obtain MLM loss + center distance.
      STEP 4: Fit median/MAD normalization on NORMAL validation only.
      STEP 5: Combine normalized components with FIXED global weights.
      STEP 6: Calibrate threshold on the robust NORMAL-validation score only.
      STEP 7: Freeze normalizer + weights + threshold.
      STEP 8: ONLY NOW load and score the held-out test set.
      STEP 9: Apply frozen robust scoring to test.
      STEP 10: Use test labels only for final evaluation.

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
    print(f"[Scoring version] {ROBUST_SCORING_VERSION}")
    print("=" * 80)

    # ==============================================================
    # STEP 1 - LOAD THE TRAINED NORMALITY MODEL
    # ==============================================================
    print("[STEP 1/10] Load trained model, tokenizer, and normal center/prototypes")
    model, tokenizer, center = load_saved_model_tokenizer_center(
        cfg=cfg,
        out_dir=out_dir,
        device=device,
        center_filename="normal_center.pt",
    )

    # ==============================================================
    # STEP 2 - LOAD NORMAL VALIDATION ONLY
    # ==============================================================
    print("[STEP 2/10] Load NORMAL validation samples only")
    val_normal_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["val_split"],
        normal_only=True,
    )
    print(f"[Normal validation sequences] {len(val_normal_df)}")

    val_normal_loader = make_loader(
        cfg,
        val_normal_df,
        tokenizer,
        shuffle=False,
    )

    # ==============================================================
    # STEP 3 - SCORE NORMAL VALIDATION COMPONENTS
    # ==============================================================
    print("[STEP 3/10] Score NORMAL validation: MLM loss + center distance")
    # score_loader provides the two raw components needed below.
    # Its temporary combined score is overwritten by robust normalization.
    val_normal_raw_scores = score_loader(
        model,
        val_normal_loader,
        center,
        device,
        alpha_mlm=ROBUST_MLM_WEIGHT,
        beta_center=ROBUST_CENTER_WEIGHT,
        desc="Scoring NORMAL validation components",
    )

    # ==============================================================
    # STEP 4 - FIT ROBUST NORMALIZATION ON NORMAL VALIDATION ONLY
    # ==============================================================
    print("[STEP 4/10] Fit robust median/MAD normalization on NORMAL validation")
    robust_params = fit_robust_score_normalizer(val_normal_raw_scores)

    # Apply fitted normalizer back to the normal validation set.
    robust_val_normal_scores = apply_robust_score_normalization(
        val_normal_raw_scores,
        robust_params,
    )

    # ==============================================================
    # STEP 5 - CALIBRATE FINAL THRESHOLD ON ROBUST NORMAL SCORES
    # ==============================================================
    print("[STEP 5/10] Calibrate threshold from robust NORMAL-validation scores")
    robust_threshold = calibrate_from_normal_validation(
        cfg,
        robust_val_normal_scores,
    )

    robust_params["threshold"] = float(robust_threshold)
    robust_params["threshold_method"] = cfg["threshold"]["method"]
    robust_params["threshold_percentile"] = float(
        cfg["threshold"].get("percentile", 95)
    )

    save_json(robust_params, out_dir / "robust_scoring_params.json")
    robust_val_normal_scores.to_csv(
        out_dir / "robust_normal_validation_scores.csv",
        index=False,
    )

    # ==============================================================
    # STEP 6 - FREEZE EVERYTHING BEFORE TEST
    # ==============================================================
    print("[STEP 6/10] FREEZE normalizer, fixed weights, and threshold")
    print(
        f"[Frozen robust scoring] mlm_weight={robust_params['mlm_weight']:.3f}, "
        f"center_weight={robust_params['center_weight']:.3f}, "
        f"threshold={robust_threshold:.10f}"
    )
    print("[TEST HAS NOT BEEN LOADED YET]")

    # ==============================================================
    # STEP 7 - ONLY NOW LOAD THE HELD-OUT TEST SET
    # ==============================================================
    print("[STEP 7/10] Load held-out TEST only after calibration is frozen")
    test_df = load_sequences_for_dataset(
        cfg,
        dataset=dataset_name,
        split=mode_cfg["test_split"],
        normal_only=mode_cfg.get("test_normal_only", False),
    )
    print(f"[Test sequences] {len(test_df)}")

    test_loader = make_loader(
        cfg,
        test_df,
        tokenizer,
        shuffle=False,
    )

    # ==============================================================
    # STEP 8 - SCORE TEST COMPONENTS (NO FITTING)
    # ==============================================================
    print("[STEP 8/10] Score TEST components with frozen model")
    test_raw_scores = score_loader(
        model,
        test_loader,
        center,
        device,
        alpha_mlm=ROBUST_MLM_WEIGHT,
        beta_center=ROBUST_CENTER_WEIGHT,
        desc="Scoring held-out TEST components",
    )

    # ==============================================================
    # STEP 9 - APPLY FROZEN ROBUST NORMALIZATION TO TEST
    # ==============================================================
    print("[STEP 9/10] Apply FROZEN robust normalization and fixed weights to TEST")
    robust_test_scores = apply_robust_score_normalization(
        test_raw_scores,
        robust_params,
    )
    robust_test_scores["prediction"] = (
        robust_test_scores["score"] > robust_threshold
    ).astype(int)

    # ==============================================================
    # STEP 10 - FINAL TEST EVALUATION ONLY
    # ==============================================================
    print("[STEP 10/10] Use TEST labels only for final metrics")
    metrics = evaluate_scores(
        robust_test_scores,
        robust_threshold,
        normal_label=0,
    )

    save_classification_report_files(
        metrics,
        out_dir,
        prefix="test",
    )

    robust_test_scores.to_csv(
        out_dir / cfg["outputs"].get("predictions_file", "predictions.csv"),
        index=False,
    )

    save_json(
        {
            "stage": "predict",
            "mode": "in_domain",
            "dataset": dataset_name,
            "metrics": metrics,
            "robust_scoring": robust_params,
            "robust_threshold": float(robust_threshold),
        },
        out_dir / cfg["outputs"].get("results_file", "results.json"),
    )

    print("=" * 80)
    print("[IN-DOMAIN PREDICTION FINISHED]")
    print("[Robust-Normalized FINAL TEST Classification Report]")
    print(metrics["classification_report_text"])
    print("=" * 80)


def run_in_domain(cfg, config_path: str):
    """
    Full in-domain pipeline:
        train
        predict
    """

    train_in_domain(cfg, config_path)
    predict_in_domain(cfg)


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
        normal_only=True,
        target_normal_ratio=mode_cfg.get("target_normal_ratio"),
        target_normal_max_samples=mode_cfg.get("target_normal_max_samples"),
    )

    target_val_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_val_split"],
        normal_only=True,
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


def predict_fewshot_target_adaptation(cfg):
    """
    Few-shot target prediction with the same robust normalization protocol.

    All robust-normalization statistics and the threshold are fitted only on
    NORMAL samples from the TARGET validation split. The target test split is
    not loaded until those values are frozen.
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
    print(f"[Scoring version] {ROBUST_SCORING_VERSION}")
    print("=" * 80)

    # ==============================================================
    # STEP 1 - LOAD TARGET-ADAPTED MODEL
    # ==============================================================
    print("[STEP 1/10] Load target-adapted model, tokenizer, and target center")
    model, tokenizer, target_center = load_saved_model_tokenizer_center(
        cfg=cfg,
        out_dir=out_dir,
        device=device,
        center_filename="target_normal_center.pt",
    )

    # ==============================================================
    # STEP 2 - LOAD TARGET NORMAL VALIDATION ONLY
    # ==============================================================
    print("[STEP 2/10] Load TARGET NORMAL validation samples only")
    target_val_normal_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_val_split"],
        normal_only=True,
    )
    print(f"[Target normal validation sequences] {len(target_val_normal_df)}")

    target_val_normal_loader = make_loader(
        cfg,
        target_val_normal_df,
        tokenizer,
        shuffle=False,
    )

    # ==============================================================
    # STEP 3 - SCORE TARGET NORMAL VALIDATION COMPONENTS
    # ==============================================================
    print("[STEP 3/10] Score TARGET NORMAL validation: MLM loss + center distance")
    val_normal_raw_scores = score_loader(
        model,
        target_val_normal_loader,
        target_center,
        device,
        alpha_mlm=ROBUST_MLM_WEIGHT,
        beta_center=ROBUST_CENTER_WEIGHT,
        desc="Scoring TARGET NORMAL validation components",
    )

    # ==============================================================
    # STEP 4 - FIT ROBUST NORMALIZATION ON TARGET NORMAL VALIDATION
    # ==============================================================
    print("[STEP 4/10] Fit robust median/MAD normalization on TARGET NORMAL validation")
    robust_params = fit_robust_score_normalizer(val_normal_raw_scores)
    robust_val_normal_scores = apply_robust_score_normalization(
        val_normal_raw_scores,
        robust_params,
    )

    # ==============================================================
    # STEP 5 - CALIBRATE TARGET THRESHOLD ON ROBUST NORMAL SCORES
    # ==============================================================
    print("[STEP 5/10] Calibrate target threshold from robust NORMAL-validation scores")
    robust_threshold = calibrate_from_normal_validation(
        cfg,
        robust_val_normal_scores,
    )

    robust_params["threshold"] = float(robust_threshold)
    robust_params["threshold_method"] = cfg["threshold"]["method"]
    robust_params["threshold_percentile"] = float(
        cfg["threshold"].get("percentile", 95)
    )
    robust_params["target_dataset"] = target_dataset

    save_json(robust_params, out_dir / "robust_scoring_params.json")
    robust_val_normal_scores.to_csv(
        out_dir / "robust_normal_validation_scores.csv",
        index=False,
    )

    # ==============================================================
    # STEP 6 - FREEZE EVERYTHING BEFORE TARGET TEST
    # ==============================================================
    print("[STEP 6/10] FREEZE target normalizer, fixed weights, and threshold")
    print(
        f"[Frozen robust scoring] mlm_weight={robust_params['mlm_weight']:.3f}, "
        f"center_weight={robust_params['center_weight']:.3f}, "
        f"threshold={robust_threshold:.10f}"
    )
    print("[TARGET TEST HAS NOT BEEN LOADED YET]")

    # ==============================================================
    # STEP 7 - ONLY NOW LOAD TARGET TEST
    # ==============================================================
    print("[STEP 7/10] Load held-out TARGET TEST only after calibration is frozen")
    target_test_df = load_sequences_for_dataset(
        cfg,
        dataset=target_dataset,
        split=mode_cfg["target_test_split"],
        normal_only=mode_cfg.get("target_test_normal_only", False),
    )
    print(f"[Target test sequences] {len(target_test_df)}")

    target_test_loader = make_loader(
        cfg,
        target_test_df,
        tokenizer,
        shuffle=False,
    )

    # ==============================================================
    # STEP 8 - SCORE TARGET TEST COMPONENTS
    # ==============================================================
    print("[STEP 8/10] Score TARGET TEST components with frozen model")
    test_raw_scores = score_loader(
        model,
        target_test_loader,
        target_center,
        device,
        alpha_mlm=ROBUST_MLM_WEIGHT,
        beta_center=ROBUST_CENTER_WEIGHT,
        desc="Scoring held-out TARGET TEST components",
    )

    # ==============================================================
    # STEP 9 - APPLY FROZEN ROBUST NORMALIZATION TO TARGET TEST
    # ==============================================================
    print("[STEP 9/10] Apply FROZEN robust normalization and fixed weights to TARGET TEST")
    robust_test_scores = apply_robust_score_normalization(
        test_raw_scores,
        robust_params,
    )
    robust_test_scores["prediction"] = (
        robust_test_scores["score"] > robust_threshold
    ).astype(int)

    # ==============================================================
    # STEP 10 - FINAL TARGET TEST EVALUATION ONLY
    # ==============================================================
    print("[STEP 10/10] Use TARGET TEST labels only for final metrics")
    metrics = evaluate_scores(
        robust_test_scores,
        robust_threshold,
        normal_label=0,
    )

    save_classification_report_files(
        metrics,
        out_dir,
        prefix="test",
    )
    robust_test_scores.to_csv(
        out_dir / cfg["outputs"].get("predictions_file", "predictions.csv"),
        index=False,
    )

    save_json(
        {
            "stage": "predict",
            "mode": "fewshot_target_adaptation",
            "target_dataset": target_dataset,
            "metrics": metrics,
            "robust_scoring": robust_params,
            "robust_threshold": float(robust_threshold),
        },
        out_dir / cfg["outputs"].get("results_file", "results.json"),
    )

    print("=" * 80)
    print("[FEW-SHOT TARGET ADAPTATION PREDICTION FINISHED]")
    print("[Robust-Normalized FINAL TARGET TEST Classification Report]")
    print(metrics["classification_report_text"])
    print("=" * 80)


def run_fewshot_target_adaptation(cfg, config_path: str):
    """
    Full few-shot target adaptation pipeline:
        train
        predict
    """

    train_fewshot_target_adaptation(cfg, config_path)
    predict_fewshot_target_adaptation(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[RUN.PY PATH] {Path(__file__).resolve()}")
    print(f"[Scoring implementation] {ROBUST_SCORING_VERSION}")
    mode = cfg["experiment"]["mode"]
    stage = cfg["experiment"].get("stage", "train_predict")

    if mode == "in_domain":
        if stage == "train":
            train_in_domain(cfg, args.config)
        elif stage == "predict":
            predict_in_domain(cfg)
        elif stage == "train_predict":
            run_in_domain(cfg, args.config)
        else:
            raise ValueError(
                f"Unsupported stage for in_domain: {stage}. "
                "Use 'train', 'predict', or 'train_predict'."
            )

    elif mode == "fewshot_target_adaptation":
        if stage == "train":
            train_fewshot_target_adaptation(cfg, args.config)
        elif stage == "predict":
            predict_fewshot_target_adaptation(cfg)
        elif stage == "train_predict":
            run_fewshot_target_adaptation(cfg, args.config)
        else:
            raise ValueError(
                f"Unsupported stage for fewshot_target_adaptation: {stage}. "
                "Use 'train', 'predict', or 'train_predict'."
            )
    else:
        raise ValueError(
            f"Unsupported mode: {mode}. "
            "Use 'in_domain' or 'fewshot_target_adaptation'."
        )


if __name__ == "__main__":
    main()