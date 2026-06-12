from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

from .config import dataset_split_path
from .utils import normalize_log_text


def is_normal_label(value: Any, normal_values: List[Any]) -> bool:
    return str(value) in {str(v) for v in normal_values}


def binary_label(value: Any, normal_values: List[Any]) -> int:
    return 0 if is_normal_label(value, normal_values) else 1


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    print(f"[Loading file] {path}")

    if path.suffix.lower() == ".pkl":
        df = pd.read_pickle(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .pkl or .csv")

    print(f"[Loaded] rows={len(df):,}, columns={len(df.columns)}")
    return df


def build_sequences_from_df(df: pd.DataFrame, cfg: Dict[str, Any], dataset_name: str, normal_only: bool = False,
        max_normal_ratio: Optional[float] = None, max_normal_samples: Optional[int] = None,
        seed: int = 42, ) -> pd.DataFrame:
    """
    Build textual event sequences from a dataframe.

    This function does NOT create sliding windows.

    It assumes the dataset has already been preprocessed and that
    Node_block_id represents the sequence/session/window identifier.

    For HDFS:
        Node_block_id = original HDFS block ID.

    For BGL, SP_150MB, TH_1G:
        Node_block_id = precomputed sliding-window/session ID.

    Therefore, all rows with the same Node_block_id are grouped into one
    textual event sequence.
    """

    columns = cfg["columns"]
    labels_cfg = cfg["labels"]
    text_cfg = cfg["text"]

    label_col = columns["label"]
    template_col = text_cfg.get("input_column", columns["template"])

    # Sequence/group identifier.
    sequence_id_col = text_cfg.get("group_by_column", columns.get("block_id"))

    timestamp_col = columns.get("timestamp")

    missing = [c for c in [label_col, template_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Available columns: {list(df.columns)}")

    print(f"[Building sequences] dataset={dataset_name}, normal_only={normal_only}")
    print(f"[Sequence ID column] {sequence_id_col}")
    print("[Note] The code groups all rows with the same sequence ID into one sequence. "
          "It does not create sliding windows here.")

    work = df.copy()

    print("[Label mapping] converting labels to binary normal/anomaly...")
    work["_binary_label"] = work[label_col].apply(lambda x: binary_label(x, labels_cfg["normal_values"]))

    if normal_only:
        before = len(work)
        work = work[work["_binary_label"] == 0].copy()
        after = len(work)
        print(f"[Normal filter] kept {after:,}/{before:,} rows")

    if work.empty:
        raise ValueError(f"No rows left after filtering. dataset={dataset_name}, normal_only={normal_only}")

    if sequence_id_col not in work.columns:
        print(f"[Warning] sequence ID column '{sequence_id_col}' not found. "
              "Each row will be treated as one sequence. "
              "Expected input should contain Node_block_id.")
        work["_sequence_group"] = np.arange(len(work))
        sequence_id_col = "_sequence_group"

    if text_cfg.get("sort_by_timestamp", True) and timestamp_col in work.columns:
        print(f"[Sorting] by {sequence_id_col} and {timestamp_col}")
        work = work.sort_values([sequence_id_col, timestamp_col])
    else:
        print(f"[Sorting] by {sequence_id_col}")
        work = work.sort_values([sequence_id_col])

    event_separator = text_cfg.get("event_separator", " [SEP] ")
    max_events = int(text_cfg.get("max_events_per_sequence", 50))
    min_events = int(text_cfg.get("min_events_per_sequence", 1))

    groups = list(work.groupby(sequence_id_col, sort=False))

    print(f"[Grouping] {len(groups):,} sequences found using column '{sequence_id_col}'")

    rows = []

    for sequence_id, g in tqdm(groups, desc=f"Assembling {dataset_name} sequences from Node_block_id",
            unit="sequence", ):
        events = [normalize_log_text(x, text_cfg) for x in g[template_col].astype(str).tolist()]

        events = [e for e in events if e]

        if len(events) < min_events:
            continue

        # We do NOT create sliding windows here.
        # We only split very long existing sequences into chunks if they exceed
        # max_events_per_sequence.
        for start in range(0, len(events), max_events):
            chunk = events[start:start + max_events]

            if len(chunk) < min_events:
                continue

            g_chunk = g.iloc[start:start + max_events]

            # Sequence label:
            # If any event inside the sequence is anomalous,
            # the whole sequence is labeled as anomaly.
            seq_label = int(g_chunk["_binary_label"].max())

            if normal_only and seq_label != 0:
                continue

            rows.append({"sequence": event_separator.join(chunk), "label": seq_label, "dataset": dataset_name,
                "group_id": str(sequence_id), "num_events": len(chunk), })

    seq_df = pd.DataFrame(rows)

    if seq_df.empty:
        raise ValueError(f"No sequences were built for dataset={dataset_name}.")

    print(f"[Sequences built] dataset={dataset_name}, "
          f"sequences={len(seq_df):,}, "
          f"normal={(seq_df['label'] == 0).sum():,}, "
          f"anomaly={(seq_df['label'] == 1).sum():,}")

    # Few-shot normal sampling for target adaptation.
    # This samples from already-built normal sequences.
    if normal_only and (max_normal_ratio is not None or max_normal_samples is not None):
        rng = np.random.default_rng(seed)
        n = len(seq_df)

        if max_normal_ratio is not None:
            sample_n = max(1, int(round(float(max_normal_ratio) * n)))
        else:
            sample_n = n

        if max_normal_samples is not None:
            sample_n = min(sample_n, int(max_normal_samples))

        sample_n = min(sample_n, n)

        print(f"[Few-shot sampling] dataset={dataset_name}, "
              f"ratio={max_normal_ratio}, max_samples={max_normal_samples}, "
              f"selected={sample_n:,}/{n:,}")

        idx = rng.choice(seq_df.index.to_numpy(), size=sample_n, replace=False)
        seq_df = seq_df.loc[idx].reset_index(drop=True)

    return seq_df.reset_index(drop=True)


def load_sequences_for_dataset(cfg: Dict[str, Any], dataset: str, split: str, normal_only: bool,
        target_normal_ratio: Optional[float] = None, target_normal_max_samples: Optional[int] = None, ) -> pd.DataFrame:
    path = dataset_split_path(cfg, dataset, split)

    print("=" * 80)
    print("[Dataset loading]")
    print(f"dataset     : {dataset}")
    print(f"split       : {split}")
    print(f"normal_only : {normal_only}")
    print(f"path        : {path}")
    print("=" * 80)

    df = load_dataframe(path)

    seq_df = build_sequences_from_df(df=df, cfg=cfg, dataset_name=dataset, normal_only=normal_only,
        max_normal_ratio=target_normal_ratio, max_normal_samples=target_normal_max_samples,
        seed=cfg["experiment"].get("seed", 42), )

    return seq_df


def concat_dataset_splits(cfg: Dict[str, Any], datasets: List[str], split: str, normal_only: bool, ) -> pd.DataFrame:
    frames = []

    for ds in tqdm(datasets, desc=f"Loading source datasets split={split}", unit="dataset", ):
        seq_df = load_sequences_for_dataset(cfg, dataset=ds, split=split, normal_only=normal_only, )
        frames.append(seq_df)

    merged = pd.concat(frames, axis=0).reset_index(drop=True)

    print("=" * 80)
    print("[Merged source datasets]")
    print(f"datasets : {datasets}")
    print(f"split    : {split}")
    print(f"rows     : {len(merged):,}")
    print(merged["dataset"].value_counts())
    print("=" * 80)

    return merged


class LogSequenceDataset(Dataset):
    def __init__(self, seq_df: pd.DataFrame, tokenizer, max_length: int):
        self.df = seq_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        enc = self.tokenizer(str(row["sequence"]), truncation=True, padding=False, max_length=self.max_length,
            return_tensors=None, )

        enc["labels_cls"] = int(row["label"])
        enc["dataset_name"] = str(row["dataset"])
        enc["group_id"] = str(row.get("group_id", idx))
        enc["sequence_text"] = str(row["sequence"])

        return enc


def build_loader_from_sequences(seq_df: pd.DataFrame, tokenizer, max_length: int, batch_size: int,
        mlm_probability: float, shuffle: bool, use_dataset_balanced_sampler: bool = False, ):
    print(f"[DataLoader] samples={len(seq_df):,}, "
          f"batch_size={batch_size}, shuffle={shuffle}, "
          f"balanced_sampler={use_dataset_balanced_sampler}")

    dataset = LogSequenceDataset(seq_df, tokenizer, max_length=max_length)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability, )

    def collate_fn(batch):
        labels_cls = torch.tensor([item.pop("labels_cls") for item in batch], dtype=torch.long, )

        dataset_names = [item.pop("dataset_name") for item in batch]
        group_ids = [item.pop("group_id") for item in batch]
        sequence_texts = [item.pop("sequence_text") for item in batch]

        lm_batch = collator(batch)
        lm_batch["labels_cls"] = labels_cls
        lm_batch["dataset_names"] = dataset_names
        lm_batch["group_ids"] = group_ids
        lm_batch["sequence_texts"] = sequence_texts

        return lm_batch

    sampler = None

    if use_dataset_balanced_sampler and "dataset" in seq_df.columns:
        counts = seq_df["dataset"].value_counts().to_dict()

        weights = seq_df["dataset"].map(lambda d: 1.0 / counts[d]).to_numpy(dtype=np.float64)

        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True, )

        shuffle = False

        print("[Balanced sampler enabled]")
        print(seq_df["dataset"].value_counts())

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler,
        collate_fn=collate_fn, )
