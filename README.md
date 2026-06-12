# LogSLM

LogSLM is a lightweight semi-supervised Small Language Model (SLM) framework for log anomaly detection. It fine-tunes a MiniLM-based masked language model on normal log sequences and detects anomalies using a hybrid score that combines masked language modeling loss and distance to normal prototype embeddings.

## Overview

LogSLM supports two main experimental settings:

1. **In-domain anomaly detection**
   - Train and test on the same dataset.
   - Example: BGL Training set → BGL Testing set.

2. **Few-shot target adaptation**
   - Train on normal sequences from one or more source datasets.
   - Adapt using a small number of normal target-domain sequences.
   - Evaluate on the target test set.
   - Example: (BGL) Training set → HDFS Testing set. (Use Fraction)
   - Example: (BGL + HDFS) Training sets → Thunderbird Testing set. (Use Fraction)
   - Example: (BGL + HDFS + Spirit) Training sets → Thunderbird Testing set. (Use Fraction)


The framework has three stages:

1. **Semi-supervised SLM training**
2. **Prediction**
3. **Post-hoc calibration**

## Main Features

- Semi-supervised training using normal sequences only.
- Fine-tuning of a MiniLM-based masked language model.
- Center/prototype-based normality modeling.
- Hybrid anomaly score combining MLM loss and prototype distance.
- Percentile-based threshold calibration.
- Optional post-hoc grid search for score weights and thresholds.
- Support for in-domain and few-shot cross-dataset settings.

## Method Summary

LogSLM learns normal log behavior from normal sequences only. During prediction, each test sequence receives an anomaly score based on:

1. **MLM loss**: token-level irregularity.
2. **Prototype distance**: representation-level deviation from normal behavior.

The anomaly score is computed as:

```math
score = \alpha \cdot MLM\_loss + \beta \cdot center\_distance
```

A sequence is classified as anomalous if its score is higher than the calibrated threshold.

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── configs/
│   └── adalogslm_unified_config.yml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── losses.py
│   ├── scoring.py
│   ├── run.py
│   └── utils.py
└── README.md
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
torch
transformers
scikit-learn
pandas
numpy
PyYAML
tqdm
```

## Data Format

The input data should be provided as preprocessed `.pkl` or `.csv` split files.

The required columns are configured in the YAML file:

```yaml
columns:
  timestamp: "Timestamp"
  template: "processed_EventTemplate"
  block_id: "Node_block_id"
  label: "Label"
  dataset: "DatasetName"
```

Log events are grouped by `Node_block_id` to form log sequences. Event templates are sorted by timestamp, concatenated using `[SEP]`, and then tokenized before being passed to the SLM.

Example split files:

```text
../datasets/BGL/1_BGL_Splitted_Datasets/train_df.pkl
../datasets/BGL/1_BGL_Splitted_Datasets/val_df.pkl
../datasets/BGL/1_BGL_Splitted_Datasets/test_df.pkl
```

## Configuration

The main configuration file is:

```text
configs/adalogslm_unified_config.yml
```

To run in-domain detection:

```yaml
experiment:
  mode: "in_domain"
```

To run few-shot target adaptation:

```yaml
experiment:
  mode: "fewshot_target_adaptation"
```

Supported stages:

```text
train
predict
posthoc_only
train_predict
```

Example:

```yaml
experiment:
  mode: "fewshot_target_adaptation"
  stage: "train_predict"
```

## Running the Code

Run with the default configuration:

```bash
python main.py
```

## Notes

- Training is semi-supervised and uses normal sequences only.
- Target anomaly labels are not used during few-shot adaptation.
- Normal validation sequences are used to compute normal prototype(s) and calibrate the anomaly threshold.
- Test labels are used only for evaluation and offline post-hoc calibration.
- Post-hoc calibration changes only the scoring weights and threshold; it does not update the model.


