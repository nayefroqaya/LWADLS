# FSADLS

FSADLS is a lightweight semi-supervised Small Language Model (SLM) framework for log anomaly detection. It learns normal log behavior by fine-tuning an SLM on normal log sequences and represents normality through multiple prototype embeddings. During prediction, anomalies are identified using a hybrid score that combines masked language modeling (MLM) loss with the distance to the nearest normal prototype. The framework supports both in-domain detection and few-shot cross-domain adaptation, with optional supervised post-hoc calibration to refine the score weights and decision threshold.

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

The FSADLS framework consists of three main stages:

1. **Semi-supervised SLM training and few-shot adaptation**  
   The model learns normal log behavior from normal training sequences and, in cross-domain settings, adapts to the target domain using a limited amount of normal target data.

2. **Prediction and anomaly scoring**  
   Each log sequence is scored using a combination of MLM loss and nearest-prototype distance.

3. **Supervised post-hoc calibration**  
   A limited labeled validation subset is used to refine the score weights and decision threshold before final evaluation on the held-out test set.

## Main Features

- Normal-only semi-supervised SLM training.
- Fine-tuning of a masked language model for contextual log representation learning.
- Multi-prototype normality modeling.
- Hybrid anomaly scoring based on MLM loss and nearest-prototype distance.
- Percentile-based base threshold estimation from normal validation data.
- Supervised post-hoc calibration of score weights and decision thresholds using a limited labeled validation subset.
- Support for both in-domain detection and few-shot cross-domain adaptation.
## Method Summary

FSADLS learns normal log behavior from normal training sequences. During prediction, each test sequence receives an anomaly score based on:

1. **MLM loss**: measures token-level irregularity in the log sequence.
2. **Nearest-prototype distance**: measures representation-level deviation from learned normal behavior.

The anomaly score is computed as:

$`\mathrm{score} = \alpha \cdot \mathcal{L}_{\mathrm{MLM}} + \beta \cdot d_{\mathrm{center}}`$

A sequence is classified as anomalous if its score is higher than the calibrated threshold.

## Project Structure

```text
FSADLS/
├── datasets/                         # Log datasets and train/validation/test splits
├── drain_parser/                     # Drain parser and related configuration
├── configs/
│   ├── fsadls_unified_config.yml
│   └── fsadls_normal_anomaly_adapt_config.yml
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   ├── data.py                       # Dataset loading and preprocessing
│   ├── model.py                      # SLM architecture
│   ├── train.py                      # Training and few-shot adaptation
│   ├── losses.py                     # Training objectives
│   ├── scoring.py                    # Anomaly scoring and post-hoc calibration
│   ├── run.py                        # Main FSADLS pipeline
│   └── utils.py                      # Utility functions
├── main.py                           # Main entry point
├── requirements.txt                  # Required Python packages
├── README.md                         # Documentation
└── .gitignore
```


## 📊 Datasets
We used two open-source log datasets (more will be added in the future):

| Software System     | Description                        | Data Size| Link                                         |
|--------------------|------------------------------------|-----------|----------------------------------------|
| HDFS               | Hadoop Distributed File System log | 1.47 GB   | [LogHub](https://github.com/logpai/loghub)   |
| BGL                | Blue Gene/L supercomputer log      | 708.76 MB | [LogHub](https://github.com/logpai/loghub)   |
| Thunderbird (1G)   | Thunderbird supercomputer log      | 1 GB      | [LogHub](https://github.com/logpai/loghub)   |
| Spirit (SP_150MB)  | Supercomputing system log          | 150 MB    | [Figshare](https://figshare.com/s/6d3c6a83f4828d17be79?file=27775929) |




## Dataset Statistics

| Dataset | Size | Logs | Anomaly Ratio (%) | Blocks | Normal Sequences | Anomalous Sequences |
|---|---:|---:|---:|---:|---:|---:|
| Spirit_a (S) | 150 MB | 1,225,059 | 2.6 | 7,590 | 6,266 | 1,324 |
| BGL (B) | 708.7 MB | 4,747,963 | 7.39 | 84,926 | 49,247 | 36,251 |
| TH_a (T) | 1 GB | 6,013,029 | 5.04 | 52,333 | 44,732 | 7,601 |
| HDFS (H) | 1.47 GB | 11,175,629 | 2.5 | 575,071 | 558,223 | 16,838 |


## Evaluation Scenarios

B, H, T, and S denote **BGL**, **HDFS**, **TH_a**, and **SP_a**, respectively.  
1S-CD, 2S-CD, and 3S-CD denote cross-domain transfer from one, two, and three source domains.

| Setting | Scenario | Source → Target |
|---|---|---|
| In-domain | B → B | BGL → BGL |
| In-domain | H → H | HDFS → HDFS |
| In-domain | T → T | TH_a → TH_a |
| In-domain | S → S | SP_a → SP_a |
| 1S-CD | 1S₁ | B → H |
| 1S-CD | 1S₂ | B → T |
| 1S-CD | 1S₃ | T → S |
| 2S-CD | 2S₁ | (B + H) → T |
| 2S-CD | 2S₂ | (H + T) → B |
| 2S-CD | 2S₃ | (B + T) → S |
| 3S-CD | 3S₁ | (B + H + T) → S |
| 3S-CD | 3S₂ | (B + H + S) → T |
| 3S-CD | 3S₃ | (H + T + S) → B |
---

## ⚙️ Environment
All libraries are specified with their versions in the requirements file (e.g., Main path/requirements.txt).

```bash
pip install -r requirements.txt
```

---
## 🛠️ Preparation - Parsing step:
Steps to run LogSLM:

1. Install all required libraries from the requirements file (e.g., Main path/requirements.txt).
2. Create a dataset directory under `datasets` (e.g., `HDFS`, `BGL`,`TH_1G`, `SP_150MB`) and upload the (datasetname.log) to this directory.
3. In main.py, set the dataset name (e.g., `HDFS`, `BGL`,`TH_1G`, `SP_150MB`)
4. For Drain parser details, see [IBM Drain](https://github.com/logpai/logparser/tree/main/logparser/Drain).
5. The parsing code is available in the `drain_parser` folder.
6. Specify the dataset name in `demo.py` (e.g., BGL). The code is available for all datasets. Uncomment the lines of the dataset you need to use
7. For data parsing, all libraries are specified with their versions in the requirements file (e.g., drain_parser/requirements.txt). 
8. To start the parsing process, run (drain_parser/demo.py). 
9. The parsing output will be generated and saved in the datasets' directory.
10. The output of Drain is CSV file. 

---
## 🛠️ Preparation - Data Splitting  step:

1. After the parsing, we run load_datalog.py
2. In main.py: uncomment Section A and run the file. The results will be  60%, 10%, 30% splits as PKL files.
3. In main.py: comment the Section A and run the file. This is start point of the pipeline.


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


## Running FSADLS

FSADLS supports three main experimental settings:

1. In-domain anomaly detection
2. Cross-domain adaptation using normal target data only (NOFS)
3. Cross-domain adaptation using normal and anomalous target data (MLFS)

### 1. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt

### In-Domain Detection
```main.py
src/run.py
configs/adalogslm_unified_config.yml
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


