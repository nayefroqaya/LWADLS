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

$`\mathrm{score} = \alpha \cdot \mathcal{L}_{\mathrm{MLM}} + \beta \cdot d_{\mathrm{center}}`$

A sequence is classified as anomalous if its score is higher than the calibrated threshold.

## Project Structure

```text
├─ datasets/               # Main entry point for LogSLM datasets  
├─ drain_parser/           # Configuration and parser scripts for Drain  with its references
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


## 📊 Datasets
We used two open-source log datasets (more will be added in the future):

| Software System     | Description                        | Data Size| Link                                         |
|--------------------|------------------------------------|-----------|----------------------------------------|
| HDFS               | Hadoop Distributed File System log | 1.47 GB   | [LogHub](https://github.com/logpai/loghub)   |
| BGL                | Blue Gene/L supercomputer log      | 708.76 MB | [LogHub](https://github.com/logpai/loghub)   |
| Thunderbird (1G)   | Thunderbird supercomputer log      | 1 GB      | [LogHub](https://github.com/logpai/loghub)   |
| Spirit (SP_150MB)  | Supercomputing system log          | 150 MB    | [Figshare](https://figshare.com/s/6d3c6a83f4828d17be79?file=27775929) |


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


