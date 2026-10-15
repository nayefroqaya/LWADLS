
# FSADLS: Adaptive Anomaly Detection via Few-Shot Learning and Calibration


**FSADLS** is a lightweight, semi-supervised framework for system-log anomaly detection using a small language model (SLM). It learns representations of normal log behavior through masked language modeling and multiple normality prototypes. At inference time, it combines masked language modeling (MLM) loss and distance to the nearest normal prototype to identify anomalous sequences. The framework supports in-domain detection and few-shot cross-domain adaptation, with supervised post-hoc calibration of score weights and the decision threshold.

## Method Overview

The framework comprises three stages:

1. **Normality learning and adaptation.** Fine-tune an SLM on normal training sequences. For cross-domain experiments, optionally adapt using a limited fraction of target-domain sequences.
2. **Anomaly scoring.** Combine MLM loss and distance to the nearest learned normal prototype:

   $$
   s(x) = \alpha\,\mathcal{L}_{\mathrm{MLM}}(x) + \beta\,d_{\mathrm{nearest\ prototype}}(x)
   $$

3. **Supervised Post-hoc calibration.** Estimate a base threshold using normal validation data; refine the scoring weights and threshold on a small labeled validation subset. A sequence is predicted anomalous when its score exceeds the selected threshold.

Calibration changes the scoring rule, **not** the SLM parameters.

## Repository Structure

```text
LWADLS/
├── datasets/                         # Log datasets and train/validation/test splits
├── drain_parser/                     # Drain parser and related configuration
=======
FSADLS/
├── datasets/                         # Raw logs and prepared dataset splits
├── drain_parser/                     # Drain parsing implementation
├── configs/
│   ├── fsadls_unified_config.yml
│   └── fsadls_normal_anomaly_adapt_config.yml
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration loading
│   ├── data.py                       # Data preparation
│   ├── model.py                      # SLM architecture
│   ├── train.py                      # Training and adaptation
│   ├── losses.py                     # Training objectives
│   ├── scoring.py                    # Anomaly scoring and calibration
│   ├── run.py                        # Experiment pipeline
│   └── utils.py                      # Utilities
├── main.py                           # Entry point
├── requirements.txt                  # Python dependencies
├── README.md
└── .gitignore
```

## Datasets

The experiments use four publicly available system-log datasets.

| Dataset | Description | Size | Source |
|---|---|---:|---|
| HDFS | Hadoop Distributed File System | 1.47 GB | [LogHub](https://github.com/logpai/loghub) |
| BGL | Blue Gene/L supercomputer | 708.76 MB | [LogHub](https://github.com/logpai/loghub) |
| Thunderbird (`TH_1G`) | Thunderbird supercomputer | 1 GB | [LogHub](https://github.com/logpai/loghub) |
| Spirit (`SP_150MB`) | Supercomputing system | 150 MB | [Figshare](https://figshare.com/s/6d3c6a83f4828d17be79?file=27775929) |

### Dataset Statistics

| Dataset | Log messages | Anomaly ratio (%) | Reported blocks/windows | Normal sequences | Anomalous sequences |
|---|---:|---:|---:|---:|---:|
| Spirit_a (S) | 1,225,059 | 2.6 | 7,590 | 6,266 | 1,324 |
| BGL (B) | 4,747,963 | 7.39 | 84,926 | 49,247 | 36,251 |
| TH_a (T) | 6,013,029 | 5.04 | 52,333 | 44,732 | 7,601 |
| HDFS (H) | 11,175,629 | 2.5 | 575,071 | 558,223 | 16,838 |

*The anomaly ratio refers to reported log-level statistics, while the final two columns report sequence-level labels. Statistics are reproduced from the project documentation and should be checked against the generated splits.*

## Experimental Settings

FSADLS evaluates **in-domain** detection and **cross-domain** adaptation. The cross-domain settings are grouped by the number of source datasets: one (1S-CD), two (2S-CD), or three (3S-CD).

| Setting | Scenario | Source → Target |
|---|---|---|
| In-domain | B → B | BGL → BGL |
| In-domain | H → H | HDFS → HDFS |
| In-domain | T → T | Thunderbird → Thunderbird |
| In-domain | S → S | Spirit → Spirit |
| 1S-CD | 1S₁ | B → H |
| 1S-CD | 1S₂ | B → T |
| 1S-CD | 1S₃ | T → S |
| 2S-CD | 2S₁ | (B + H) → T |
| 2S-CD | 2S₂ | (H + T) → B |
| 2S-CD | 2S₃ | (B + T) → S |
| 3S-CD | 3S₁ | (B + H + T) → S |
| 3S-CD | 3S₂ | (B + H + S) → T |
| 3S-CD | 3S₃ | (H + T + S) → B |

In cross-domain experiments, the target-domain adaptation mode can be:

- **NOFS:** Normal-only few-shot adaptation using a specified fraction of target training sequences.
- **MLFS:** Mixed-label few-shot adaptation, with limited normal and anomalous target training sequences. Their binary labels are not used as discriminative targets during SLM adaptation; labeled validation data may be used for calibration.

## Installation and Data Preparation

### 1. Install Dependencies

From the repository root: pip install -r requirements.txt


## 🛠️ Preparation - Parsing step:
Steps to run FSADLS:

1. Install all required libraries from the requirements file (e.g., Main path/requirements.txt).
2. Create a dataset directory under `datasets` (e.g., `HDFS`, `BGL`,`TH_1G`, `SP_150MB`) and upload the (datasetname.log) to this directory.
3. In main.py, set the dataset name (e.g., `HDFS`, `BGL`,`TH_1G`, `SP_150MB`)
4. For Drain parser details, see [IBM Drain](https://github.com/logpai/logparser/tree/main/logparser/Drain).
5. The parsing code is available in the `drain_parser` folder in our project.
6. Specify the dataset name in `demo.py` (e.g., BGL). The code is available for all datasets. Uncomment the lines of the dataset you need to use
7. For data parsing, all libraries are specified with their versions in the requirements file (e.g., drain_parser/requirements.txt). 
8. To start the parsing process, run (drain_parser/demo.py). 
9. The parsing output will be generated and saved in the datasets' directory.
10. The output of Drain is CSV file. 
=======
### 2. Parse Raw Logs

Create a dataset folder, such as `datasets/BGL/`, and place its raw `.log` file inside. The project uses the [Drain parser](https://github.com/logpai/logparser/tree/main/logparser/Drain).

1. Select the dataset in `drain_parser/demo.py` by enabling its corresponding configuration.
2. Install parser-specific requirements if needed: `pip install -r drain_parser/requirements.txt`.
3. Run:

<<<<<<< :
1. After the parsing, we run load_datalog.py
2. In main.py: uncomment Section A and run the file: python main.py. The results will be  60%, 10%, 30% splits as PKL files.
3. In main.py: comment the Section A and run the file: python main.py. This is start point of the pipeline.
=======
   ```bash
   python drain_parser/demo.py
   ```

The parser writes structured CSV output to the dataset directory.

### 3. Create Training, Validation, and Test Splits

The preprocessing pipeline uses `load_datalog.py` and **Section A** of `main.py`:

The required columns are configured in the YAML file root/configs/ filename.yml:

1. Enable Section A in `main.py` and execute `python main.py` to prepare the data.
2. The resulting split is **60% training, 10% validation, and 30% testing**, saved as pickle files.
3. Disable Section A before running model experiments.

Example for BGL (round 1):

```text
datasets/BGL/1_BGL_Splitted_Datasets/
├── train_df.pkl
├── val_df.pkl
└── test_df.pkl
```

### Input Data Format

Column names are specified in YAML. The supplied configuration example is:

```yaml
columns:
  timestamp: "Timestamp"
  template: "processed_EventTemplate"
  block_id: "Node_block_id"
  label: "Label"
  dataset: "DatasetName"
```

Log events are grouped into sequences using `Node_block_id`, ordered by timestamp, joined using `[SEP]`, and tokenized for the SLM. The precise grouping procedure depends on the preprocessing configuration.

 
Example split files:
First run with Round =1 
```text
../datasets/BGL/1_BGL_Splitted_Datasets/train_df.pkl
../datasets/BGL/1_BGL_Splitted_Datasets/val_df.pkl
../datasets/BGL/1_BGL_Splitted_Datasets/test_df.pkl

'datasets' is the name of the folder. For example: BGL is the name of the dataset we use.
 1_BGL_Splitted_Datasets is a folder which includes the PKL files that we generated  from (Preparation - Parsing step) and (Preparation - Data Splitting  step) sections.
```


## Running FSADLS Experiments 

FSADLS supports three main experimental settings:

1. In-domain anomaly detection
2. Cross-domain adaptation using normal target data only (NOFS)
3. Cross-domain adaptation using normal and anomalous target data (MLFS)

### Install Dependencies and run the experiments 

### pip install -r requirements.txt
=======
## Running Experiments

After preparing the datasets, choose the experiment mode and configuration, then run the entry point:

```bash
python main.py
```

The original documentation also describes `--config` as a command-line option; use it if supported by your version of `main.py`. Verify the configuration filename against the actual files in `configs/` because the original README uses both `fsadls_...` and `adalogslm_...` names.

### In-Domain Detection

Set the following experiment mode in the relevant YAML configuration:

```yaml
experiment:
  mode: "in_domain"
```

### Cross-Domain Adaptation: NOFS

Normal-only target adaptation, using 20% of target normal training sequences in the documented example:

```yaml
experiment:
  mode: "fewshot_target_adaptation"
  stage: "train_predict"

datasets:
  fewshot_target_adaptation:
    target_adapt_normal_only: true
    target_normal_ratio: 0.20
```

### Cross-Domain Adaptation: MLFS

The documented mixed-label adaptation example uses 20% of target normal and 20% of target anomalous training sequences:

```yaml
experiment:
  mode: "fewshot_target_adaptation"
  stage: "train_predict"

datasets:
  fewshot_target_adaptation:
    target_adapt_normal_only: false
    target_adapt_normal_fraction: 0.20
    target_adapt_anomaly_fraction: 0.20
```

**Entry-point note:** The supplied instructions mention selecting either `src/run.py` or `run_nom_anom_train_frac.py` through imports in `main.py`. Confirm which runner and configuration are paired with each adaptation mode in the current code before execution.

## Training and Evaluation Protocol

- The base SLM is trained on normal training sequences.
- NOFS adaptation uses normal target training sequences only.
- MLFS adaptation can expose limited normal and anomalous target training sequences, without using binary anomaly labels as discriminative SLM training targets.
- Normal validation sequences support normality-prototype construction and initial threshold estimation.
- A limited labeled validation subset can calibrate the anomaly-score weights and decision threshold.
- The held-out test set is reserved for final evaluation and is not used for training, adaptation, or calibration.

 
## Baselines - Re-reproduced

These approaches should read the data from our data path : For example : FSADLS/datasets/BGL. Inside BGL, you can see :
- BGL/1_BGL_Splitted_Datasets/train_df.pkl
- BGL/1_BGL_Splitted_Datasets/val_df.pkl
- BGL/1_BGL_Splitted_Datasets/test_df.pkl

##======================================X

#From the main forder of each baseline, you should read the datasets folder in our project directory.

- LogAnomaly: [Code](https://github.com/nayefroqaya/Exper_LOAGAD/tree/main-before-2-months)
- DeepLog: [Code](https://github.com/nayefroqaya/Exper_LOAGAD/tree/main-before-2-months)
- LogRobust: [Code](https://github.com/nayefroqaya/Exper_LOAGAD/tree/main-before-2-months)
- NeuralLog: [Code](https://github.com/nayefroqaya/Exper_LOAGAD/tree/main-before-2-months)
  
In the file Main_run.py, there are the paths to data for LogAnomaly, Deeplog, LogRobust and NeuralLog. For Example: 

    file_path_train = '../root forlder/datasets/SP_150MB_ratio/1_SP_150MB_ratio_Splitted_Datasets/train_df.pkl'
    file_path_test = '../root forlder/datasets/SP_150MB_ratio/1_SP_150MB_ratio_Splitted_Datasets/test_df.pkl'
    file_path_val = '../root forlder/datasets/SP_150MB_ratio/1_SP_150MB_ratio_Splitted_Datasets/val_df.pkl'

 + generate embeddings for log templates
 $ python generate_embeddings.py <dataset> <strategy>
 where <dataset> is one of {HDFS, BGL, Thunderbird, or Spirit} and <strategy> is one of {average or tfidf}

We have three run for each method: 

- Cross domain (20% Normal data )
- Cross domain (20% Normal data + 20% Anomaly data )
- In domain. 
From yml file, we can decide source_datasets and Target dataset.

## Baseline Implementations

| Baseline | Implementation |
|---|---|
| DeepLog | [Exper_LOAGAD](https://github.com/nayefroqaya/Exper_LOAGAD/tree/main-before-2-months) |
| LogAnomaly | [Exper_LOAGAD](https://github.com/nayefroqaya/Exper_LOAGAD/tree/main-before-2-months) |
| LogRobust | [Exper_LOAGAD](https://github.com/nayefroqaya/Exper_LOAGAD/tree/main-before-2-months) |
| NeuralLog | [Exper_LOAGAD](https://github.com/nayefroqaya/Exper_LOAGAD/tree/main-before-2-months) |
| PLELog | [Exper_PLELOG](https://github.com/nayefroqaya/Exper_PLELOG) |
| LogFormer | [Exper_LogForm](https://github.com/nayefroqaya/Exper_LogForm) |


##### DeepLog : 
 - Cross domain (20% Normal data + 20% Anomaly data ) : 
python main_run.py --config_file config/deeplog_cross_20_normal_anomaly.yml
 - Cross domain (20% Normal data ):
python main_run.py --config_file config/deeplog_cross_20_normal_only.yml
 - In domain : 
python main_run.py --config_file config/deeplog.yml
=======

##### LogAnomaly : 
 - Cross domain (20% Normal data + 20% Anomaly data ) : 
python main_run.py --config_file config/loganomaly_cross_20_normal_anomaly.yml
 - Cross domain (20% Normal data ):
python main_run.py --config_file config/loganomaly_cross_20_normal_only.yml
 - In domain : 
python main_run.py --config_file config/loganomaly.yml

##### Logrobust : 
 - Cross domain (20% Normal data + 20% Anomaly data ) : 
python main_run.py --config_file config/logrobust_cross_20_normal_anomaly.yml
 - Cross domain (20% Normal data ):
python main_run.py --config_file config/logrobust_cross_20_normal_only.yml
 - In domain : 
python main_run.py --config_file config/Logrobust.yml


##### NeuralLog : 
 - Cross domain (20% Normal data + 20% Anomaly data ) : 
python main_run.py --config_file config/neurallog_cross_20_normal_anomaly.yml
 - Cross domain (20% Normal data ):
python main_run.py --config_file config/neurallog_cross_20_normal_only.yml
 - In domain : 
python main_run.py --config_file config/neurallog.yml


##======================================X
- PLELog: [Code](https://github.com/nayefroqaya/Exper_PLELOG)

  ../../root forlder/datasets is in the approaches/file.py

  ##### Cross domain (20% Normal only for training) - PLELog
- With assumption you run the log parser in proposed paper code, our PKL files are ready. 
- Set the path to data folder. 
- go to approaches folder.
- Open the file : PLELog_cross_20_only_normal.py
- Set the SOURCE_DATASETS : e.g - ["TH_1G"]
- Target data :argparser.add_argument('--dataset', default='BGL', type=str)  
- Set default to cross_dataset : argparser.add_argument('--case', default='cross_dataset', type=str, choices=['in_domain', 'cross_dataset'])

- run : ### python PLELog_cross_20_only_normal.py


##### Cross domain (20% Normal + 20% Anomaly  for training)- PLELog
- With assumption you run the log parser in proposed paper code, our PKL files are ready. 
- Set the path to data folder. default='../../root folder/datasets'
- go to approaches folder.
- Open the file : PLELog_cross_20_normal_anomaly.py
- Set the SOURCE_DATASETS : e.g - ["TH_1G"]
- Target data :argparser.add_argument('--dataset', default='BGL', type=str)  
- Set default to cross_dataset : argparser.add_argument('--case', default='cross_dataset', type=str, choices=['in_domain', 'cross_dataset'])
- run : python PLELog_cross_20_normal_anomaly.py


##### In domain :- PLELog
- With assumption you run the log parser in proposed paper code, our PKL files are ready. 
- Set the path to data folder. default='../../root folder/datasets'
- go to approaches folder.
- Open the file : PLELog_in_domain.py
- Target data :argparser.add_argument('--dataset', default='BGL', type=str)  # One one dataset in domain 
- Set default to in_domain : argparser.add_argument('--case', default='in_domain', type=str, choices=['in_domain', 'cross_dataset'])
- run : PLELog_in_domain.py

  

##======================================X
 
- LogFormer: [Code](https://github.com/nayefroqaya/Exper_LogForm)
In Exper_LogForm/preprocess/file.yml you can find all path fo all datasets. For example:

    - train_pkl: /storage/home/system name/root forlder/datasets/BGL/1_BGL_Splitted_Datasets/train_df.pkl
    - val_pkl: /storage/home/system name/root forlder/datasets/BGL/1_BGL_Splitted_Datasets/val_df.pkl
    - test_pkl: /storage/home/system name/root forlder/datasets/BGL/1_BGL_Splitted_Datasets/test_df.pkl
 
#####======================================X
##### Cross-dataset supervised LogFormer: i.e. BGL -> HDFS. LogFormer
#####======================================X

##### Normakl + anomaly in train - LogFormer

In yml file : setting: cross_dataset

python preprocess/preprocess_normalandanomaly_dataset_selector_from_config_no_overlap_last.py \
  --config preprocess/config_cross_dataset_last.yml


##### 2. Pre-train LogFormer on SOURCE BGL only - LogFormer
python Fraktion_normaandanomal_train_transformer_pkl_ready_last.py \
  --config preprocess/config_cross_dataset_last.yml


##### 3. Tune the pretrained BGL model on TARGET HDFS - LogFormer
#####    Uses the configured fraction of HDFS train:
#####    normal + anomaly
python Frackition_normalandanomal_tune_transformer_pkl_ready_last.py \
  --config preprocess/config_cross_dataset_last.yml

###------

##### Only Normal : - LogFormer
In yml file : setting: cross_dataset
python preprocess/preprocess_only_normal_dataset_selector_from_config_no_overlap_last.py \
  --config preprocess/config_only_normal_cross_dataset_last.yml


python Fraktion_only_normal_train_transformer_pkl_ready_last.py \
  --config preprocess/config_only_normal_cross_dataset_last.yml

python Frackition_only_normal_tune_transformer_pkl_ready_last.py \
  --config preprocess/config_only_normal_cross_dataset_last.yml


##### In domain  : LogFormer
- From YML file : setting =  in_domain
python preprocess/preprocess_only_normal_dataset_selector_from_config_no_overlap_last.py \
  --config preprocess/config_only_normal_cross_dataset_last.yml


python Fraktion_only_normal_train_transformer_pkl_ready_last.py \
  --config preprocess/config_only_normal_cross_dataset_last.yml

python Frackition_only_normal_tune_transformer_pkl_ready_last.py \
  --config preprocess/config_only_normal_cross_dataset_last.yml


### 📬 Contact
We are happy to answer your questions:   

| Name               | Email Address                             |
|--------------------|-------------------------------------------|
| Nayef Roqaya       | roqaya@staff.uni-marburg.de               |
| Thorsten Papenbrock| papenbrock@informatik.uni-marburg.de      |

