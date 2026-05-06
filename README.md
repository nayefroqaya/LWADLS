# AdaLogSLM Updated Project

This project implements a unified semi-supervised SLM anomaly detection framework with two modes:

1. `in_domain`
2. `fewshot_target_adaptation`

It matches the YAML structure in `configs/adalogslm_unified_config.yml`.

## Supported input

The code loads `.pkl` split files such as:

```text
../datasets/BGL/1_BGL_Splitted_Datasets/train_df.pkl
../datasets/BGL/1_BGL_Splitted_Datasets/val_df.pkl
../datasets/BGL/1_BGL_Splitted_Datasets/test_df.pkl
```

The required columns are configured in YAML:

```yaml
columns:
  timestamp: "Timestamp"
  template: "processed_EventTemplate"
  block_id: "Node_block_id"
  label: "Label"
  dataset: "DatasetName"
```

The code constructs sequence-level samples by grouping events using `Node_block_id`.

## Run

```bash
pip install -r requirements.txt
python -m src.run --config configs/adalogslm_unified_config.yml
```

## Change mode

For in-domain:

```yaml
experiment:
  mode: "in_domain"
```

For few-shot target adaptation:

```yaml
experiment:
  mode: "fewshot_target_adaptation"
```

## Important design

For cross-dataset/few-shot mode, the code uses:

```text
source normal training: BGL + HDFS
target normal adaptation: TH_1G small normal subset
target normal center: TH_1G validation normal logs
target threshold: TH_1G validation normal logs
target test: TH_1G test logs
```

The model does not use source normal center for target detection.


## Run from PyCharm

Open the project folder in PyCharm:

```text
adalogslm_project_pycharm/
```

Then run:

```text
main.py
```

The default config is:

```text
configs/adalogslm_unified_config.yml
```

To change the running mode, edit this part in the YAML:

```yaml
experiment:
  mode: "fewshot_target_adaptation"
```

or:

```yaml
experiment:
  mode: "in_domain"
```

No terminal command is required if you run `main.py` directly from PyCharm.
