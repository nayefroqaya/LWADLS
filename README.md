# 1. (optional) MLM
python run_mlm.py --config configs/experiment.yaml

# 2. Teacher
python run_experiment.py --config configs/experiment.yaml --mode teacher

# 3. Student baseline
python run_experiment.py --config configs/experiment.yaml --mode student

# 4. Distilled student
python run_experiment.py --config configs/experiment.yaml --mode distill

## =====================


| Experiment    | What it proves         |
| ------------- | ---------------------- |
| In-domain     | model works            |
| Cross-dataset | generalization is hard |
| Multi-source  | diversity helps        |
| Leave-one-out | robustness             |

## Experiment 1 — In-domain
- HDFS → HDFS
- BGL → BGL
- Spirit → Spirit
- Tnunderbird → Tnunderbird
               
## Experiment 2 — Cross-dataset

- HDFS → BGL
- BGL → HDFS
- BGL   → Thunderbird 
- Thunderbird→ BGL 

## Experiment 3 — Multi-source (MAIN)

- Train: Thunderbird + BGL 
- Test: Spirit 
                 
- Train: Thunderbird + HDFS 
- Test: BGL

## Experiment 4 — Leave-one-out (BEST)

- Train: BGL + HDFS + Spirit + TH_a
- Test : TH_b
     
- Train: HDFS + Spirit + TH_a +  TH_b
- Test : BGL   


## Example : 

🔹 In our leave-one-out example

- train_datasets: ["HDFS", "BGL", "Spirit", "TH_a"]
- val_datasets:   ["HDFS", "BGL", "Spirit", "TH_a"]
- test_datasets:  ["TH_b"]
 ## --------
- Train: HDFS_train + BGL_train + Spirit_train + TH_a_train
- Val:   HDFS_val   + BGL_val   + Spirit_val   + TH_a_val
- Test:  TH_b_test# LWADLS

 ## ../outputs/
└── predictions/
    └── teacher__train-BGL__val-BGL__test-BGL__best_model__split-test__datasets-BGL/
        ├── evaluation_report.txt          ✅ (READ THIS)
        ├── metrics.csv
        ├── classification_report.csv
        ├── per_dataset_metrics.csv
        ├── predictions.csv
        └── per_dataset_classification_reports/
