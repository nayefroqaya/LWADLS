import os
import warnings
from datetime import datetime

import colorama
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize colorama
colorama.init()
GREEN = colorama.Fore.GREEN
YELLOW = colorama.Fore.YELLOW
RESET = colorama.Fore.RESET
GRAY = colorama.Fore.LIGHTBLACK_EX


class Utilities:

    @staticmethod
    def insert_rows(df, row):
        """Insert a row at the end of the DataFrame."""
        insert_loc = df.index.max()
        if pd.isna(insert_loc):
            df.loc[0] = row
        else:
            df.loc[insert_loc + 1] = row

    @staticmethod
    def clean_up_df(df_features):
        """Remove rows with null content and display summary."""
        df_features = df_features[df_features['Content'].notnull()]
        print(GREEN + "[INFO] DataFrame cleaned. Null 'Content' rows removed." + RESET)
        df_features.info()
        return df_features

    @staticmethod
    def dataset_splitting(All_dataset_path_as_csv, dataset, round, Mix_or_stable):

        if Mix_or_stable == '0' and dataset == 'S_BGL':  # Stable
            print(GREEN + f"[INFO] Preparing dataset '{dataset}'..." + RESET)
            df_features = pd.read_csv('../datasets/S_BGL/stable_equal_subset.csv', escapechar='\\')
            df_features.info()

        elif Mix_or_stable == '1' and dataset == 'S_BGL':  # Mix
            print(GREEN + f"[INFO] Preparing dataset '{dataset}'..." + RESET)
            df_features = pd.read_csv('../datasets/S_BGL/50_50_mixed_subset.csv', escapechar='\\')
            df_features.info()

        else:
            """Load dataset CSV and split into train, validation, and test sets."""
            print(GREEN + f"[INFO] Preparing dataset '{dataset}'..." + RESET)
            df_features = pd.read_csv(All_dataset_path_as_csv, escapechar='\\')
            df_features.info()

            df1 = df_features.query("Label == 'Normal'").reset_index(drop=True)  # Normal logs
            df2 = df_features.query("Label != 'Normal'").reset_index(drop=True)  # Anomaly logs

            print(f"Normal logs: {len(df1):,}")  # TGH_1G :
            print(f"Anomaly logs: {len(df2):,}")  # TGH_1G :
            #exit()

        # Clean data
        df_features = Utilities.clean_up_df(df_features)

        # Standardize timestamps
        def update_timestamp(original_timestamp, desired_format='%Y-%m-%d %H:%M:%S.%f'):
            """Ensure timestamp matches the desired format."""
            try:
                datetime.strptime(str(original_timestamp), desired_format)
                return original_timestamp
            except ValueError:
                original_format = '%Y-%m-%d %H:%M:%S'
                parsed_timestamp = datetime.strptime(str(original_timestamp), original_format)
                if '%f' in desired_format and parsed_timestamp.microsecond == 0:
                    parsed_timestamp = parsed_timestamp.replace(microsecond=0)
                return parsed_timestamp.strftime(desired_format)

        df_features['Timestamp'] = df_features['Timestamp'].apply(update_timestamp)
        df_features.sort_values(by=['Node_block_id', 'Timestamp'], inplace=True)
        df_features.reset_index(drop=True, inplace=True)
        df_features = df_features[['Timestamp', 'Date', 'Time', 'Content', 'Original_Label', 'EventId', 'EventTemplate',
                                   'processed_EventTemplate', 'Node_block_id', 'Label']]
        df_features.info()
        print(GREEN + "[INFO] Dataset timestamps standardized and sorted." + RESET)
        #        exit()

        # Split dataset based on dataset type
        unique_ids = df_features['Node_block_id'].unique()
        total_ids = len(unique_ids)

        if dataset in ['HDFS', 'BGL',  'SP_150MB_ratio', 'TH_1G','S_BGL']:

            # Shuffle and split
            shuffled_ids = np.random.permutation(unique_ids)
            train_size, val_size = int(0.6 * total_ids), int(0.1 * total_ids)
            train_ids, val_ids, test_ids = shuffled_ids[:train_size], shuffled_ids[
                train_size:train_size + val_size], shuffled_ids[train_size + val_size:]
        else:
            raise ValueError(f"[ERROR] Unsupported dataset type: {dataset}")

        # Check for overlaps between splits
        set_train, set_val, set_test = set(train_ids), set(val_ids), set(test_ids)
        intersections = {"train_val": set_train.intersection(set_val), "train_test": set_train.intersection(set_test),
            "val_test": set_val.intersection(set_test)}
        for k, v in intersections.items():
            print(YELLOW + f"[CHECK] Intersection {k}: {v}" + RESET)

        if all(len(v) == 0 for v in intersections.values()):
            print(GREEN + "[INFO] No overlaps found between train, validation, and test sets." + RESET)
        else:
            raise ValueError("[ERROR] Overlaps detected between dataset splits!")

        # Create DataFrames for splits
        train_df = df_features[df_features['Node_block_id'].isin(train_ids)].copy()
        train_df['Type_ds'] = 'Train'
        val_df = df_features[df_features['Node_block_id'].isin(val_ids)].copy()
        val_df['Type_ds'] = 'Validation'
        test_df = df_features[df_features['Node_block_id'].isin(test_ids)].copy()
        test_df['Type_ds'] = 'Test'

        df_features.info()
        # =======
        if Mix_or_stable == '0' and dataset == 'S_BGL':  # Stable
            # Create folder to save splits
            save_path = os.path.join(f"../datasets/{dataset}", f"{round}_{dataset}_'Stable'_Splitted_Datasets")
            os.makedirs(save_path, exist_ok=True)
            # Save each dataframe as PKL
            train_df.to_pickle(os.path.join(save_path, "train_df.pkl"))
            val_df.to_pickle(os.path.join(save_path, "val_df.pkl"))
            test_df.to_pickle(os.path.join(save_path, "test_df.pkl"))

        elif Mix_or_stable == '1' and dataset == 'S_BGL':  # Mix
            # Create folder to save splits
            save_path = os.path.join(f"../datasets/{dataset}", f"{round}_{dataset}_'Mix'_Splitted_Datasets")
            os.makedirs(save_path, exist_ok=True)
            # Save each dataframe as PKL
            train_df.to_pickle(os.path.join(save_path, "train_df.pkl"))
            val_df.to_pickle(os.path.join(save_path, "val_df.pkl"))
            test_df.to_pickle(os.path.join(save_path, "test_df.pkl"))



        else:
            # Create folder to save splits
            save_path = os.path.join(f"../datasets/{dataset}", f"{round}_{dataset}_Splitted_Datasets")
            os.makedirs(save_path, exist_ok=True)
            # Save each dataframe as PKL
            train_df.to_pickle(os.path.join(save_path, "train_df.pkl"))
            val_df.to_pickle(os.path.join(save_path, "val_df.pkl"))
            test_df.to_pickle(os.path.join(save_path, "test_df.pkl"))

        # Display split info
        print(
            GREEN + f"[INFO] Dataset split complete. Sizes -> Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}" + RESET)
        df_block_train = train_df.drop_duplicates(subset=['Node_block_id']).reset_index(
            drop=True)  # Unique Normal Blocks
        df3 = df_block_train.query("Label == 'Normal'").reset_index(drop=True)
        df4 = df_block_train.query("Label == 'Anomaly'").reset_index(drop=True)
        print(' Normal seq Train : ' + str(len(df3)))
        print(' Anomaly seq Train : ' + str(len(df4)))
        df_block_test = test_df.drop_duplicates(subset=['Node_block_id']).reset_index(drop=True)  # Unique Normal Blocks
        df3 = df_block_test.query("Label == 'Normal'").reset_index(drop=True)
        df4 = df_block_test.query("Label == 'Anomaly'").reset_index(drop=True)
        print(' Normal seq Test : ' + str(len(df3)))
        print(' Anomaly seq Test : ' + str(len(df4)))

    @staticmethod
    def processing_data_portion(train_df, validate_df, test_df):
        """Create labeled and unlabeled portions for training and mark test set."""

        # Seq count  Train
        unique_normal_train = train_df[train_df['Label'] == 'Normal']['Node_block_id'].unique()
        unique_anomaly_train = train_df[train_df['Label'] == 'Anomaly']['Node_block_id'].unique()
        #print(train_df['Label'].unique())
        #exit()
        # Logs count Train 
        train_normal_logs = (train_df['Label'] == 'Normal').sum()
        train_anomaly_logs = (train_df['Label'] == 'Anomaly').sum()

        # Seq count  Test
        unique_normal_test = test_df[test_df['Label'] == 'Normal']['Node_block_id'].unique()
        unique_anomaly_test = test_df[test_df['Label'] == 'Anomaly']['Node_block_id'].unique()
        # Logs count Test 
        test_normal_logs = (test_df['Label'] == 'Normal').sum()
        test_anomaly_logs = (test_df['Label'] == 'Anomaly').sum()

        print(
            GREEN + f"[INFO] Training data : Total Seq  unique normal Node_block_ids: {len(unique_normal_train)}" + RESET)
        print(
            GREEN + f"[INFO] Training data : Total Seq  unique anomaly Node_block_ids: {len(unique_anomaly_train)}" + RESET)
        print(GREEN + f"[INFO] Training data : Total  normal logs: {train_normal_logs}" + RESET)
        print(GREEN + f"[INFO] Training data :Total  anomaly  logs: {train_anomaly_logs}" + RESET)

        print(
            GREEN + f"[INFO] Testing data : Total Seq  unique normal Node_block_ids: {len(unique_normal_test)}" + RESET)
        print(
            GREEN + f"[INFO] Testing data : Total Seq  unique anomaly Node_block_ids: {len(unique_anomaly_test)}" + RESET)
        print(GREEN + f"[INFO] Testing data : Total  normal logs: {test_normal_logs}" + RESET)
        print(GREEN + f"[INFO] Testing data : Total  anomaly  logs: {test_anomaly_logs}" + RESET)

        # Select 50% of normal blocks for labeled training
        selected_normal_50 = np.random.choice(unique_normal_train, size=len(unique_normal_train) // 2, replace=False)
        df_train_normal_50 = train_df[train_df['Node_block_id'].isin(selected_normal_50)].copy()
        df_train_normal_50['Temp_label'] = 0

        # Remaining normal + all anomaly blocks are unlabeled
        remaining_normal = set(unique_normal_train) - set(selected_normal_50)
        df_train_unlabeled = train_df[train_df['Node_block_id'].isin(remaining_normal) | train_df['Node_block_id'].isin(
            unique_anomaly_train)].copy()
        df_train_unlabeled['Temp_label'] = 999

        # Mark test set with Temp_label = 888
        test_df['Temp_label'] = 888
        validate_df['Temp_label'] = 777

        # Combine all datasets
        final_dataset = pd.concat([df_train_normal_50, df_train_unlabeled, validate_df, test_df], ignore_index=True)

        print(
            GREEN + f"[INFO] Labeled normal blocks: {df_train_normal_50['Node_block_id'].nunique()} (Temp_label=0)" + RESET)
        print(
            GREEN + f"[INFO] Unlabeled blocks (remaining normal + anomaly): {df_train_unlabeled['Node_block_id'].nunique()} (Temp_label=999)" + RESET)
        print(GREEN + f"[INFO] Test blocks: {test_df['Node_block_id'].nunique()} (Temp_label=888)" + RESET)
        print(GREEN + f"[INFO] Test blocks: {validate_df['Node_block_id'].nunique()} (Temp_label=777)" + RESET)

        print(GREEN + f"[INFO] Final combined dataset size: {len(final_dataset)} rows" + RESET)

        final_dataset.info()
        return final_dataset
