import re
import warnings

import colorama
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

nltk.download('punkt_tab')
nltk.download('stopwords')

warnings.filterwarnings('ignore')
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW

# initialize the set of links (unique links)
internal_urls = set()
external_urls = set()


class LogdataRead:

    @staticmethod
    def get_dataset_text_content_message(log_message):
        # Check if the log_message is empty from the beginning
        if not log_message or str(log_message).strip() == '':
            print("Log message is empty. Returning default value.")
            exit()
            return ""  # Return a default value or placeholder for empty input

        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()

        # 1. Remove non-characters (digits and punctuation)
        def remove_non_characters(text):
            # Ensure that text is a string
            if not isinstance(text, str):
                print(f"Warning: Expected string, but received {type(text)}. Returning empty string.")
                print('It is not ok  with the text ------' + str(text))
                exit()
                return ""  # Return an empty string if input is not a valid string
            else:
                print('--ok text ------' + str(text))
                # Apply regular expression to remove non-alphabetical characters
                cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
                return cleaned_text

        def camel_case_split(text):
            if not text:
                return []

            words = [[text[0]]]
            for c in text[1:]:
                if words[-1][-1].islower() and c.isupper():
                    words.append([c])
                else:
                    words[-1].append(c)
            return [''.join(word) for word in words]

        # Step 1: Clean the input text by removing non-alphabetical characters
        cleaned_text = remove_non_characters(log_message)
        tokens_word = word_tokenize(cleaned_text)

        # Step 2: Remove common English stop words
        print('Removing common English stop words...')
        stop_words = set(stopwords.words('english'))
        tokens_stopword = [token for token in tokens_word if token.lower() not in stop_words]

        # If no stop words were removed, retain the original tokens
        if not tokens_stopword:
            tokens_stopword = tokens_word

        # Step 3: Lemmatize tokens
        tokens_lemma = [lemmatizer.lemmatize(token) for token in tokens_stopword]

        # Step 4: Handle empty tokens after processing
        if not tokens_lemma:
            print('Tokenization issues, returning default value for empty tokens...')
            print(log_message)
            lst_words_ready = camel_case_split(' '.join(tokens_stopword).strip())
            final_sentence = ' '.join(lst_words_ready)
            return final_sentence.strip()  # Return default value if no valid tokens
        else:
            lst_words_ready = camel_case_split(' '.join(tokens_lemma).strip())
            final_sentence = ' '.join(lst_words_ready)

            print('Processed message:')
            print(final_sentence)
            return final_sentence.strip()

    @staticmethod
    def get_block_id_hdfs(line_content):
        # Define regex pattern to match "blk_" IDs
        pattern = r"blk_-?[0-9]+"

        # Search for block ID in the log line
        match = re.search(pattern, line_content)

        if match:
            return match.group(0)  # Return the first matched block ID
        else:
            # Log the error and return None instead of exiting
            import logging
            logging.error(f'No block ID found in the sentence: {line_content}')

            return None  # Returning None allows the calling code to handle the error

    def read_original_data_log_from_log_to_csv(self, dataset, All_dataset_path_as_csv):
        '''
        def fill_unknown_node_block_id(df):
            prev_valid_id = None  # Store the last valid Node_block_id
            prev_label = None  # Store the last Label

            for i in tqdm(range(len(df)), desc="Processing Rows", unit="row"):

                if df.loc[i, 'Node_block_id'] == 'UNKNOWN':
                    if prev_valid_id is not None and df.loc[i, 'Label'] == prev_label:
                        df.loc[i, 'Node_block_id'] = prev_valid_id  # Assign previous Node_block_id
                else:
                    prev_valid_id = df.loc[i, 'Node_block_id']  # Update previous valid ID
                    prev_label = df.loc[i, 'Label']  # Update previous Label

            return df
        '''

        def fill_unknown_node_block_id(df):

            prev_valid_id = None

            for i in tqdm(range(len(df)), desc="Processing Rows", unit="row"):
                current_id = df.loc[i, 'Node_block_id']

                if current_id != 'UNKNOWN':
                    prev_valid_id = current_id
                else:
                    if prev_valid_id is not None:
                        df.loc[i, 'Node_block_id'] = prev_valid_id

            return df

        if dataset == 'BGL':

            #  Define dtype mapping for efficient memory usage
            dtype_mapping = {"Node": "str", "NodeRepeat": "str", "EventTemplate": "category", "Content": "str",
                             "Date": "str", "Time": "str", "Level": "category", "Component": "category",
                             "EventId": "str", "Label": "category"}

            df = pd.read_csv(f'../datasets/{dataset}/{dataset}.log_structured.csv', dtype=dtype_mapping)
            df.info()
            x = len(df)
            #  Rename columns for consistency
            df = df.rename(columns={"Node": "Node_block_id"})

            # Step 1: Replace NaN with "UNKNOWN"
            df['Node_block_id'] = df['Node_block_id'].fillna('UNKNOWN')
            df = fill_unknown_node_block_id(df)

            df = df[~df['Node_block_id'].astype(str).str.contains("UNKNOWN", case=False, na=False)].reset_index(
                drop=True)
            total_rows = len(df)
            print("Total rows in df :", total_rows)

            counts_per_block = df.groupby('Node_block_id').size()
            empty_blocks = counts_per_block[counts_per_block == 0].index.tolist()
            print("Node_block_id with 0 rows:", empty_blocks)
            print("Number of empty Node_block_id groups:", len(empty_blocks))
            # exit()

            #  Parse Timestamp Correctly (Format: YYYY-MM-DD-HH.MM.SS.ffffff)
            df['Timestamp'] = pd.to_datetime(df['Time'], format="%Y-%m-%d-%H.%M.%S.%f", errors='coerce')

            nan_count = df['Content'].isna().sum()
            print(f"Number of NaN values in 'log_message': {nan_count}")

            #  Process EventTemplate
            df["processed_EventTemplate"] = df["EventTemplate"].apply(self.get_dataset_text_content_message)
            df['processed_EventTemplate'].fillna(df['Content'], inplace=True)
            nan_count = df['processed_EventTemplate'].isna().sum()
            print(f"Number of NaN values in 'processed_EventTemplate': {nan_count}")

            df['processed_EventTemplate'] = df['processed_EventTemplate'].astype(str)
            # Select only required columns (memory efficiency)
            df = df[['Timestamp', 'Date', 'Time', 'Content', 'EventId', 'EventTemplate', 'processed_EventTemplate',
                     'Node_block_id', 'Label']]

            print(' length df before windows ' + str(len(df)))

            def process_logs(df, window_size=120):
                df = df.copy()
                df.sort_values(by=['Node_block_id', 'Timestamp'], inplace=True)  # Ensure order

                blocks = []

                for node_id, group in df.groupby('Node_block_id'):
                    entries = group.to_dict('records')  # Convert to list of dicts
                    num_entries = len(entries)
                    block_count = (num_entries + window_size - 1) // window_size  # Number of blocks

                    for i in range(block_count):
                        start_idx = i * window_size
                        end_idx = min(start_idx + window_size, num_entries)
                        block_entries = entries[start_idx:end_idx]

                        # Define block name
                        block_name = f"{node_id}_block_{i}"
                        block_label = "Anomaly" if any(entry['Label'] != '-' for entry in block_entries) else "Normal"

                        for entry in block_entries:
                            entry['Block'] = block_name
                            entry['Updated_Label'] = block_label
                            blocks.append(entry)

                return pd.DataFrame(blocks)

            # Example usage
            df = process_logs(df, window_size=120)  # Block , Updated_Label
            # print(df_processed.head())
            print(' length df after windows ' + str(len(df)))

            unknown_blocks = df[df['Node_block_id'].astype(str).str.contains("UNKNOWN", case=False, na=False)]

            print("Total rows where Node_block_id contains UNKNOWN:", len(unknown_blocks))
            #            exit()

            print('check....')
            # Separate Normal & Anomaly Logs
            df1 = df.query("Label == '-'").reset_index(drop=True)  # Normal logs
            df2 = df.query("Label != '-'").reset_index(drop=True)  # Anomaly logs
            df_block = df.drop_duplicates(subset=['Block']).reset_index(drop=True)  # Unique Normal Blocks
            df3 = df_block.query("Updated_Label == 'Normal'").reset_index(drop=True)
            df4 = df_block.query("Updated_Label == 'Anomaly'").reset_index(drop=True)

            # Print Dataset Statistics
            print(f" logs Messages : {x:,}")
            print(f"Normal logs: {len(df1):,}")  # 4,365,033
            print(f"Anomaly logs: {len(df2):,}")  # 348,460
            print(f"Unique normal blocks: {len(df3):,}")  # 49,247
            print(f"Unique anomaly blocks: {len(df4):,}")  # 36,251
            print(f"All unique blocks: {len(df_block):,}")
            df = df.drop(columns=['Node_block_id'])
            df = df.rename(columns={'Label': 'Original_Label'})
            # -----------------------------------------------
            # df = df.drop(columns=['Node_block_id', 'Label'])
            # df = df.rename(columns={'Block': 'Node_block_id', 'Updated_Label': 'Label'})
            # -----------------------------------------------
            df = df.rename(columns={'Block': 'Node_block_id', 'Updated_Label': 'Label'})
            df.info()
            print(' save as csv file ....')
            # Save Processed Dataset Efficiently
            df.to_csv(All_dataset_path_as_csv, escapechar='\\', index=False)

        elif dataset == 'S_BGL':
            path = f'../datasets/BGL/BGL.csv'
            df = pd.read_csv(path, escapechar='\\')
            df.info()
            num_templates = df['EventTemplate'].nunique()
            num_logs = len(df)
            stability_ratio = num_logs / num_templates
            rare_ratio = (df['EventTemplate'].value_counts() < 10).mean()
            print(f"Templates with <10 occurrences: {rare_ratio:.2%}")
            print(f"Each template appears on average {stability_ratio:.2f} times")
            vc = df['EventTemplate'].value_counts()
            coverage = vc[vc >= 10].sum() / vc.sum()
            print(f"Log coverage by templates with ≥10 occurrences: {coverage:.2%}")

            df['LogType'] = df['Original_Label'].apply(lambda x: 'Normal' if x == '-' else 'Anomaly')
            df['LogType'].value_counts(normalize=True) * 100
            print(df['LogType'].value_counts())
            # Percentages
            percentages = df['LogType'].value_counts(normalize=True) * 100
            counts = df['LogType'].value_counts()

            print("Counts:")
            print(counts)

            print("\nPercentages:")
            print(percentages.round(2))
            subset_frac = 0.5  # 10% subset
            subset_df = (
                df.groupby('LogType', group_keys=False).apply(lambda x: x.sample(frac=subset_frac, random_state=42)))
            # Verify
            subset_df['LogType'].value_counts(normalize=True) * 100
            print(subset_df['LogType'].value_counts())

            # Stable templates: ≥10 occurrences in subset_df
            stable_templates = subset_df['EventTemplate'].value_counts()[lambda x: x >= 10].index

            # Unstable templates: <10 occurrences in subset_df
            unstable_templates = subset_df['EventTemplate'].value_counts()[lambda x: x < 10].index

            stable_part = subset_df[subset_df['EventTemplate'].isin(stable_templates)]
            unstable_part = subset_df[subset_df['EventTemplate'].isin(unstable_templates)]

            # total_size = len(subset_df)
            # stable_size = total_size // 2
            # unstable_size = total_size - stable_size

            total_size = len(subset_df)
            stable_size = int(total_size * 0.25)
            unstable_size = total_size - stable_size

            # Ensure we never sample more than available rows
            stable_sample_size = min(stable_size, len(stable_part))
            unstable_sample_size = min(unstable_size, len(unstable_part))

            sampled_stable = stable_part.sample(n=stable_sample_size, random_state=42)
            sampled_unstable = unstable_part.sample(n=unstable_sample_size, random_state=42)

            mixed_subset = pd.concat([sampled_stable, sampled_unstable]).sample(frac=1, random_state=42)

            # Calculate stable/unstable percentages
            stable_percentage = mixed_subset['EventTemplate'].value_counts()[lambda x: x >= 10].sum() / len(
                mixed_subset) * 100
            unstable_percentage = mixed_subset['EventTemplate'].value_counts()[lambda x: x < 10].sum() / len(
                mixed_subset) * 100

            print(f"Stable templates: {stable_percentage:.2f}%")
            print(f"Unstable templates: {unstable_percentage:.2f}%")
            print(mixed_subset['LogType'].value_counts(normalize=True).mul(100).round(2))

            # Count of Normal / Anomaly in mixed_subset
            counts = mixed_subset['LogType'].value_counts()
            normal_ratio = counts.get('Normal', 0) / len(mixed_subset)
            anomaly_ratio = counts.get('Anomaly', 0) / len(mixed_subset)

            # Sample stable-only rows from subset_df to match count & ratio
            stable_only_df = subset_df[subset_df['EventTemplate'].isin(stable_templates)]

            stable_normal_count = int(normal_ratio * len(mixed_subset))
            stable_anomaly_count = int(anomaly_ratio * len(mixed_subset))

            # Ensure we don't sample more than available
            stable_normal_count = min(stable_normal_count, len(stable_only_df[stable_only_df['LogType'] == 'Normal']))
            stable_anomaly_count = min(stable_anomaly_count,
                                       len(stable_only_df[stable_only_df['LogType'] == 'Anomaly']))

            stable_normal_sample = stable_only_df[stable_only_df['LogType'] == 'Normal'].sample(n=stable_normal_count,
                                                                                                random_state=42)
            stable_anomaly_sample = stable_only_df[stable_only_df['LogType'] == 'Anomaly'].sample(
                n=stable_anomaly_count, random_state=42)

            stable_equal_subset = pd.concat([stable_normal_sample, stable_anomaly_sample]).sample(frac=1,
                                                                                                  random_state=42)

            # ----------- Debug prints
            print(f"Rows in stable_equal_subset: {len(stable_equal_subset)}")
            print(f"Rows in mixed_subset: {len(mixed_subset)}")

            print('---------------------------------')
            print("stable_equal_subset Normal/Anomaly:")
            print(stable_equal_subset['LogType'].value_counts(normalize=True).mul(100).round(2))

            print("\nmixed_subset Normal/Anomaly:")
            print(mixed_subset['LogType'].value_counts(normalize=True).mul(100).round(2))
            # Drop 'LogType' column before saving
            mixed_subset_to_save = mixed_subset.drop(columns=['LogType'])
            stable_equal_subset_to_save = stable_equal_subset.drop(columns=['LogType'])

            # Save to CSV
            mixed_subset_to_save.to_csv(f'../datasets/S_BGL/50_50_mixed_subset.csv', index=False)
            stable_equal_subset_to_save.to_csv('../datasets/S_BGL/stable_equal_subset.csv', index=False)
            print("CSV files saved: 'mixed_subset.csv' and 'stable_equal_subset.csv' (without LogType column).")

        elif dataset == 'HDFS':  # 11.170.033
            # new ---------
            # File paths
            dataset_path = f'../datasets/{dataset}/{dataset}.log_structured.csv'
            hdfs_label_file_path = '../datasets/HDFS/anomaly_label.csv'

            # Optimized Data Loading (Using dtypes for Memory Efficiency)
            dtype_mapping = {'Node_block_id': 'str', 'Content': 'str', 'Date': 'str', 'Time': 'str',
                             'Level': 'category', 'Component': 'category', 'EventId': 'category',
                             'EventTemplate': 'str', 'ParameterList': 'str'}
            df = pd.read_csv(dataset_path, dtype=dtype_mapping)
            df.info()
            x = len(df)
            # Extract Block ID
            df["Node_block_id"] = df["Content"].apply(self.get_block_id_hdfs)
            df_missing = df[df['Node_block_id'].isna()]
            print(len(df_missing))

            # Read Labels (Optimized merge with dtype)
            df_labels = pd.read_csv(hdfs_label_file_path, dtype={'BlockId': 'str', 'Label': 'category'})

            count_block_original = df_labels.drop_duplicates(subset=['BlockId'])  # Unique Normal Blocks
            count_block_ds = df.drop_duplicates(subset=['Node_block_id'])  # Unique Normal Blocks
            print(len(count_block_original))
            print(len(count_block_ds))

            # Merge Logs with Anomaly Labels
            # Debugging log count before merging
            print(f"Total logs before merging: {len(df)}")
            df = pd.merge(df, df_labels, how='left', left_on='Node_block_id', right_on='BlockId')
            print(f"Total logs after merging: {len(df)}")

            # Remove Rows with Missing BlockId (Ensures valid merges)
            # Check logs with missing labels
            missing_labels = df['Node_block_id'].isna().sum()
            print(f"Logs with missing labels (NaN Node_block_id): {missing_labels}")

            # Print example missing Node_block_id values
            if missing_labels > 0:
                print("Example of missing Node_block_id values:")
                print(df[df['BlockId'].isna()].head(10))
                exit()
            # Debugging before dropna
            print(f"Total logs before dropna: {len(df)}")
            df['BlockId'] = df['BlockId'].replace(to_replace='None', value=np.nan)
            #            df = df.dropna(subset=['BlockId'])
            print(f"Total logs after dropna: {len(df)}")

            # Ensure 'Date' has leading zeros (Padding)
            print(' Zfill ..........')
            df['Date'] = df['Date'].astype(str).str.zfill(6)

            # Safe Parsing of Date & Time
            def safe_parse(date, time):
                try:
                    return pd.to_datetime(str(date) + str(time), format='%y%m%d%H%M%S')
                except ValueError:
                    return pd.NaT  # Avoid exiting program, mark invalid timestamps

            print('preparing Timestamp.......')
            df['Timestamp'] = df.apply(lambda row: safe_parse(row['Date'], row['Time']), axis=1)

            # Process EventTemplate
            df["processed_EventTemplate"] = df["EventTemplate"].apply(self.get_dataset_text_content_message)
            df['processed_EventTemplate'].fillna(df['Content'], inplace=True)
            nan_count = df['processed_EventTemplate'].isna().sum()
            if nan_count > 0:
                print(f"1_Number of NaN values in 'processed_EventTemplate': {nan_count}")
                exit()
            # Select Relevant Columns Only (Reduce Memory Usage)
            df = df[['Timestamp', 'Date', 'Time', 'Content', 'EventId', 'EventTemplate', 'processed_EventTemplate',
                     'Node_block_id', 'Label']]

            df = df.rename(columns={'Label': 'Original_Label'})
            # Create a new column 'updated_Label' with the same values as 'Original_Label'
            df['Label'] = df['Original_Label']

            nan_count = df['processed_EventTemplate'].isna().sum()

            if nan_count > 0:
                print(f"2_Number of NaN values in 'processed_EventTemplate': {nan_count}")
                exit()

            print(len(df))
            # Reset Index (Avoids Gaps)
            df.reset_index(drop=True, inplace=True)
            print(len(df))
            #            exit()

            #  Separate Normal & Anomaly Logs
            df1 = df[df['Label'] == 'Normal']  # Normal logs
            df2 = df[df['Label'] == 'Anomaly']  # Anomalous logs

            #  Extract Unique Node Block IDs
            df3 = df1.drop_duplicates(subset=['Node_block_id'])  # Unique Normal Blocks
            df4 = df2.drop_duplicates(subset=['Node_block_id'])  # Unique Anomalous Blocks

            #  Reset Index to Optimize Performance
            for d in [df1, df2, df3, df4]:
                d.reset_index(drop=True, inplace=True)

            #  Print Dataset Statistics
            print(f"All logs: {x:,}")
            print(f"Normal logs: {len(df1):,}")  # 10,887,379
            print(f"Anomaly logs: {len(df2):,}")  # 288,250
            print(f"Unique normal blocks: {len(df3):,}")  # 558,223
            print(f"Unique anomaly blocks: {len(df4):,}")  # 16,838
            n_total = len(df)
            n_anomaly = (df['Label'] != 'Normal').sum()
            n_normal = (df['Label'] == 'Normal').sum()

            print(f"Total: {n_total}")
            print(f"Normal: {n_normal} ({n_normal / n_total:.2%})")
            print(f"Anomaly: {n_anomaly} ({n_anomaly / n_total:.2%})")
            print(' ----- Completed ------')

            #  Check Memory Usage
            df.info()
            #  Save Processed Dataset Efficiently
            df.to_csv(All_dataset_path_as_csv, index=False)

        elif dataset == 'TH_1G' or dataset == 'TH_2G' or dataset == 'TH_5G' or dataset == 'TH_10G':
            #  Define dtype mapping for efficient memory usage
            dtype_mapping = {"User": "str", "EventTemplate": "category", "Content": "str", "Date": "str", "Time": "str",
                             "Component": "category", "EventId": "str", "Label": "category"}

            df = pd.read_csv(f'../datasets/{dataset}/{dataset}.log_structured.csv', dtype=dtype_mapping)
            df.info()
            x = len(df)

            #  Rename columns for consistency
            df = df.rename(columns={"User": "Node_block_id"})

            # Step 1: Replace NaN with "UNKNOWN"
            df['Node_block_id'] = df['Node_block_id'].fillna('UNKNOWN')
            df = fill_unknown_node_block_id(df)

            #  Parse Timestamp Correctly (Format: YYYY-MM-DD-HH.MM.SS.ffffff)
            df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M:%S")

            nan_count = df['Content'].isna().sum()
            print(f"Number of NaN values in 'log_message': {nan_count}")

            #  Process EventTemplate
            df["processed_EventTemplate"] = df["EventTemplate"].apply(self.get_dataset_text_content_message)
            df['processed_EventTemplate'].fillna(df['Content'], inplace=True)
            nan_count = df['processed_EventTemplate'].isna().sum()
            print(f"Number of NaN values in 'processed_EventTemplate': {nan_count}")

            df['processed_EventTemplate'] = df['processed_EventTemplate'].astype(str)
            # Select only required columns (memory efficiency)
            df = df[['Timestamp', 'Date', 'Time', 'Content', 'EventId', 'EventTemplate', 'processed_EventTemplate',
                     'Node_block_id', 'Label']]

            print(' length df before windows ' + str(len(df)))

            def process_logs(df, window_size=120):

                df = df.copy()
                df.sort_values(by=['Node_block_id', 'Timestamp'], inplace=True)  # Ensure order

                blocks = []

                for node_id, group in df.groupby('Node_block_id'):
                    entries = group.to_dict('records')  # Convert to list of dicts
                    num_entries = len(entries)
                    block_count = (num_entries + window_size - 1) // window_size  # Number of blocks

                    for i in range(block_count):
                        start_idx = i * window_size
                        end_idx = min(start_idx + window_size, num_entries)
                        block_entries = entries[start_idx:end_idx]

                        # Define block name
                        block_name = f"{node_id}_block_{i}"
                        block_label = "Anomaly" if any(entry['Label'] != '-' for entry in block_entries) else "Normal"

                        for entry in block_entries:
                            entry['Block'] = block_name
                            entry['Updated_Label'] = block_label
                            blocks.append(entry)

                return pd.DataFrame(blocks)

            # Example usage
            df = process_logs(df, window_size=120)  # Block , Updated_Label
            # print(df_processed.head())
            print(' length df after windows ' + str(len(df)))

            print('check....')
            # Separate Normal & Anomaly Logs
            df1 = df.query("Label == '-'").reset_index(drop=True)  # Normal logs
            df2 = df.query("Label != '-'").reset_index(drop=True)  # Anomaly logs
            df_block = df.drop_duplicates(subset=['Block']).reset_index(drop=True)  # Unique Normal Blocks
            df3 = df_block.query("Updated_Label == 'Normal'").reset_index(drop=True)
            df4 = df_block.query("Updated_Label == 'Anomaly'").reset_index(drop=True)

            # Print Dataset Statistics
            print(f"All logs: {x:,}")
            print(f"Normal logs: {len(df1):,}")  # TH_1G : 4,365,033
            print(f"Anomaly logs: {len(df2):,}")  # TH_1G :348,460
            print(f"Unique normal blocks: {len(df3):,}")  # 49,247
            print(f"Unique anomaly blocks: {len(df4):,}")  # 36,251
            print(f"All unique blocks: {len(df_block):,}")
            df = df.drop(columns=['Node_block_id'])
            df = df.rename(columns={'Label': 'Original_Label'})
            # -----------------------------------------------
            # df = df.drop(columns=['Node_block_id', 'Label'])
            # df = df.rename(columns={'Block': 'Node_block_id', 'Updated_Label': 'Label'})
            # -----------------------------------------------
            df = df.rename(columns={'Block': 'Node_block_id', 'Updated_Label': 'Label'})
            df.info()
            print(' save as csv file ....')
            # Save Processed Dataset Efficiently
            df.to_csv(All_dataset_path_as_csv, escapechar='\\', index=False)


        elif dataset == 'TH_1G_ratio' or dataset == 'TH_2G_ratio' or dataset == 'TH_3G_ratio' or dataset == 'TH_5G_ratio' or dataset == 'TH_Full' or dataset == 'TH_6G_ratio' or dataset == 'TH_9G_ratio' or dataset == 'TH_16G_ratio' or dataset == 'TH_12G_ratio' or dataset == 'TH_20G_ratio':
            #  Define dtype mapping for efficient memory usage
            dtype_mapping = {"User": "str", "EventTemplate": "category", "Content": "str", "Date": "str", "Time": "str",
                             "Component": "category", "EventId": "str", "Label": "category"}

            df = pd.read_csv(f'../datasets/{dataset}/{dataset}.log_structured.csv', dtype=dtype_mapping)
            df.info()
            x = len(df)

            #  Rename columns for consistency
            df = df.rename(columns={"User": "Node_block_id"})

            # Step 1: Replace NaN with "UNKNOWN"
            df['Node_block_id'] = df['Node_block_id'].fillna('UNKNOWN')
            df = fill_unknown_node_block_id(df)

            #  Parse Timestamp Correctly (Format: YYYY-MM-DD-HH.MM.SS.ffffff)
            df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M:%S")

            nan_count = df['Content'].isna().sum()
            print(f"Number of NaN values in 'log_message': {nan_count}")

            #  Process EventTemplate
            df["processed_EventTemplate"] = df["EventTemplate"].apply(self.get_dataset_text_content_message)
            df['processed_EventTemplate'].fillna(df['Content'], inplace=True)
            nan_count = df['processed_EventTemplate'].isna().sum()
            print(f"Number of NaN values in 'processed_EventTemplate': {nan_count}")

            df['processed_EventTemplate'] = df['processed_EventTemplate'].astype(str)
            # Select only required columns (memory efficiency)
            df = df[['Timestamp', 'Date', 'Time', 'Content', 'EventId', 'EventTemplate', 'processed_EventTemplate',
                     'Node_block_id', 'Label']]

            # -------take anomaly ration and normal ration based on original dataset. In Thunderbird 4.3% Anomaly
            '''
            r = 0.043
            a = df[df['Label'] != '-']  # anomaly
            n = df[df['Label'] == '-']  # normal
            # take all available normal rows
            n_n = len(n)

            # compute how many anomalies are needed so anomaly ratio becomes 4.3%
            n_a = int(n_n * r / (1 - r))

            # sample anomalies
            df_final = pd.concat([a.sample(n=n_a, random_state=42), n]).sample(frac=1, random_state=42).reset_index(
                drop=True)

            # check
            n_total = len(df_final)
            n_anomaly = (df_final['Label'] != '-').sum()
            n_normal = (df_final['Label'] == '-').sum()

            print(f"Total: {n_total}")
            print(f"Normal: {n_normal} ({n_normal / n_total:.2%})")
            print(f"Anomaly: {n_anomaly} ({n_anomaly / n_total:.2%})")
            print(len(df))
            print(len(df_final))

            df=df_final  # new dataset with the new portion
            print(len(df))
            '''
            '''

            r = 0.043

            a = df[df['Label'] != '-']  # anomaly
            n = df[df['Label'] == '-']  # normal

            total_anomalies_available = len(a)
            total_normals_available = len(n)

            if total_anomalies_available == 0:
                raise ValueError("No anomaly samples available in df.")

            if total_normals_available == 0:
                raise ValueError("No normal samples available in df.")

            n_a_needed = round(total_normals_available * r / (1 - r))

            if total_anomalies_available >= n_a_needed:
                a_sampled = a.sample(n=n_a_needed, random_state=42)
                n_sampled = n

                print("Enough anomaly samples found.")
                print(f"Sampling {n_a_needed} anomalies and keeping all {total_normals_available} normal samples.")
            else:
                n_n_needed = round(total_anomalies_available * (1 - r) / r)
                n_n_needed = min(n_n_needed, total_normals_available)

                a_sampled = a
                n_sampled = n.sample(n=n_n_needed, random_state=42)

                print("Not enough anomaly samples found.")
                print(f"Keeping all {total_anomalies_available} anomalies and sampling {n_n_needed} normal samples.")

            df_final = pd.concat([a_sampled, n_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

            n_total = len(df_final)
            n_anomaly = (df_final['Label'] != '-').sum()
            n_normal = (df_final['Label'] == '-').sum()

            print(f"Total: {n_total}")
            print(f"Normal: {n_normal} ({n_normal / n_total:.2%})")
            print(f"Anomaly: {n_anomaly} ({n_anomaly / n_total:.2%})")
            print(f"Original df length: {len(df)}")
            print(f"Final df_final length: {len(df_final)}")

            df = df_final
            print(f"Updated df length: {len(df)}")
            '''

            n_total = len(df)
            n_anomaly = (df['Label'] != '-').sum()
            n_normal = (df['Label'] == '-').sum()

            print(f"Total: {n_total}")
            print(f"Normal: {n_normal} ({n_normal / n_total:.2%})")
            print(f"Anomaly: {n_anomaly} ({n_anomaly / n_total:.2%})")
            print(f"Original df length: {len(df)}")
            print(f"Final df_final length: {len(df)}")

            df.sort_values(by=['Timestamp'], inplace=True)  # Ensure order

            # exit()

            print(' length df before windows ' + str(len(df)))

            def process_logs(df, window_size=120):

                df = df.copy()
                df.sort_values(by=['Node_block_id', 'Timestamp'], inplace=True)  # Ensure order

                blocks = []

                for node_id, group in df.groupby('Node_block_id'):
                    entries = group.to_dict('records')  # Convert to list of dicts
                    num_entries = len(entries)
                    block_count = (num_entries + window_size - 1) // window_size  # Number of blocks

                    for i in range(block_count):
                        start_idx = i * window_size
                        end_idx = min(start_idx + window_size, num_entries)
                        block_entries = entries[start_idx:end_idx]

                        # Define block name
                        block_name = f"{node_id}_block_{i}"
                        block_label = "Anomaly" if any(entry['Label'] != '-' for entry in block_entries) else "Normal"

                        for entry in block_entries:
                            entry['Block'] = block_name
                            entry['Updated_Label'] = block_label
                            blocks.append(entry)

                return pd.DataFrame(blocks)

            # Example usage
            df = process_logs(df, window_size=120)  # Block , Updated_Label
            # print(df_processed.head())
            print(' length df after windows ' + str(len(df)))

            print('check....')
            # Separate Normal & Anomaly Logs
            df1 = df.query("Label == '-'").reset_index(drop=True)  # Normal logs
            df2 = df.query("Label != '-'").reset_index(drop=True)  # Anomaly logs
            df_block = df.drop_duplicates(subset=['Block']).reset_index(drop=True)  # Unique Normal Blocks
            df3 = df_block.query("Updated_Label == 'Normal'").reset_index(drop=True)
            df4 = df_block.query("Updated_Label == 'Anomaly'").reset_index(drop=True)

            # Print Dataset Statistics
            print(f"All logs: {x:,}")
            print(f"Normal logs: {len(df1):,}")  # TH_1G : 4,365,033
            print(f"Anomaly logs: {len(df2):,}")  # TH_1G :348,460
            print(f"Unique normal blocks: {len(df3):,}")  # 49,247
            print(f"Unique anomaly blocks: {len(df4):,}")  # 36,251
            print(f"All unique blocks: {len(df_block):,}")
            df = df.drop(columns=['Node_block_id'])
            df = df.rename(columns={'Label': 'Original_Label'})
            # -----------------------------------------------
            # df = df.drop(columns=['Node_block_id', 'Label'])
            # df = df.rename(columns={'Block': 'Node_block_id', 'Updated_Label': 'Label'})
            # -----------------------------------------------
            df = df.rename(columns={'Block': 'Node_block_id', 'Updated_Label': 'Label'})
            df.info()
            print(f"Total: {n_total}")
            print(f"Normal: {n_normal} ({n_normal / n_total:.2%})")
            print(f"Anomaly: {n_anomaly} ({n_anomaly / n_total:.2%})")
            print(f"Original df length: {len(df)}")
            print(f"Final df_final length: {len(df)}")
            print(' save as csv file ....')
            # Save Processed Dataset Efficiently
            df.to_csv(All_dataset_path_as_csv, escapechar='\\', index=False)

        elif dataset == 'OS':
            #  Define dtype mapping for efficient memory usage
            dtype_mapping = {"ADDR": "str", "EventTemplate": "category", "Content": "str", "Date": "str", "Time": "str",
                             "Component": "category", "EventId": "str"}

            df_abnormal = pd.read_csv(f'../datasets/{dataset}/{dataset}_abnormal.log_structured.csv',
                                      dtype=dtype_mapping)
            df_abnormal['Label'] = 'AAAA'

            df_normal_1 = pd.read_csv(f'../datasets/{dataset}/{dataset}_normal1.log_structured.csv',
                                      dtype=dtype_mapping)
            df_normal_1['Label'] = '-'

            df_normal_2 = pd.read_csv(f'../datasets/{dataset}/{dataset}_normal2.log_structured.csv',
                                      dtype=dtype_mapping)
            df_normal_2['Label'] = '-'

            df = pd.concat([df_abnormal, df_normal_1, df_normal_2], ignore_index=True)
            df.info()
            #  Rename columns for consistency
            df = df.rename(columns={"ADDR": "Node_block_id"})

            # Step 1: Replace NaN with "UNKNOWN"
            df['Node_block_id'] = df['Node_block_id'].fillna('UNKNOWN')
            df = fill_unknown_node_block_id(df)

            #  Parse Timestamp Correctly (Format: YYYY-MM-DD-HH.MM.SS.ffffff)
            df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y-%m-%d %H:%M:%S.%f")

            nan_count = df['Content'].isna().sum()
            print(f"Number of NaN values in 'log_message': {nan_count}")

            #  Process EventTemplate
            df["processed_EventTemplate"] = df["EventTemplate"].apply(self.get_dataset_text_content_message)
            df['processed_EventTemplate'].fillna(df['Content'], inplace=True)
            nan_count = df['processed_EventTemplate'].isna().sum()
            print(f"Number of NaN values in 'processed_EventTemplate': {nan_count}")

            df['processed_EventTemplate'] = df['processed_EventTemplate'].astype(str)
            # Select only required columns (memory efficiency)
            df = df[['Timestamp', 'Date', 'Time', 'Content', 'EventId', 'EventTemplate', 'processed_EventTemplate',
                     'Node_block_id', 'Label']]

            # Sampling the datasets:
            block_labels = df[['Node_block_id', 'Label']].drop_duplicates()
            normal_blocks = block_labels[block_labels['Label'] == '-']['Node_block_id']
            anomaly_blocks = block_labels[block_labels['Label'] != '-']['Node_block_id']

            sampled_normal = normal_blocks.sample(frac=0.5, random_state=42)
            sampled_anomaly = anomaly_blocks.sample(frac=0.5, random_state=42)

            selected_blocks = pd.concat([sampled_normal, sampled_anomaly])
            df_half = df[df['Node_block_id'].isin(selected_blocks)].reset_index(drop=True)
            print("Original dataset rows:", len(df))
            print("Half dataset rows:", len(df_half))
            print("Normal blocks selected:", len(sampled_normal))
            print("Anomaly blocks selected:", len(sampled_anomaly))
            df = df_half

            print(' length df before windows ' + str(len(df)))

            def process_logs(df, window_size=120):
                df = df.copy()
                df.sort_values(by=['Node_block_id', 'Timestamp'], inplace=True)  # Ensure order

                blocks = []

                for node_id, group in df.groupby('Node_block_id'):
                    entries = group.to_dict('records')  # Convert to list of dicts
                    num_entries = len(entries)
                    block_count = (num_entries + window_size - 1) // window_size  # Number of blocks

                    for i in range(block_count):
                        start_idx = i * window_size
                        end_idx = min(start_idx + window_size, num_entries)
                        block_entries = entries[start_idx:end_idx]

                        # Define block name
                        block_name = f"{node_id}_block_{i}"
                        block_label = "Anomaly" if any(entry['Label'] != '-' for entry in block_entries) else "Normal"

                        for entry in block_entries:
                            entry['Block'] = block_name
                            entry['Updated_Label'] = block_label
                            blocks.append(entry)

                return pd.DataFrame(blocks)

            # Example usage
            df = process_logs(df, window_size=120)  # Block , Updated_Label
            # print(df_processed.head())
            print(' length df after windows ' + str(len(df)))

            print('check....')
            # Separate Normal & Anomaly Logs
            df1 = df.query("Label == '-'").reset_index(drop=True)  # Normal logs
            df2 = df.query("Label != '-'").reset_index(drop=True)  # Anomaly logs
            df_block = df.drop_duplicates(subset=['Block']).reset_index(drop=True)  # Unique Normal Blocks
            df3 = df_block.query("Updated_Label == 'Normal'").reset_index(drop=True)
            df4 = df_block.query("Updated_Label == 'Anomaly'").reset_index(drop=True)

            # Print Dataset Statistics
            print(f"Normal logs: {len(df1):,}")  # 4,365,033
            print(f"Anomaly logs: {len(df2):,}")  # 348,460
            print(f"Unique normal blocks: {len(df3):,}")  # 49,247
            print(f"Unique anomaly blocks: {len(df4):,}")  # 36,251
            print(f"All unique blocks: {len(df_block):,}")
            df = df.drop(columns=['Node_block_id'])
            df = df.rename(columns={'Label': 'Original_Label'})
            # -----------------------------------------------
            # df = df.drop(columns=['Node_block_id', 'Label'])
            # df = df.rename(columns={'Block': 'Node_block_id', 'Updated_Label': 'Label'})
            # -----------------------------------------------
            df = df.rename(columns={'Block': 'Node_block_id', 'Updated_Label': 'Label'})
            df.info()
            print(' save as csv file ....')
            # Save Processed Dataset Efficiently
            df.to_csv(All_dataset_path_as_csv, escapechar='\\', index=False)

        elif dataset == 'HDO':
            #  Define dtype mapping for efficient memory usage
            dtype_mapping = {"Process": "str", "EventTemplate": "category", "Content": "str", "Date": "str",
                             "Time": "str", "Component": "category", "EventId": "str"}

            df_abnormal = pd.read_csv(f'../datasets/{dataset}/{dataset}_abnormal.log_structured.csv',
                                      dtype=dtype_mapping)
            df_abnormal['Label'] = 'AAAA'

            df_normal = pd.read_csv(f'../datasets/{dataset}/{dataset}_normal.log_structured.csv', dtype=dtype_mapping)
            df_normal['Label'] = '-'

            len_abnormal = len(df_abnormal)
            len_normal = len(df_normal)

            print(f"Number of abnormal rows: {len_abnormal}")
            print(f"Number of normal rows: {len_normal}")
            df_abnormal.info()
            df_normal.info()
            #            exit()

            # --- Sampling ---
            # Take 20,000 normal logs
            #            df_normal_sampled = df_normal.sample(n=20000, random_state=42)

            # Take 5% of abnormal logs
            #            df_abnormal_sampled = df_abnormal.sample(frac=1000, random_state=42)
            df_normal_sampled = df_normal.sample(n=25000, random_state=42)

            # Take exactly 1,000 abnormal logs
            df_abnormal_sampled = df_abnormal.sample(n=1250, random_state=42)
            # --- Final combined dataset ---
            df_final = pd.concat([df_normal_sampled, df_abnormal_sampled], ignore_index=True)
            df_final = df_final.drop(columns=['LineId'])

            df_final.info()

            print("Normal selected:", len(df_normal_sampled))
            print("Abnormal selected:", len(df_abnormal_sampled))
            print("Total:", len(df_final))
            print('')

            df = df_final
            #            df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y-%m-%d %H:%M:%S")
            df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])

            # Process EventTemplate
            df["processed_EventTemplate"] = df["EventTemplate"].apply(self.get_dataset_text_content_message)
            df['processed_EventTemplate'].fillna(df['Content'], inplace=True)
            nan_count = df['processed_EventTemplate'].isna().sum()
            print(f"Number of NaN values in 'processed_EventTemplate': {nan_count}")
            df['processed_EventTemplate'] = df['processed_EventTemplate'].astype(str)

            def process_logs(df, window_size=120):
                df = df.copy()
                df.sort_values(by=['Timestamp'], inplace=True)  # Ensure order

                blocks = []

                for node_id, group in df.groupby('Timestamp'):
                    entries = group.to_dict('records')  # Convert to list of dicts
                    num_entries = len(entries)
                    block_count = (num_entries + window_size - 1) // window_size  # Number of blocks

                    for i in range(block_count):
                        start_idx = i * window_size
                        end_idx = min(start_idx + window_size, num_entries)
                        block_entries = entries[start_idx:end_idx]

                        # Define block name
                        block_name = f"{node_id}_block_{i}"
                        block_label = "Anomaly" if any(entry['Label'] != '-' for entry in block_entries) else "Normal"

                        for entry in block_entries:
                            entry['Block'] = block_name
                            entry['Updated_Label'] = block_label
                            blocks.append(entry)

                return pd.DataFrame(blocks)

            df = process_logs(df, window_size=120)  # Block , Updated_Label

            # Separate Normal & Anomaly Logs
            df1 = df.query("Label == '-'").reset_index(drop=True)  # Normal logs
            df2 = df.query("Label != '-'").reset_index(drop=True)  # Anomaly logs
            df_block = df.drop_duplicates(subset=['Block']).reset_index(drop=True)  # Unique Normal Blocks
            df3 = df_block.query("Updated_Label == 'Normal'").reset_index(drop=True)
            df4 = df_block.query("Updated_Label == 'Anomaly'").reset_index(drop=True)

            # Print Dataset Statistics
            print(f"Normal logs: {len(df1):,}")  # 4,365,033
            print(f"Anomaly logs: {len(df2):,}")  # 348,460
            print(f"Unique normal blocks: {len(df3):,}")  # 49,247
            print(f"Unique anomaly blocks: {len(df4):,}")  # 36,251
            print(f"All unique blocks: {len(df_block):,}")

            df = df.rename(columns={'Label': 'Original_Label'})
            df = df.rename(columns={'Block': 'Node_block_id', 'Updated_Label': 'Label'})
            df.info()

            print(' save as csv file ....')
            # Save Processed Dataset Efficiently
            df.to_csv(All_dataset_path_as_csv, escapechar='\\', index=False)

        elif dataset == 'SP_150MB' or dataset == 'SP_100MB':
            #  Define dtype mapping for efficient memory usage
            dtype_mapping = {"User": "str", "EventTemplate": "category", "Content": "str", "Date": "str", "Time": "str",
                             "Component": "category", "EventId": "str", "Label": "category"}

            df = pd.read_csv(f'../datasets/{dataset}/{dataset}.log_structured.csv', dtype=dtype_mapping)
            df.info()
            x = len(df)

            #  Rename columns for consistency
            df = df.rename(columns={"User": "Node_block_id"})

            # Step 1: Replace NaN with "UNKNOWN"
            df['Node_block_id'] = df['Node_block_id'].fillna('UNKNOWN')
            df = fill_unknown_node_block_id(df)

            #  Parse Timestamp Correctly (Format: YYYY-MM-DD-HH.MM.SS.ffffff)
            df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M:%S")

            nan_count = df['Content'].isna().sum()
            print(f"Number of NaN values in 'log_message': {nan_count}")

            #  Process EventTemplate
            df["processed_EventTemplate"] = df["EventTemplate"].apply(self.get_dataset_text_content_message)
            df['processed_EventTemplate'].fillna(df['Content'], inplace=True)
            nan_count = df['processed_EventTemplate'].isna().sum()
            print(f"Number of NaN values in 'processed_EventTemplate': {nan_count}")

            df['processed_EventTemplate'] = df['processed_EventTemplate'].astype(str)
            # Select only required columns (memory efficiency)
            df = df[['Timestamp', 'Date', 'Time', 'Content', 'EventId', 'EventTemplate', 'processed_EventTemplate',
                     'Node_block_id', 'Label']]

            print(' length df before windows ' + str(len(df)))

            def process_logs(df, window_size=120):
                df = df.copy()
                df.sort_values(by=['Node_block_id', 'Timestamp'], inplace=True)  # Ensure order

                blocks = []

                for node_id, group in df.groupby('Node_block_id'):
                    entries = group.to_dict('records')  # Convert to list of dicts
                    num_entries = len(entries)
                    block_count = (num_entries + window_size - 1) // window_size  # Number of blocks

                    for i in range(block_count):
                        start_idx = i * window_size
                        end_idx = min(start_idx + window_size, num_entries)
                        block_entries = entries[start_idx:end_idx]

                        # Define block name
                        block_name = f"{node_id}_block_{i}"
                        block_label = "Anomaly" if any(entry['Label'] != '-' for entry in block_entries) else "Normal"

                        for entry in block_entries:
                            entry['Block'] = block_name
                            entry['Updated_Label'] = block_label
                            blocks.append(entry)

                return pd.DataFrame(blocks)

            # Example usage
            df = process_logs(df, window_size=120)  # Block , Updated_Label
            # print(df_processed.head())
            print(' length df after windows ' + str(len(df)))

            print('check....')
            # Separate Normal & Anomaly Logs
            df1 = df.query("Label == '-'").reset_index(drop=True)  # Normal logs
            df2 = df.query("Label != '-'").reset_index(drop=True)  # Anomaly logs
            df_block = df.drop_duplicates(subset=['Block']).reset_index(drop=True)  # Unique Normal Blocks
            df3 = df_block.query("Updated_Label == 'Normal'").reset_index(drop=True)
            df4 = df_block.query("Updated_Label == 'Anomaly'").reset_index(drop=True)
            print(f"All logs: {x:,}")  #

            # Print Dataset Statistics
            print(f"Normal logs: {len(df1):,}")  # 4,365,033
            print(f"Anomaly logs: {len(df2):,}")  # 348,460
            print(f"Unique normal blocks: {len(df3):,}")  # 49,247
            print(f"Unique anomaly blocks: {len(df4):,}")  # 36,251
            print(f"All unique blocks: {len(df_block):,}")
            df = df.drop(columns=['Node_block_id'])
            df = df.rename(columns={'Label': 'Original_Label'})
            # -----------------------------------------------
            # df = df.drop(columns=['Node_block_id', 'Label'])
            # df = df.rename(columns={'Block': 'Node_block_id', 'Updated_Label': 'Label'})
            # -----------------------------------------------
            df = df.rename(columns={'Block': 'Node_block_id', 'Updated_Label': 'Label'})
            df.info()
            print(' save as csv file ....')
            # Save Processed Dataset Efficiently
            df.to_csv(All_dataset_path_as_csv, escapechar='\\', index=False)



        elif dataset == 'SP_150MB_ratio' or dataset == 'SP_100MB_ratio' or dataset == 'SP_1G_ratio':
            #  Define dtype mapping for efficient memory usage
            dtype_mapping = {"User": "str", "EventTemplate": "category", "Content": "str", "Date": "str", "Time": "str",
                             "Component": "category", "EventId": "str", "Label": "category"}

            df = pd.read_csv(f'../datasets/{dataset}/{dataset}.log_structured.csv', dtype=dtype_mapping)
            df.info()
            x = len(df)

            #  Rename columns for consistency
            df = df.rename(columns={"User": "Node_block_id"})

            # Step 1: Replace NaN with "UNKNOWN"
            df['Node_block_id'] = df['Node_block_id'].fillna('UNKNOWN')
            df = fill_unknown_node_block_id(df)

            #  Parse Timestamp Correctly (Format: YYYY-MM-DD-HH.MM.SS.ffffff)
            df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M:%S")

            nan_count = df['Content'].isna().sum()
            print(f"Number of NaN values in 'log_message': {nan_count}")

            #  Process EventTemplate
            df["processed_EventTemplate"] = df["EventTemplate"].apply(self.get_dataset_text_content_message)
            df['processed_EventTemplate'].fillna(df['Content'], inplace=True)
            nan_count = df['processed_EventTemplate'].isna().sum()
            print(f"Number of NaN values in 'processed_EventTemplate': {nan_count}")

            df['processed_EventTemplate'] = df['processed_EventTemplate'].astype(str)
            # Select only required columns (memory efficiency)
            df = df[['Timestamp', 'Date', 'Time', 'Content', 'EventId', 'EventTemplate', 'processed_EventTemplate',
                     'Node_block_id', 'Label']]

            n_total = len(df)
            n_anomaly = (df['Label'] != '-').sum()
            n_normal = (df['Label'] == '-').sum()

            print(f"Total: {n_total}")
            print(f"Normal: {n_normal} ({n_normal / n_total:.2%})")
            print(f"Anomaly: {n_anomaly} ({n_anomaly / n_total:.2%})")
            print(len(df))

            # -------take anomaly ration and normal ration based on original dataset. In Spirit 2.6% Anomaly
            r = 0.026

            a = df[df['Label'] != '-']  # anomaly
            n = df[df['Label'] == '-']  # normal

            # take all available normal rows
            n_n = len(n)

            # compute how many anomalies are needed so anomaly ratio becomes 2.6%
            n_a = int(n_n * r / (1 - r))

            # sample anomalies
            df_final = pd.concat([a.sample(n=n_a, random_state=42), n]).sample(frac=1, random_state=42).reset_index(
                drop=True)

            # check
            n_total = len(df_final)
            n_anomaly = (df_final['Label'] != '-').sum()
            n_normal = (df_final['Label'] == '-').sum()

            print(f"Total: {n_total}")
            print(f"Normal: {n_normal} ({n_normal / n_total:.2%})")
            print(f"Anomaly: {n_anomaly} ({n_anomaly / n_total:.2%})")
            print(len(df))
            print(len(df_final))
            df = df_final  # new dataset with the new portion
            print(len(df))

            # exit()
            # ============================================================

            print(' length df before windows ' + str(len(df)))

            def process_logs(df, window_size=120):
                df = df.copy()
                df.sort_values(by=['Node_block_id', 'Timestamp'], inplace=True)  # Ensure order

                blocks = []

                for node_id, group in df.groupby('Node_block_id'):
                    entries = group.to_dict('records')  # Convert to list of dicts
                    num_entries = len(entries)
                    block_count = (num_entries + window_size - 1) // window_size  # Number of blocks

                    for i in range(block_count):
                        start_idx = i * window_size
                        end_idx = min(start_idx + window_size, num_entries)
                        block_entries = entries[start_idx:end_idx]

                        # Define block name
                        block_name = f"{node_id}_block_{i}"
                        block_label = "Anomaly" if any(entry['Label'] != '-' for entry in block_entries) else "Normal"

                        for entry in block_entries:
                            entry['Block'] = block_name
                            entry['Updated_Label'] = block_label
                            blocks.append(entry)

                return pd.DataFrame(blocks)

            # Example usage
            df = process_logs(df, window_size=120)  # Block , Updated_Label
            # print(df_processed.head())
            print(' length df after windows ' + str(len(df)))

            print('check....')
            # Separate Normal & Anomaly Logs
            df1 = df.query("Label == '-'").reset_index(drop=True)  # Normal logs
            df2 = df.query("Label != '-'").reset_index(drop=True)  # Anomaly logs
            df_block = df.drop_duplicates(subset=['Block']).reset_index(drop=True)  # Unique Normal Blocks
            df3 = df_block.query("Updated_Label == 'Normal'").reset_index(drop=True)
            df4 = df_block.query("Updated_Label == 'Anomaly'").reset_index(drop=True)
            print(f"All logs: {x:,}")  #

            # Print Dataset Statistics
            print(f"Normal logs: {len(df1):,}")  # 4,365,033
            print(f"Anomaly logs: {len(df2):,}")  # 348,460
            print(f"Unique normal blocks: {len(df3):,}")  # 49,247
            print(f"Unique anomaly blocks: {len(df4):,}")  # 36,251
            print(f"All unique blocks: {len(df_block):,}")
            df = df.drop(columns=['Node_block_id'])
            df = df.rename(columns={'Label': 'Original_Label'})
            # -----------------------------------------------
            # df = df.drop(columns=['Node_block_id', 'Label'])
            # df = df.rename(columns={'Block': 'Node_block_id', 'Updated_Label': 'Label'})
            # -----------------------------------------------
            df = df.rename(columns={'Block': 'Node_block_id', 'Updated_Label': 'Label'})
            df.info()

            print(' save as csv file ....')
            # Save Processed Dataset Efficiently
            df.to_csv(All_dataset_path_as_csv, escapechar='\\', index=False)
