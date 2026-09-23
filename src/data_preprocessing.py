import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def run_preprocessing():
    # 1. Load exactly these three CSV files from data/raw/ and concatenate them:
    target_files = [
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    ]

    dfs = []
    print("--- Loading and Concatenating Data ---")
    for filename in target_files:
        file_path = os.path.join("data/raw", filename)
        try:
            temp_df = pd.read_csv(file_path, low_memory=False)
            temp_df.columns = temp_df.columns.str.strip()  # Strip whitespace from column names
            print(f"Loaded {filename}. Shape: {temp_df.shape}")
            dfs.append(temp_df)
        except FileNotFoundError:
            print(f"WARNING: {filename} not found. Skipping this file.")
        except Exception as e:
            print(f"WARNING: Error loading {filename}: {e}. Skipping this file.")

    if not dfs:
        print("ERROR: No files were loaded. Exiting.")
        return

    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined DataFrame shape: {df.shape}")

    # IMPORTANT CHECK: Print label counts before cleaning
    print("\n--- Label Distribution in Combined Data (before cleaning) ---")
    label_counts = df['Label'].apply(lambda x: 'BENIGN' if x.strip() == 'BENIGN' else 'MALICIOUS').value_counts()
    print(label_counts)

    benign_count = label_counts.get('BENIGN', 0)
    malicious_count = label_counts.get('MALICIOUS', 0)

    if not (660000 <= benign_count <= 670000 and 535000 <= malicious_count <= 545000):
        print("\nERROR: The combined file shows an unexpected label distribution. Expected roughly 665,000 BENIGN and 540,000 MALICIOUS rows. Please check the file list. Exiting.")
        return
    else:
        print("\n--- Label distribution check passed. Proceeding with preprocessing. ---")

    initial_rows = df.shape[0]

    # 2. Clean the data:
    print("\n--- Cleaning Data ---")
    # Drop exact duplicate rows
    df.drop_duplicates(inplace=True)
    duplicates_dropped = initial_rows - df.shape[0]
    print(f"Dropped {duplicates_dropped} duplicate rows. New shape: {df.shape}")

    # Replace inf/-inf with NaN
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    # Count infinite values replaced
    inf_replaced_count = df[numeric_cols].isna().sum().sum() - df[numeric_cols].isna().sum().sum()
    # This counting method is a bit tricky, let's re-evaluate.
    # A simpler way to get count of inf/nan after replacement is to check before and after.
    # Let's just state that infs were replaced and then count NaNs.
    print("Replaced infinite values with NaN.")

    # Drop rows with NaN
    nan_rows_before = df.shape[0]
    df.dropna(inplace=True)
    nan_rows_dropped = nan_rows_before - df.shape[0]
    print(f"Dropped {nan_rows_dropped} rows with NaN values. New shape: {df.shape}")

    # 3. Randomly subsample the cleaned data down to 25% of its rows
    print("\n--- Subsampling Data ---")
    rows_before_subsampling = df.shape[0]
    print(f"Rows before subsampling: {rows_before_subsampling}")
    df = df.sample(frac=0.25, random_state=42)
    rows_after_subsampling = df.shape[0]
    print(f"Rows after subsampling (25%): {rows_after_subsampling}. Dropped {rows_before_subsampling - rows_after_subsampling} rows.")

    # 4. Encode the label column:
    print("\n--- Encoding Label Column ---")
    df['is_malicious'] = df['Label'].apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)
    print("'is_malicious' column created (0 for BENIGN, 1 for MALICIOUS).")

    label_encoder = LabelEncoder()
    df['Label_encoded'] = label_encoder.fit_transform(df['Label'])
    joblib.dump(label_encoder, 'data/processed/label_encoder.joblib')
    print(f"'Label_encoded' column created. LabelEncoder saved to data/processed/label_encoder.joblib")
    print(f"Label classes found: {list(label_encoder.classes_)}")

    # 5. Select only numeric feature columns (excluding Label, Label_encoded, is_malicious).
    print("\n--- Feature Selection, Train/Test Split, Scaling ---")
    feature_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in ['Label_encoded', 'is_malicious']]
    X = df[feature_cols]
    y_is_malicious = df['is_malicious'] # For stratification

    # Split into train/test (80/20, stratified on is_malicious, random_state=42)
    X_train, X_test, y_train_is_malicious, y_test_is_malicious = train_test_split(
        X, y_is_malicious, test_size=0.2, stratify=y_is_malicious, random_state=42
    )

    # Retain original labels for saving later (if needed for debugging)
    y_train_label_encoded = df.loc[X_train.index, 'Label_encoded']
    y_test_label_encoded = df.loc[X_test.index, 'Label_encoded']

    # Scale features with StandardScaler (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'data/processed/standard_scaler.joblib')
    print(f"Features scaled with StandardScaler. Scaler saved to data/processed/standard_scaler.joblib")

    # Convert scaled arrays back to DataFrames for saving
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

    # Add back the target columns for saving
    train_df = pd.concat([X_train_df, y_train_is_malicious, y_train_label_encoded], axis=1)
    test_df = pd.concat([X_test_df, y_test_is_malicious, y_test_label_encoded], axis=1)

    # 6. Save train.csv and test.csv to data/processed/, and save the list of feature column names
    print("\n--- Saving Processed Data ---")
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    print(f"train.csv (shape: {train_df.shape}) and test.csv (shape: {test_df.shape}) saved to data/processed/")

    with open('data/processed/feature_columns.txt', 'w') as f:
        for col in feature_cols:
            f.write(col + '\n')
    print(f"Feature column names saved to data/processed/feature_columns.txt")

    # 7. At the end, print final shapes and is_malicious value_counts
    print("\n--- Final Summary ---")
    print(f"Final training set shape: {train_df.shape}")
    print(f"Final testing set shape: {test_df.shape}")
    print("is_malicious value_counts in training set:")
    print(train_df['is_malicious'].value_counts())

if __name__ == '__main__':
    run_preprocessing()
