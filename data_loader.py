import pandas as pd
import numpy as np
import os
from tdc.multi_pred import DTI
from sklearn.model_selection import train_test_split

def download_TDC_data(dataset_names):
    """
    Download datasets from TDC.

    Parameters:
        dataset_names (list): List of dataset names to fetch (e.g., ['BindingDB_Kd', 'DAVIS', 'KIBA']).

    Returns:
        pd.DataFrame: Combined dataset containing SMILES strings, protein sequences, and labels.
    """
    combined_data = []
    for dataset_name in dataset_names:
        try:
            print(f"Downloading {dataset_name} from TDC...")
            data = DTI(name=dataset_name).get_data()
            combined_data.append(data)
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
    return pd.concat(combined_data, ignore_index=True)

def preprocess_data(data, drug_col="Drug_ID", target_col="Target_ID", label_col="Y"):
    """
    Preprocess raw data from TDC by renaming columns and handling missing values.

    Parameters:
        data (pd.DataFrame): Raw data from TDC.
        drug_col (str): Column name for drug SMILES.
        target_col (str): Column name for target sequences.
        label_col (str): Column name for interaction labels.

    Returns:
        pd.DataFrame: Processed dataset.
    """
    print("Preprocessing data...")
    # Rename columns for consistency
    data = data.rename(columns={drug_col: "Drug", target_col: "Target_Sequence", label_col: "Label"})
    # Drop missing values
    data = data.dropna(subset=["Drug", "Target_Sequence", "Label"])
    print(f"Processed dataset: {len(data)} samples.")
    return data

def split_data(data, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Parameters:
        data (pd.DataFrame): Processed dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets.
    """
    print("Splitting data into train and test sets...")
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    print(f"Training samples: {len(train_data)}, Testing samples: {len(test_data)}")
    return train_data, test_data

def save_data(train_data, test_data, output_dir="processed_data"):
    """
    Save processed training and testing datasets as CSV files.

    Parameters:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        output_dir (str): Directory to save the processed datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    print(f"Saving training data to {train_path}...")
    train_data.to_csv(train_path, index=False)

    print(f"Saving testing data to {test_path}...")
    test_data.to_csv(test_path, index=False)
    print("Data saved successfully.")

def load_data(dataset_names=["BindingDB_Kd", "DAVIS", "KIBA"]):
    """
    Complete data loading pipeline: download, preprocess, split, and save.

    Parameters:
        dataset_names (list): List of dataset names to fetch.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets.
    """
    # Download datasets
    raw_data = download_TDC_data(dataset_names)

    # Preprocess dataset
    processed_data = preprocess_data(raw_data)

    # Split data into train and test sets
    train_data, test_data = split_data(processed_data)

    # Save processed datasets
    save_data(train_data, test_data)

    return train_data, test_data

if __name__ == "__main__":
    dataset_names = ["BindingDB_Kd", "DAVIS", "KIBA"]  # List of datasets to include
    train_data, test_data = load_data(dataset_names)