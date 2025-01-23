import pandas as pd
import numpy as np
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a given file path.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    try:
        data = pd.read_csv(file_path)
        print(f"Data successfully loaded from {file_path}")
        return data
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the data: {e}")