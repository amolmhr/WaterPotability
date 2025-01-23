import pandas as pd
import numpy as np
import os
import logging
from data.load_data import load_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Constants
DATA_PATH = "D:/water_potability/WaterPotability/data/raw/water_quality.csv"
PROCESSED_PATH = "D:/water_potability/WaterPotability/data/processed/processed_data.csv"
CATEGORICAL_COLUMNS = []  # Replace with actual column names
NUMERICAL_COLUMNS = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity']       # Replace with actual column names


def handle_missing_values_mean(df,columnlist:list) -> pd.DataFrame:
    """
    Handles missing values in the dataset by filling it with mean.
    Args:
        df (pd.DataFrame): Input data.
    Returns:
        pd.DataFrame: Data with missing values handled.
    """
    logging.info("Handling missing values with mean...")
    for column in columnlist:
        df[column] = df[column].fillna(df[column].mean())
    
    return df

def handle_missing_values_median(df,columnlist:list) -> pd.DataFrame:
    """
    Handles missing values in the dataset by filling it with median.
    Args:
        df (pd.DataFrame): Input data.
    Returns:
        pd.DataFrame: Data with missing values handled.
    """
    logging.info("Handling missing values with mean...")
    for column in columnlist:
        df[column] = df[column].fillna(df[column].median())
    
    return df

def handle_missing_values_mode(df, columnlist:list) -> pd.DataFrame:
    """
    Handles missing values in the dataset by filling it with mode.
    Args:
        df (pd.DataFrame): Input data.
    Returns:
        pd.DataFrame: Data with missing values handled.
    """
    logging.info("Handling missing values with mode...")
    for column in columnlist:
        df[column] = df[column].fillna(df[column].mode()[0])
    
    return df

def encode_categorical_columns(df, columns):
    """
    Encodes categorical columns using one-hot encoding.
    Args:
        df (pd.DataFrame): Input data.
        columns (list): List of categorical columns to encode.
    Returns:
        pd.DataFrame: Data with encoded categorical columns.
    """
    logging.info("Encoding categorical columns...")
    return pd.get_dummies(df, columns=columns, drop_first=True)

def normalize_numerical_columns(df, columns):
    """
    Normalizes numerical columns using z-score normalization.
    Args:
        df (pd.DataFrame): Input data.
        columns (list): List of numerical columns to normalize.
    Returns:
        pd.DataFrame: Data with normalized numerical columns.
    """
    logging.info("Normalizing numerical columns...")
    for col in columns:
        if col in df:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def standarize_numerical_columns(df, columns):
    """
    Normalizes numerical columns using z-score normalization.
    Args:
        df (pd.DataFrame): Input data.
        columns (list): List of numerical columns to normalize.
    Returns:
        pd.DataFrame: Data with normalized numerical columns.
    """
    logging.info("Normalizing numerical columns...")
    for col in columns:
        if col in df:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def save_data(df, filepath):
    """
    Saves the processed DataFrame to a CSV file.
    Args:
        df (pd.DataFrame): Data to save.
        filepath (str): Output file path.
    """
    logging.info(f"Saving processed data to {filepath}")
    df.to_csv(filepath, index=False)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to orchestrate the data preprocessing workflow.
    """
    try:
        
        
        # Step 2: Handle missing values
        data = handle_missing_values_median(data,['Trihalomethanes','Sulfate','ph'])
        
        # Step 3: Normalize numerical columns
        data = normalize_numerical_columns(data, NUMERICAL_COLUMNS)
        
        # Step 4: Save the processed data
        save_data(data, PROCESSED_PATH)
        
        logging.info("Data preprocessing completed successfully!")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    data = load_data(DATA_PATH)
    preprocess_data()
