import os
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from data.load_data import load_data
from data.preprocess import preprocess_data

# Configure logging
logging.basicConfig(
    filename="D:/water_potability/WaterPotability/logs/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Prepare a list to store metrics for saving into a CSV
all_model_metrics = []

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_single_model(model, X_train, y_train):
    """Train a single machine learning model."""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the performance of a machine learning model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": confusion.tolist(),
    }
    return metrics

def save_model(model, filepath):
    """Save a machine learning model to a file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")

def train_all_models(data):
    logging.info("Starting the training pipeline.")
    try:
        X = data.drop("Potability", axis=1)
        y = data["Potability"]

        # Split the data into training and testing sets
        logging.info("Splitting data into training and testing sets.")
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Define the models
        models = [
            LogisticRegression(),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            GaussianNB(),
            GradientBoostingClassifier(),
            AdaBoostClassifier(),
            SVC(),
            KNeighborsClassifier()
        ]

        # Train, evaluate, and save each model
        for model in models:
            model_name = type(model).__name__
            logging.info(f"Training model: {model_name}")
            trained_model = train_single_model(model, X_train, y_train)
            metrics = evaluate_model(trained_model, X_test, y_test)

            # Log evaluation metrics to the training log file
            logging.info(f"Model: {model_name}")
            logging.info(f"Accuracy: {metrics['accuracy']}")
            logging.info(f"Precision: {metrics['precision']}")
            logging.info(f"Recall: {metrics['recall']}")
            logging.info(f"F1-score: {metrics['f1']}")
            logging.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
            logging.info(f"Classification Report:\n{metrics['classification_report']}")

            # Save metrics to the list for CSV saving later
            all_model_metrics.append({
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1"]
            })

            # Optionally save the trained model
            # save_model(trained_model, f"models/{model_name}.joblib")

        # Optionally save metrics as CSV after all models are trained
        metrics_df = pd.DataFrame(all_model_metrics)
        metrics_df.to_csv("model_metrics.csv", index=False)

        logging.info("Training pipeline completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    data = load_data("D:/water_potability/WaterPotability/data/processed/processed_data.csv")   
    train_all_models(data)
