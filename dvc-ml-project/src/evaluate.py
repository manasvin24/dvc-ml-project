#!/usr/bin/env python3
"""
Model evaluation script
"""

import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import yaml
import mlflow


def load_params():
    """Load parameters from params.yaml"""
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
            return params if params else {}
    except FileNotFoundError:
        print("Warning: params.yaml not found, using defaults")
        return {}


def evaluate_model():
    """
    Evaluation pipeline:
    1. Load model
    2. Load processed data
    3. Compute accuracy, F1, precision, recall
    4. Write metrics to metrics/scores.json
    """
    # Set MLflow experiment
    mlflow.set_experiment("dvc-ml-project")
    
    # Load parameters
    params = load_params()
    train_params = params.get('train', {})
    random_state = train_params.get('random_state', 42)
    test_size = train_params.get('test_size', 0.2)
    
    # Step 1: Load model
    print("Step 1: Loading model")
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("  Model loaded from models/model.pkl")
    
    # Step 2: Load processed data
    print("Step 2: Loading processed data")
    df = pd.read_csv('data/processed/features.csv')
    
    # Drop non-numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_numeric = df[numeric_cols]
    
    # Split features and target
    X = df_numeric.iloc[:, :-1]
    y = df_numeric.iloc[:, -1]
    
    # Use same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Test set size: {len(X_test)}")
    
    # Step 3: Compute metrics
    print("Step 3: Computing metrics")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    # Step 4: Write metrics
    print("Step 4: Saving metrics")
    Path('metrics').mkdir(exist_ok=True)
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }
    
    with open('metrics/scores.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("  Metrics saved to metrics/scores.json")
    
    # Log metrics to MLflow (find or create run)
    with mlflow.start_run():
        mlflow.log_metric("eval_accuracy", accuracy)
        mlflow.log_metric("eval_f1_score", f1)
        mlflow.log_metric("eval_precision", precision)
        mlflow.log_metric("eval_recall", recall)
        mlflow.log_artifact("metrics/scores.json")
    with open('metrics/scores.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("  Metrics saved to metrics/scores.json")


if __name__ == "__main__":
    evaluate_model()
