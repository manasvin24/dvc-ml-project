#!/usr/bin/env python3
"""
Model training script
"""

import pandas as pd
import yaml
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn


def load_params():
    """Load parameters from params.yaml"""
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
            return params if params else {}
    except FileNotFoundError:
        print("Warning: params.yaml not found, using defaults")
        return {}


def train_model():
    """
    Training pipeline:
    1. Read processed data
    2. Read hyperparameters from params.yaml
    3. Train a simple model
    4. Save model to models/model.pkl
    """
    # Set MLflow experiment
    mlflow.set_experiment("dvc-ml-project")
    
    # Load parameters
    params = load_params()
    train_params = params.get('train', {})
    
    # Default hyperparameters
    n_estimators = train_params.get('n_estimators', 100)
    max_depth = train_params.get('max_depth', 10)
    random_state = train_params.get('random_state', 42)
    test_size = train_params.get('test_size', 0.2)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)
        
        # Step 1: Read processed data
        print("Step 1: Loading processed data")
        df = pd.read_csv('data/processed/features.csv')
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        mlflow.log_param("dataset_rows", len(df))
        mlflow.log_param("dataset_cols", len(df.columns))
        
        # Step 2: Prepare features and target
        print("Step 2: Preparing features")
        
        # Drop non-numeric columns (datetime, text columns)
        numeric_cols = df.select_dtypes(include=['number']).columns
        df_numeric = df[numeric_cols]
        print(f"  Using {len(numeric_cols)} numeric features")
        mlflow.log_param("num_features", len(numeric_cols) - 1)         
        
        # Split features and target (assuming last numeric column is target)
        X = df_numeric.iloc[:, :-1]
        y = df_numeric.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size_actual", len(X_test))   
        
        # Step 3: Train model
        print("Step 3: Training model")
        print(f"  Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"  Train accuracy: {train_score:.4f}")
        print(f"  Test accuracy: {test_score:.4f}")
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)
        
        # Log model with MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Step 4: Save model
        print("Step 4: Saving model")
        Path('models').mkdir(exist_ok=True)
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("  Model saved to models/model.pkl")


if __name__ == "__main__":
    train_model()
