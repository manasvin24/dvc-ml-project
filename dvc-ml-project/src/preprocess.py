#!/usr/bin/env python3
"""
Data preprocessing script
"""

import pandas as pd
import yaml
from pathlib import Path


def load_params():
    """Load parameters from params.yaml"""
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
            return params if params else {}
    except FileNotFoundError:
        print("Warning: params.yaml not found, using defaults")
        return {}


def preprocess_data():
    """
    Preprocessing pipeline:
    1. Read raw CSV
    2. Clean data (remove nulls, duplicates)
    3. Split features/labels
    4. Write to data/processed/features.csv
    """
    # Load parameters (optional)
    all_params = load_params()
    params = all_params.get('preprocess', {})
    
    # Step 1: Read raw data (Excel file)
    print("Step 1: Loading raw data")
    df = pd.read_excel('data/raw/dataset.csv', engine='openpyxl')
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Step 2: Clean data
    print("\nStep 2: Cleaning data")
    initial_rows = len(df)
    
    # Remove null values
    df = df.dropna()
    print(f"  Removed {initial_rows - len(df)} rows with null values")
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {initial_rows - len(df)} duplicate rows")
    
    # Step 3: Split features/labels
    print("\nStep 3: Splitting features and labels")
    # Assuming last column is the label/target
    feature_columns = df.columns[:-1]
    label_column = df.columns[-1]
    
    print(f"  Features: {len(feature_columns)} columns")
    print(f"  Label: {label_column}")
    
    # Step 4: Write processed data
    print("\nStep 4: Saving processed data")
    output_path = Path('data/processed/features.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"  ✓ Processed data saved to {output_path}")
    print(f"  Final dataset: {len(df)} rows, {len(df.columns)} columns")


if __name__ == '__main__':
    preprocess_data()
