#!/usr/bin/env python3
"""
Script to automatically create a DVC ML project structure
"""

import os
from pathlib import Path


def create_dvc_project(project_name="dvc-ml-project"):
    """
    Create a complete DVC ML project structure with all necessary files and folders.
    
    Args:
        project_name (str): Name of the project directory to create
    """
    # Define the project structure
    structure = {
        "data/raw": [],
        "data/processed": [],
        "src": ["preprocess.py", "train.py", "evaluate.py"],
        "models": [],
        "metrics": [],
        "": ["params.yaml", "requirements.txt", "README.md"]
    }
    
    # Create project root directory
    project_root = Path(project_name)
    project_root.mkdir(exist_ok=True)
    print(f"✓ Created project directory: {project_name}/")
    
    # Create folder structure
    for folder_path, files in structure.items():
        if folder_path:
            full_path = project_root / folder_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {folder_path}/")
        
        # Create empty files in each directory
        for file_name in files:
            file_path = project_root / folder_path / file_name if folder_path else project_root / file_name
            file_path.touch(exist_ok=True)
            print(f"  ✓ Created empty file: {file_path.relative_to(project_root)}")
    
    print(f"\n✅ Project structure created successfully at: {project_root.absolute()}")



if __name__ == "__main__":
    import sys
    
    # Get project name from command line or use default
    project_name = sys.argv[1] if len(sys.argv) > 1 else "dvc-ml-project"
    
    create_dvc_project(project_name)
