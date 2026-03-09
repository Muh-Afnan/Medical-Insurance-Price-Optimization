import pandas as pd
from models.train import ModelTrainer
import os

def run_training_pipeline(data_path='dataset/insurance.csv'):
    """
    Production-grade training pipeline.
    """
    print("Initializing Training Pipeline...")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows of data.")

    # Initialize trainer
    trainer = ModelTrainer(model_dir='models')
    
    # Train and evaluate
    metrics = trainer.train(df)
    
    print("Training Pipeline Completed successfully.")
    return metrics

if __name__ == "__main__":
    run_training_pipeline()
