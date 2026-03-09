import os
import pandas as pd
import numpy as np
import argparse
from models.predict import InsurancePredictor

def run_batch_inference(input_path, output_path):
    """
    Production-grade batch inference pipeline.
    """
    print(f"Starting batch inference for: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    try:
        # Initialize predictor
        predictor = InsurancePredictor(model_path='models/insurance_model_pipeline.joblib')
        
        # Load input data
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows of data.")

        # Make predictions
        predictions = predictor.predict(df)
        
        # Add predictions to the dataframe
        df['predicted_charges'] = predictions
        
        # Save output
        df.to_csv(output_path, index=False)
        print(f"Batch inference complete. Results saved to {output_path}")
        
        return df

    except Exception as e:
        print(f"Error in Batch Inference Pipeline: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Insurance Batch Inference")
    parser.add_argument("--input", type=str, default="dataset/insurance.csv", help="Path to input CSV")
    parser.add_argument("--output", type=str, default="dataset/predictions.csv", help="Path to output CSV")
    
    args = parser.parse_args()
    
    # Ensure current directory is in PYTHONPATH for relative imports if run as script
    # (Though we expect to run this with $env:PYTHONPATH=".")
    
    run_batch_inference(args.input, args.output)
