import pandas as pd
from models.predict import InsurancePredictor
import os
import sys

def run_inference_pipeline(input_data=None):
    """
    Production-grade inference pipeline.
    """
    print("Initializing Inference Pipeline...")
    
    try:
        predictor = InsurancePredictor(model_path='models/insurance_model_pipeline.joblib')
        
        if input_data is None:
            # For demonstration, use a sample if none provided
            input_data = {
                'age': 33,
                'sex': 'male',
                'bmi': 22.7,
                'children': 0,
                'smoker': 'no',
                'region': 'northwest'
            }
            print("No input provided, using sample data.")

        predictions = predictor.predict(input_data)
        
        if isinstance(input_data, dict):
            print(f"Prediction: ${predictions[0]:.2f}")
        else:
            print(f"Generated {len(predictions)} predictions.")
            
        return predictions

    except Exception as e:
        print(f"Error in Inference Pipeline: {e}")
        return None

if __name__ == "__main__":
    run_inference_pipeline()
