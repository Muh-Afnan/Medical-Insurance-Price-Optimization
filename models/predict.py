import os
import joblib
import pandas as pd
import numpy as np

class InsurancePredictor:
    """
    Handles loading the trained model pipeline and making predictions.
    """
    def __init__(self, model_path='models/insurance_model_pipeline.joblib'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please run training first.")
        
        self.pipeline = joblib.load(model_path)
        print(f"Loaded model pipeline from {model_path}")

    def predict(self, data):
        """
        Make predictions for input data (dict or DataFrame).
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input data must be a dictionary or pandas DataFrame")

        # The pipeline handles engineering, preprocessing, and the regressor
        y_pred_log = self.pipeline.predict(df)
        
        # Invert the log transformation applied during training
        y_pred = np.expm1(y_pred_log)
        
        return y_pred

    def get_feature_importance(self):
        """
        Extract feature importance from the trained model.
        Returns a DataFrame with features and their importance scores.
        """
        try:
            regressor = self.pipeline.named_steps['regressor']
            preprocessor = self.pipeline.named_steps['preprocessor']
            
            # Try to get feature names from preprocessor
            # Note: handle_unknown='ignore' might make feature names slightly complex
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                # Fallback if get_feature_names_out fails
                feature_names = [f"Feature {i}" for i in range(regressor.n_features_in_)]
                
            importances = regressor.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            return importance_df
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Test prediction
    try:
        predictor = InsurancePredictor()
        sample_input = {
            'age': 19,
            'sex': 'female',
            'bmi': 27.9,
            'children': 0,
            'smoker': 'yes',
            'region': 'southwest'
        }
        prediction = predictor.predict(sample_input)
        print(f"Predicted Charges: ${prediction[0]:.2f}")
    except Exception as e:
        print(f"Error making prediction: {e}")
