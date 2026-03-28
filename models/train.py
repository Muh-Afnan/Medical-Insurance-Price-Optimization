import os
import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline
from preprocessing.preprocessing import preprocessing_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor

class ModelTrainer:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_model = None
        self.pipeline = None
    
    def linear_regression_model(x_train,preprocessing_pipeline, y_train):
        model_pipeline = make_pipeline(preprocessing_pipeline, LinearRegression())
        model_pipeline.fit(x_train, y_train)
        return model_pipeline

    def train(self, df, target='charges', test_size=0.2, random_state=42):
        print(f"Starting training process...")
        
        X = df.drop(columns=[target])
        y = df[target]
        
        x_train, x_test, y_train, y_test, preprocessing = preprocessing_pipeline(X,y)

        model_pipeline = self.linear_regression_model(x_train,preprocessing,y_train)

        model_path = os.path.join(self.model_dir, 'insurance_model_pipeline.joblib')
        joblib.dump(model_pipeline, model_path)
        print(f"Model pipeline saved to {model_path}")