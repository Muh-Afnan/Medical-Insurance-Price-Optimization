import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from features.feature_pipline import create_feature_pipeline

class ModelTrainer:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_model = None
        self.pipeline = None

    def train(self, df, target='charges', test_size=0.2, random_state=42):
        print(f"Starting training process...")
        
        X = df.drop(columns=[target])
        y = df[target]
        
        # Log transformation for target as recommended in EDA
        y_log = np.log1p(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=test_size, random_state=random_state
        )
        
        # Create pipeline factory
        pipeline_factory = create_feature_pipeline()
        # The factory returns the full pipeline including engineer and preprocessor
        self.pipeline = pipeline_factory(X_train)
        
        # Search space for models
        # We tune the regressor part specifically
        models = {
            'RandomForest': {
                'model': RandomForestRegressor(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 4]
                }
            }
        }
        
        best_score = -np.inf
        
        # Transform data once with engineer to speed up CV
        engineer = self.pipeline.named_steps['engineer']
        X_train_eng = engineer.fit_transform(X_train)
        
        for name, config in models.items():
            print(f"Tuning {name}...")
            
            # Preprocessor needs to be part of CV if we want to be strict, 
            # but usually it's fine to fit it once on X_train_eng.
            preprocessor = self.pipeline.named_steps['preprocessor']
            X_train_proc = preprocessor.fit_transform(X_train_eng)
            
            search = RandomizedSearchCV(
                config['model'], config['params'], n_iter=5, 
                cv=3, scoring='r2', n_jobs=-1, random_state=random_state
            )
            
            search.fit(X_train_proc, y_train)
            
            if search.best_score_ > best_score:
                best_score = search.best_score_
                self.best_model = search.best_estimator_
                self.best_model_name = name
                print(f"New best model: {name} (R2: {best_score:.4f})")
        
        # Final Evaluation
        X_test_eng = engineer.transform(X_test)
        X_test_proc = self.pipeline.named_steps['preprocessor'].transform(X_test_eng)
        y_pred = self.best_model.predict(X_test_proc)
        
        # Invert log transform for metrics
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
        
        metrics = {
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test_orig, y_pred_orig),
            'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        }
        
        print(f"Training Finished. Best Model: {self.best_model_name}")
        print(f"Test Metrics: {metrics}")
        
        # Save artifacts
        self.save_artifacts()
        return metrics

    def save_artifacts(self):
        # We save the full logic as a single pipeline object
        final_pipeline = Pipeline(steps=[
            ('engineer', self.pipeline.named_steps['engineer']),
            ('preprocessor', self.pipeline.named_steps['preprocessor']),
            ('regressor', self.best_model)
        ])
        
        model_path = os.path.join(self.model_dir, 'insurance_model_pipeline.joblib')
        joblib.dump(final_pipeline, model_path)
        print(f"Model pipeline saved to {model_path}")

if __name__ == "__main__":
    import sys
    # Basic CLI for testing
    csv_path = 'dataset/insurance.csv'
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found at {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    trainer = ModelTrainer()
    trainer.train(df)
