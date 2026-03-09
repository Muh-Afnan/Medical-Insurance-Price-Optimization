import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from features.feature_engineering import AdvancedFeatureEngineer

def create_feature_pipeline(selected_features=None):
    """
    Creates a comprehensive preprocessing pipeline.
    """
    
    # 1. Advanced Feature Engineering Step
    feat_eng_step = AdvancedFeatureEngineer(verbose=False)
    
    # 2. Categorical & Numerical Transformation Step
    # We define these based on the OUTPUT of the AdvancedFeatureEngineer
    
    def get_pipeline(X):
        # Apply feature engineering first to see the columns
        X_engineered = feat_eng_step.fit_transform(X)
        
        numerical_cols = X_engineered.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_engineered.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if selected_features:
            numerical_cols = [c for c in numerical_cols if c in selected_features]
            categorical_cols = [c for c in categorical_cols if c in selected_features]
            
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, numerical_cols),
                ('cat', cat_transformer, categorical_cols)
            ],
            remainder='drop' if selected_features else 'passthrough'
        )
        
        full_pipeline = Pipeline(steps=[
            ('engineer', feat_eng_step),
            ('preprocessor', preprocessor)
        ])
        
        return full_pipeline

    return get_pipeline
