import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature engineering pipeline with domain knowledge.
    Refactored to be Scikit-Learn compatible.
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.feature_names_ = []
        self.original_features_ = []
        
    def fit(self, X, y=None):
        """
        Nothing to fit, but required for Transformer interface.
        """
        self.original_features_ = X.columns.tolist()
        return self
        
    def transform(self, X):
        """
        Create all engineered features.
        """
        if self.verbose:
            print("Starting Feature Engineering Pipeline...")
        
        X_feat = X.copy()
        
        # ====================================================================
        # 1. DOMAIN-SPECIFIC FEATURES
        # ====================================================================
        if self.verbose:
            print("Creating Domain-Specific Features...")
        
        # BMI Categories (WHO Classification)
        X_feat['bmi_category'] = pd.cut(
            X_feat['bmi'],
            bins=[0, 18.5, 25, 30, 35, 40, 100],
            labels=['Underweight', 'Normal', 'Overweight', 
                   'Obese_I', 'Obese_II', 'Obese_III']
        )
        
        # Simplified BMI risk
        X_feat['bmi_risk'] = X_feat['bmi_category'].map({
            'Underweight': 1,
            'Normal': 0,
            'Overweight': 1,
            'Obese_I': 2,
            'Obese_II': 3,
            'Obese_III': 4
        }).astype(float)
        
        # Age Groups (Insurance Industry Standard)
        X_feat['age_group'] = pd.cut(
            X_feat['age'],
            bins=[17, 25, 35, 45, 55, 65],
            labels=['18-25', '26-35', '36-45', '46-55', '56-64']
        )
        
        # Age risk score
        X_feat['age_risk'] = pd.cut(
            X_feat['age'],
            bins=[17, 30, 40, 50, 60, 65],
            labels=[1, 2, 3, 4, 5]
        ).astype(float)
        
        # Family size categories
        X_feat['family_size'] = X_feat['children'].map({
            0: 'single',
            1: 'small_family',
            2: 'small_family',
            3: 'large_family',
            4: 'large_family',
            5: 'large_family'
        })
        
        # ====================================================================
        # 2. INTERACTION FEATURES
        # ====================================================================
        if self.verbose:
            print("Creating Interaction Features...")
        
        # Critical interaction: Smoker × BMI
        X_feat['smoker_bmi'] = (X_feat['smoker'] == 'yes').astype(int) * X_feat['bmi']
        
        # Smoker × Age
        X_feat['smoker_age'] = (X_feat['smoker'] == 'yes').astype(int) * X_feat['age']
        
        # Age × BMI
        X_feat['age_bmi'] = X_feat['age'] * X_feat['bmi']
        
        # Age × Children
        X_feat['age_children'] = X_feat['age'] * X_feat['children']
        
        # BMI × Children
        X_feat['bmi_children'] = X_feat['bmi'] * X_feat['children']
        
        # High-risk indicator (smoker with high BMI)
        X_feat['high_risk'] = ((X_feat['smoker'] == 'yes') & 
                                (X_feat['bmi'] > 30)).astype(int)
        
        # ====================================================================
        # 3. POLYNOMIAL FEATURES
        # ====================================================================
        if self.verbose:
            print("Creating Polynomial Features...")
        
        # Age polynomials
        X_feat['age_squared'] = X_feat['age'] ** 2
        X_feat['age_cubed'] = X_feat['age'] ** 3
        X_feat['age_sqrt'] = np.sqrt(X_feat['age'])
        
        # BMI polynomials
        X_feat['bmi_squared'] = X_feat['bmi'] ** 2
        X_feat['bmi_cubed'] = X_feat['bmi'] ** 3
        X_feat['bmi_sqrt'] = np.sqrt(X_feat['bmi'])
        
        # ====================================================================
        # 4. LOGARITHMIC TRANSFORMATIONS
        # ====================================================================
        if self.verbose:
            print("Creating Logarithmic Features...")
        
        X_feat['log_age'] = np.log1p(X_feat['age'])
        X_feat['log_bmi'] = np.log1p(X_feat['bmi'])
        X_feat['log_age_bmi'] = np.log1p(X_feat['age'] * X_feat['bmi'])
        
        # ====================================================================
        # 5. STATISTICAL FEATURES
        # ====================================================================
        if self.verbose:
            print("Creating Statistical Features...")
        
        # Z-scores (using population mean/std approximation if not fitted)
        # In production, these should ideally use fitted mean/std from Training
        X_feat['age_zscore'] = (X_feat['age'] - X_feat['age'].mean()) / (X_feat['age'].std() + 1e-9)
        X_feat['bmi_zscore'] = (X_feat['bmi'] - X_feat['bmi'].mean()) / (X_feat['bmi'].std() + 1e-9)
        
        # Percentile ranks
        X_feat['age_percentile'] = X_feat['age'].rank(pct=True)
        X_feat['bmi_percentile'] = X_feat['bmi'].rank(pct=True)
        
        # ====================================================================
        # 6. COMPOSITE RISK SCORES
        # ====================================================================
        if self.verbose:
            print("Creating Composite Risk Scores...")
        
        # Overall risk score
        X_feat['risk_score'] = (
            X_feat['age_risk'] * 0.25 +
            X_feat['bmi_risk'] * 0.20 +
            (X_feat['smoker'] == 'yes').astype(int) * 5 +
            X_feat['children'] * 0.1
        )
        
        # Health score (inverse of risk factors)
        X_feat['health_score'] = (
            (100 - X_feat['age']) * 0.3 +
            (50 - np.abs(X_feat['bmi'] - 22)) * 0.3 +  # 22 is optimal BMI
            (X_feat['smoker'] == 'no').astype(int) * 40
        )
        
        # ====================================================================
        # 7. REGIONAL ENCODING
        # ====================================================================
        if self.verbose:
            print("Creating Regional Features...")
        
        region_risk_map = {
            'southeast': 1.2,
            'southwest': 1.0,
            'northeast': 1.1,
            'northwest': 0.9
        }
        X_feat['region_risk_factor'] = X_feat['region'].map(region_risk_map).fillna(1.0)
        
        # ====================================================================
        # 8. BINNING CONTINUOUS FEATURES
        # ====================================================================
        if self.verbose:
            print("Creating Binned Features...")
        
        X_feat['bmi_bin_5'] = pd.cut(X_feat['bmi'], bins=5, labels=False)
        X_feat['bmi_bin_10'] = pd.cut(X_feat['bmi'], bins=10, labels=False)
        X_feat['age_bin_5'] = pd.cut(X_feat['age'], bins=5, labels=False)
        X_feat['age_bin_10'] = pd.cut(X_feat['age'], bins=10, labels=False)
        
        # Convert categoricals to object to ensure they are handled by OneHotEncoder later
        cat_cols = ['bmi_category', 'age_group', 'family_size']
        for col in cat_cols:
            X_feat[col] = X_feat[col].astype(str)

        self.feature_names_ = X_feat.columns.tolist()
        
        if self.verbose:
            print(f"Engineering complete. Total features: {len(self.feature_names_)}")
        
        return X_feat

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_
