from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def preprocessing_pipeline(dataframe, target):
    X = dataframe.drop(target, axis=1)
    y = dataframe[target]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_selector = make_column_selector(dtype_include=["object", "category"])
    numerical_selector = make_column_selector(dtype_include=np.number)

    preprocessing = make_column_transformer(
        (OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_selector),
        (StandardScaler(), numerical_selector),
        remainder="drop"
    )
    return x_train, x_test, y_train, y_test, preprocessing