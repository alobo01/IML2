import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os


class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None

    def fit(self, data: pd.DataFrame):
        """Fit the preprocessor on the given data."""
        # Identify numeric and categorical columns
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = data.select_dtypes(include=['object']).columns

        # Create preprocessing steps for numeric features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Create preprocessing steps for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Fit the preprocessor on the data
        self.preprocessor.fit(data)
        self.feature_names = self.preprocessor.get_feature_names_out()

    def transform(self, data: pd.DataFrame):
        """Transform the given data using the fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        # If preprocessing a single row, we need to reshape it
        if data.shape[0] == 1:
            data = data.to_frame().T if isinstance(data, pd.Series) else data

        # Transform the data
        transformed_data = self.preprocessor.transform(data)
        preprocessed_data = pd.DataFrame(transformed_data, columns=self.feature_names)

        # Get tags column
        tags = preprocessed_data.get("class")

        return preprocessed_data.drop("class", axis=1), tags

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessor and transform the data in one step."""
        self.fit(data)
        return self.transform(data)

    def save(self, filepath: str):
        """Save the preprocessor to a file."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Nothing to save.")
        joblib.dump((self.preprocessor, self.feature_names), filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load a preprocessor from a file."""
        preprocessor = cls()
        preprocessor.preprocessor, preprocessor.feature_names = joblib.load(filepath)
        return preprocessor

    @staticmethod
    def load_arff(filepath: str) -> pd.DataFrame:
        """Load an ARFF file and return a pandas DataFrame."""
        data, meta = arff.loadarff(filepath)
        return pd.DataFrame(data)