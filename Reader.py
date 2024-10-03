import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

from SVM import SVM


class Reader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.meta = None
        try:
            # Load the ARFF file
            data, meta = arff.loadarff(self.file_path)
            self.data = pd.DataFrame(data)
            self.meta = meta
        except Exception as e:
            print(f"Error reading ARFF file: {e}")

    def get_raw_df(self):
        return self.data

    def get_preprocessed_df(self):
        df = self.data.copy()
        # Separate features and target
        X = df.drop(columns=["class"])
        y = df["class"]

        # Handle missing values
        # For numerical columns: use mean, for categorical: use most frequent
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = X.select_dtypes(include=['object']).columns

        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')

        X[num_cols] = num_imputer.fit_transform(X[num_cols])
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

        # Encode categorical features
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        return X, y, encoders, scaler

    def get_metadata(self):
        if self.meta:
            return self.meta
        else:
            print("ARFF file not loaded yet.")
            return None
