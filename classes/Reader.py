import pandas as pd
from scipy.io.arff import loadarff
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import os


class DataPreprocessor:
    @staticmethod
    def load_arff(file_path):
        """Load ARFF file and convert it to a DataFrame."""
        data, _ = loadarff(file_path)
        df = pd.DataFrame(data)
        for column in df.select_dtypes([object]).columns:
            df[column] = df[column].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        return df

    def __init__(self, arff_filepath=None):
        self.preprocessor = None
        self.data = None
        self.feature_names_ = None
        self.categorical_cols = None
        self.numeric_cols = None
        self.label_encoders = {}
        self.encoder = None

        if isinstance(arff_filepath, pd.DataFrame):
            self.data = arff_filepath
        elif arff_filepath:
            try:
                self.data = self.load_arff(arff_filepath)
            except FileNotFoundError:
                raise FileNotFoundError(f"The file {arff_filepath} was not found.")
            except Exception as e:
                raise Exception(f"Error loading ARFF file: {str(e)}")

    @staticmethod
    def get_whole_dataset_as_df(test_fold_path, train_fold_path=None):
        """Concatenate train and test datasets if both paths are provided."""
        test_data = DataPreprocessor.load_arff(test_fold_path)
        if train_fold_path:
            train_data = DataPreprocessor.load_arff(train_fold_path)
        else:
            raise ValueError("No train data provided.")
        return pd.concat([train_data, test_data], ignore_index=True)

    def fit(self, data: pd.DataFrame = None,
            cat_imputer_strategy='most_frequent', cat_encoding='binary',
            num_imputer_strategy='mean', num_scaler='robust'):
        if data is None:
            if self.data is None:
                raise ValueError("No data provided. Either pass data to fit() or load an ARFF file.")
            data = self.data

        self.data = data = data.replace("?", np.nan)
        self.categorical_cols = list(data.select_dtypes(include=['object']).columns)
        self.numeric_cols = list(data.select_dtypes(include=['number']).columns)

        transformers = []
        self.feature_names_ = []

        if self.categorical_cols:
            if cat_encoding == 'label':
                for col in self.categorical_cols:
                    self.label_encoders[col] = LabelEncoder()
                    non_null_values = data[col].dropna()
                    self.label_encoders[col].fit(non_null_values)

                cat_steps = [
                    ('imputer', SimpleImputer(strategy=cat_imputer_strategy)),
                    ('label_encoder', FunctionTransformer(self._label_encode_transform))
                ]
                self.feature_names_.extend(self.categorical_cols)
            else:
                cat_steps = [
                    ('imputer', SimpleImputer(strategy=cat_imputer_strategy))
                ]
                if cat_encoding in ['onehot', 'binary']:
                    self.encoder = OneHotEncoder(
                        handle_unknown='ignore',
                        sparse_output=False,
                        drop='if_binary' if cat_encoding == 'binary' else None
                    )
                    cat_steps.append(('encoder', self.encoder))

            transformers.append(('cat', Pipeline(steps=cat_steps), self.categorical_cols))

        if self.numeric_cols:
            num_steps = [
                ('imputer', SimpleImputer(strategy=num_imputer_strategy))
            ]
            if num_scaler == 'robust':
                num_steps.append(('scaler', RobustScaler()))
            elif num_scaler == 'minmax':
                num_steps.append(('scaler', MinMaxScaler()))
            transformers.append(('num', Pipeline(steps=num_steps), self.numeric_cols))
            self.feature_names_.extend(self.numeric_cols)

        self.preprocessor = ColumnTransformer(transformers=transformers)
        self.preprocessor.fit(data)

        # Get feature names after fitting
        if cat_encoding != 'label' and self.categorical_cols:
            cat_transformer = self.preprocessor.named_transformers_['cat']
            self.encoder = cat_transformer.named_steps.get('encoder')
            if self.encoder:
                cat_feature_names = self.encoder.get_feature_names_out(self.categorical_cols)
                self.feature_names_ = list(cat_feature_names) + self.numeric_cols

    def _label_encode_transform(self, X):
        """Helper method to apply label encoding to categorical columns."""
        X = pd.DataFrame(X, columns=self.categorical_cols)
        for col in self.categorical_cols:
            encoder = self.label_encoders[col]
            # Handle unknown categories
            X[col] = X[col].map(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
            X[col] = encoder.transform(X[col])
        return X.values

    def transform(self, data: pd.DataFrame = None):
        """Transform the given data using the fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        if data is None:
            data = self.data
        else:
            # Replace '?' with NaN before transformation
            data = data.replace('?', np.nan)

        transformed_data = self.preprocessor.transform(data)
        return pd.DataFrame(transformed_data, columns=self.feature_names_)

    def fit_transform(self, data: pd.DataFrame = None, **kwargs):
        """Fit the preprocessor and transform the data in one step."""
        self.fit(data, **kwargs)
        return self.transform(data)

    def save(self, filepath: str):
        """Save the preprocessor and label encoders to a file."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Nothing to save.")
        save_dict = {
            'preprocessor': self.preprocessor,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names_,
            'categorical_cols': self.categorical_cols,
            'numeric_cols': self.numeric_cols,
            'encoder': self.encoder
        }
        joblib.dump(save_dict, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load a preprocessor and label encoders from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} was not found.")
        preprocessor = cls()
        save_dict = joblib.load(filepath)
        preprocessor.preprocessor = save_dict['preprocessor']
        preprocessor.label_encoders = save_dict['label_encoders']
        preprocessor.feature_names_ = save_dict['feature_names']
        preprocessor.categorical_cols = save_dict['categorical_cols']
        preprocessor.numeric_cols = save_dict['numeric_cols']
        preprocessor.encoder = save_dict['encoder']
        return preprocessor
