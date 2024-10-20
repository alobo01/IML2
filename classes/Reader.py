import pandas as pd
import json
from scipy.io import arff
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os


class DataPreprocessor:
    @staticmethod
    def load_arff(arff_filepath):
        try:
            data, meta = arff.loadarff(arff_filepath)
            return pd.DataFrame(data), meta
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {arff_filepath} was not found.")
        except Exception as e:
            raise Exception(f"Error loading ARFF file: {str(e)}")

    def __init__(self, arff_filepath=None):
        self.preprocessor = None
        self.feature_names = None
        self.data = None
        self.meta = None

        if arff_filepath:
            try:
                self.data, self.meta = self.load_arff(arff_filepath)
                self.data = pd.DataFrame(self.data)
            except FileNotFoundError:
                raise FileNotFoundError(f"The file {arff_filepath} was not found.")
            except Exception as e:
                raise Exception(f"Error loading ARFF file: {str(e)}")

    def load_config(self, config_path):
        """Load configuration from a JSON file."""
        with open(config_path, 'r') as config_file:
            return json.load(config_file)

    def fit(self, data: pd.DataFrame = None, config_path: str = None):
        """Fit the preprocessor on the given data."""
        if data is None:
            if self.data is None:
                raise ValueError("No data provided. Either pass data to fit() or load an ARFF file in the constructor.")
            data = self.data

        if config_path is None:
            raise ValueError("No configuration file provided. Please provide a path to a JSON configuration file.")

        config = self.load_config(config_path)
        defaults = config.get('defaults', {})
        columns_config = config.get('columns', {})

        transformers = []

        for column in data.columns:
            column_config = columns_config.get(column, {})
            column_type = column_config.get('type') or (
                'numeric' if pd.api.types.is_numeric_dtype(data[column]) else 'categorical'
            )

            # Merge default config with column-specific config
            merged_config = {**defaults.get(column_type, {}), **column_config}

            steps = []

            # Imputation
            imputer = merged_config.get('imputer', 'simple')
            if imputer == 'simple':
                strategy = merged_config.get('imputer_strategy', 'median' if column_type == 'numeric' else 'constant')
                fill_value = merged_config.get('imputer_fill_value',
                                               'missing' if column_type == 'categorical' else None)
                steps.append(('imputer', SimpleImputer(strategy=strategy, fill_value=fill_value)))
            elif imputer == 'knn':
                steps.append(('imputer', KNNImputer(n_neighbors=merged_config.get('knn_neighbors', 5))))

            # Scaling (for numeric columns)
            if column_type == 'numeric':
                scaler = merged_config.get('scaler', 'standard')
                if scaler == 'standard':
                    steps.append(('scaler', StandardScaler()))
                elif scaler == 'minmax':
                    steps.append(('scaler', MinMaxScaler()))
                elif scaler == 'robust':
                    steps.append(('scaler', RobustScaler()))

            # Encoding (for categorical columns)
            if column_type == 'categorical':
                encoder = merged_config.get('encoder', 'onehot')
                if encoder == 'onehot':
                    steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
                elif encoder == 'ordinal':
                    steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))

            # Feature selection and dimensionality reduction
            if merged_config.get('use_variance_threshold', False):
                steps.append(
                    ('variance_threshold', VarianceThreshold(threshold=merged_config.get('variance_threshold', 0.0))))

            if merged_config.get('use_pca', False):
                steps.append(('pca', PCA(n_components=merged_config.get('pca_components', 0.95))))

            transformers.append((f'col_{column}', Pipeline(steps=steps), [column]))

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

        # Fit the preprocessor on the data
        self.preprocessor.fit(data)

        # Get feature names
        self.feature_names = []
        for name, trans, column in self.preprocessor.transformers_:
            if name != 'remainder':
                if hasattr(trans, 'get_feature_names_out'):
                    self.feature_names.extend(trans.get_feature_names_out([column[0]]))
                else:
                    self.feature_names.append(column[0])

    def transform(self, data: pd.DataFrame = None):
        """Transform the given data using the fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        if data is None:
            if self.data is None:
                raise ValueError(
                    "No data provided. Either pass data to transform() or load an ARFF file in the constructor.")
            data = self.data

        # Transform the data
        transformed_data = self.preprocessor.transform(data)

        # Ensure transformed_data is 2D
        if transformed_data.ndim == 1:
            transformed_data = transformed_data.reshape(-1, 1)

        # If the number of columns doesn't match the feature names, adjust feature names
        if transformed_data.shape[1] != len(self.feature_names):
            print(f"Warning: Number of columns in transformed data ({transformed_data.shape[1]}) "
                  f"doesn't match number of feature names ({len(self.feature_names)}). "
                  f"Adjusting feature names.")
            self.feature_names = [f'feature_{i}' for i in range(transformed_data.shape[1])]

        preprocessed_data = pd.DataFrame(transformed_data, columns=self.feature_names)
        return preprocessed_data

    def fit_transform(self, data: pd.DataFrame = None, config_path: str = None):
        """Fit the preprocessor and transform the data in one step."""
        self.fit(data, config_path)
        return self.transform(data)

    def save(self, filepath: str):
        """Save the preprocessor to a file."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Nothing to save.")

        try:
            joblib.dump((self.data, self.preprocessor, self.feature_names), filepath)
        except PermissionError:
            raise PermissionError(f"Permission denied: Unable to save file to {filepath}")
        except Exception as e:
            raise Exception(f"Error saving preprocessor: {str(e)}")

    @classmethod
    def load(cls, filepath: str):
        """Load a preprocessor from a file."""
        preprocessor = cls()

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} was not found.")

        try:
            preprocessor.data, preprocessor.preprocessor, preprocessor.feature_names = joblib.load(filepath)
        except Exception as e:
            raise Exception(f"Error loading preprocessor: {str(e)}")

        return preprocessor