import pandas as pd
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
    def __init__(self, arff_filepath=None):
        self.preprocessor = None
        self.feature_names = None
        self.data = None
        self.meta = None

        if arff_filepath:
            try:
                self.data, self.meta = arff.loadarff(arff_filepath)
            except FileNotFoundError:
                raise FileNotFoundError(f"The file {arff_filepath} was not found.")
            except Exception as e:
                raise Exception(f"Error loading ARFF file: {str(e)}")

    def fit(self, data: pd.DataFrame = None, config: dict = None):
        """Fit the preprocessor on the given data."""
        if data is None:
            if self.data is None:
                raise ValueError("No data provided. Either pass data to fit() or load an ARFF file in the constructor.")
            data = self.data

        if config is None:
            config = {}

        # Identify numeric and categorical columns
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = data.select_dtypes(include=['object']).columns

        # Create preprocessing steps for numeric features
        numeric_steps = []
        numeric_imputer = config.get('numeric_imputer', 'simple')
        if numeric_imputer == 'simple':
            numeric_steps.append(('imputer', SimpleImputer(strategy=config.get('numeric_imputer_strategy', 'median'))))
        elif numeric_imputer == 'knn':
            numeric_steps.append(('imputer', KNNImputer(n_neighbors=config.get('knn_neighbors', 5))))

        numeric_scaler = config.get('numeric_scaler', 'standard')
        if numeric_scaler == 'standard':
            numeric_steps.append(('scaler', StandardScaler()))
        elif numeric_scaler == 'minmax':
            numeric_steps.append(('scaler', MinMaxScaler()))
        elif numeric_scaler == 'robust':
            numeric_steps.append(('scaler', RobustScaler()))

        if config.get('use_variance_threshold', False):
            numeric_steps.append(
                ('variance_threshold', VarianceThreshold(threshold=config.get('variance_threshold', 0.0))))

        if config.get('use_pca', False):
            numeric_steps.append(('pca', PCA(n_components=config.get('pca_components', 0.95))))

        numeric_transformer = Pipeline(steps=numeric_steps)

        # Create preprocessing steps for categorical features
        categorical_steps = []
        categorical_imputer = config.get('categorical_imputer', 'simple')
        if categorical_imputer == 'simple':
            categorical_steps.append(('imputer', SimpleImputer(strategy='constant',
                                                               fill_value=config.get('categorical_imputer_fill',
                                                                                     'missing'))))

        categorical_encoder = config.get('categorical_encoder', 'onehot')
        if categorical_encoder == 'onehot':
            categorical_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
        elif categorical_encoder == 'ordinal':
            categorical_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))

        categorical_transformer = Pipeline(steps=categorical_steps)

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Fit the preprocessor on the data
        self.preprocessor.fit(data)
        self.feature_names = self.preprocessor.get_feature_names_out()

    def transform(self, data: pd.DataFrame = None):
        """Transform the given data using the fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        if data is None:
            if self.data is None:
                raise ValueError(
                    "No data provided. Either pass data to transform() or load an ARFF file in the constructor.")
            data = self.data

        # If preprocessing a single row, we need to reshape it
        if data.shape[0] == 1:
            data = data.to_frame().T if isinstance(data, pd.Series) else data

        # Transform the data
        transformed_data = self.preprocessor.transform(data)
        preprocessed_data = pd.DataFrame(transformed_data, columns=self.feature_names)

        # Get tags column
        tags = preprocessed_data.get("class")

        return preprocessed_data.drop("class", axis=1), tags

    def fit_transform(self, data: pd.DataFrame = None, config: dict = None):
        """Fit the preprocessor and transform the data in one step."""
        self.fit(data, config)
        return self.transform(data)

    def save(self, filepath: str):
        """Save the preprocessor to a file."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Nothing to save.")

        try:
            joblib.dump((self.preprocessor, self.feature_names), filepath)
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
            preprocessor.preprocessor, preprocessor.feature_names = joblib.load(filepath)
        except Exception as e:
            raise Exception(f"Error loading preprocessor: {str(e)}")

        return preprocessor