import pandas as pd
from scipy.io.arff import loadarff
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, TargetEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import os


class DataPreprocessor:
    """
    A class for preprocessing data with support for ARFF files, handling both numerical and categorical features.
    Supports OneHotEncoder and TargetEncoder for categorical features, with class column always using LabelEncoder.
    Supports different scaling options for numerical features: None, MinMax, or RobustScaler.
    """

    def __init__(self, data=None, class_column='class'):
        self.preprocessor = None
        self.data = None
        self.feature_names_ = []
        self.categorical_cols = []
        self.numeric_cols = []
        self.class_column = class_column
        self.class_encoder = LabelEncoder()
        self.has_class = False
        self.class_data = None

        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, str):
            try:
                self.data = self.load_arff(data)
            except FileNotFoundError:
                raise FileNotFoundError(f"The file {data} was not found.")
            except Exception as e:
                raise Exception(f"Error loading ARFF file: {str(e)}")

    def fit(self, data=None, cat_encoding='binary', num_scaling=None):
        """
        Fit the preprocessor to the data.

        Parameters:
        -----------
        data : pandas.DataFrame, optional
            The input data to fit on. If None, uses the data provided during initialization.
        cat_encoding : str, optional (default='binary')
            The encoding method for categorical variables. Options: 'binary', 'target'
        num_scaling : str, optional (default=None)
            The scaling method for numerical variables. Options: None, 'minmax', 'robust'
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data provided. Either pass data to fit() or initialize with data.")
            data = self.data.copy()

        if cat_encoding not in ['binary', 'target']:
            raise ValueError("cat_encoding must be 'binary' or 'target'")

        if num_scaling not in [None, 'minmax', 'robust']:
            raise ValueError("num_scaling must be None, 'minmax', or 'robust'")

        self.has_class = self.class_column in data.columns
        if self.has_class:
            self.class_data = data[self.class_column].copy()
            self.class_encoder.fit(self.class_data.dropna())
            data = data.drop(columns=[self.class_column])

        self.categorical_cols = list(data.select_dtypes(include=['object']).columns)
        self.numeric_cols = list(data.select_dtypes(include=['number']).columns)

        transformers = []

        if self.categorical_cols:
            if cat_encoding == 'binary':
                cat_steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot_encoder', OneHotEncoder(drop='if_binary'))
                ]
            else:
                if not self.has_class:
                    raise ValueError("Target encoding requires a class column.")
                cat_steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('target_encoder', TargetEncoder())
                ]

            transformers.append(('cat', Pipeline(steps=cat_steps), self.categorical_cols))
            self.feature_names_.extend(self.categorical_cols)

        if self.numeric_cols:
            num_steps = [('imputer', SimpleImputer(strategy='median'))]

            if num_scaling == 'minmax':
                num_steps.append(('scaler', MinMaxScaler()))
            elif num_scaling == 'robust':
                num_steps.append(('scaler', RobustScaler()))

            transformers.append(('num', Pipeline(steps=num_steps), self.numeric_cols))
            self.feature_names_.extend(self.numeric_cols)

        self.preprocessor = ColumnTransformer(transformers=transformers)
        self.preprocessor.fit(data, y=self.class_data if cat_encoding == 'target' else None)

    def transform(self, data=None):
        """
        Transform the input data using the fitted preprocessor.

        Parameters:
        -----------
        data : Union[str, pandas.DataFrame, None], optional
            The input data to transform. Can be:
            - A pandas DataFrame
            - A file path to an ARFF file
            - None (uses the data provided during initialization)

        Returns:
        --------
        pandas.DataFrame
            The transformed data

        Raises:
        -------
        ValueError
            If the preprocessor is not fitted or if the input format is invalid
        FileNotFoundError
            If the provided file path does not exist
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        # Handle different input types
        if data is None:
            if self.data is None:
                raise ValueError("No data provided. Either pass data to transform() or initialize with data.")
            data_to_transform = self.data.copy()
        elif isinstance(data, str):
            try:
                data_to_transform = self.load_arff(data)
            except FileNotFoundError:
                raise FileNotFoundError(f"The file {data} was not found.")
            except Exception as e:
                raise Exception(f"Error loading ARFF file: {str(e)}")
        elif isinstance(data, pd.DataFrame):
            data_to_transform = data.copy()
        else:
            raise ValueError("data must be None, a pandas DataFrame, or a file path string")

        # Extract class data if present
        if self.class_column in data_to_transform.columns:
            class_data = data_to_transform[self.class_column].copy()
            data_to_transform = data_to_transform.drop(columns=[self.class_column])
        else:
            class_data = None

        # Verify all required columns are present
        missing_cols = set(self.feature_names_) - set(data_to_transform.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")

        # Ensure columns are in the correct order
        data_to_transform = data_to_transform[self.feature_names_]

        # Transform the data
        transformed_data = self.preprocessor.transform(data_to_transform)
        result_df = pd.DataFrame(transformed_data, columns=self.feature_names_)

        # Handle class column if present
        if class_data is not None:
            # Map unknown classes to the first known class
            class_data = class_data.map(
                lambda x: x if x in self.class_encoder.classes_ else self.class_encoder.classes_[0])
            result_df[self.class_column] = self.class_encoder.transform(class_data)

        return result_df

    def fit_transform(self, data=None, **kwargs):
        self.fit(data, **kwargs)
        return self.transform(data)

    def save(self, filepath):
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Nothing to save.")

        save_dict = {
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names_,
            'categorical_cols': self.categorical_cols,
            'numeric_cols': self.numeric_cols,
            'class_column': self.class_column,
            'class_encoder': self.class_encoder,
            'has_class': self.has_class
        }
        joblib.dump(save_dict, filepath)

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} was not found.")

        preprocessor = cls()
        save_dict = joblib.load(filepath)

        preprocessor.preprocessor = save_dict['preprocessor']
        preprocessor.feature_names_ = save_dict['feature_names']
        preprocessor.categorical_cols = save_dict['categorical_cols']
        preprocessor.numeric_cols = save_dict['numeric_cols']
        preprocessor.class_column = save_dict['class_column']
        preprocessor.class_encoder = save_dict['class_encoder']
        preprocessor.has_class = save_dict['has_class']

        return preprocessor

    @staticmethod
    def load_arff(file_path):
        data, _ = loadarff(file_path)
        df = pd.DataFrame(data)
        for column in df.select_dtypes([object]).columns:
            df[column] = df[column].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        df = df.replace("?", np.nan)
        return df

    @staticmethod
    def get_whole_dataset_as_df(test_fold_path, train_fold_path):
        test_data = DataPreprocessor.load_arff(test_fold_path)
        train_data = DataPreprocessor.load_arff(train_fold_path)
        return pd.concat([train_data, test_data], ignore_index=True)
