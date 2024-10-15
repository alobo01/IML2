# DataPreprocessor Class Documentation

## Overview

The `DataPreprocessor` class is a versatile tool for loading, preprocessing, and transforming data, particularly suited for ARFF (Attribute-Relation File Format) files. It offers a flexible configuration system for various preprocessing steps including imputation, scaling, encoding, feature selection, and dimensionality reduction.

## Class Methods

### `__init__(self, arff_filepath=None)`

Initializes the DataPreprocessor object.

- **Parameters:**
  - `arff_filepath` (str, optional): Path to an ARFF file to load.

- **Raises:**
  - `FileNotFoundError`: If the specified ARFF file is not found.
  - `Exception`: For other errors during ARFF file loading.

### `load_arff(arff_filepath)`

Static method to load an ARFF file.

- **Parameters:**
  - `arff_filepath` (str): Path to the ARFF file.

- **Returns:**
  - tuple: (pandas.DataFrame, meta)

- **Raises:**
  - `FileNotFoundError`: If the specified ARFF file is not found.
  - `Exception`: For other errors during ARFF file loading.

### `load_config(self, config_path)`

Loads a JSON configuration file.

- **Parameters:**
  - `config_path` (str): Path to the JSON configuration file.

- **Returns:**
  - dict: Configuration as a dictionary.

### `fit(self, data=None, config_path=None)`

Fits the preprocessor on the given data using the specified configuration.

- **Parameters:**
  - `data` (pandas.DataFrame, optional): Data to fit the preprocessor on.
  - `config_path` (str): Path to the JSON configuration file.

- **Raises:**
  - `ValueError`: If no data or configuration file is provided.

### `transform(self, data=None)`

Transforms the given data using the fitted preprocessor.

- **Parameters:**
  - `data` (pandas.DataFrame, optional): Data to transform.

- **Returns:**
  - pandas.DataFrame: Transformed data.

- **Raises:**
  - `ValueError`: If the preprocessor is not fitted or no data is provided.

### `fit_transform(self, data=None, config_path=None)`

Fits the preprocessor and transforms the data in one step.

- **Parameters:**
  - `data` (pandas.DataFrame, optional): Data to fit and transform.
  - `config_path` (str): Path to the JSON configuration file.

- **Returns:**
  - pandas.DataFrame: Transformed data.

### `save(self, filepath)`

Saves the preprocessor to a file.

- **Parameters:**
  - `filepath` (str): Path to save the preprocessor.

- **Raises:**
  - `ValueError`: If the preprocessor is not fitted.
  - `PermissionError`: If unable to save the file due to permissions.
  - `Exception`: For other errors during saving.

### `load(cls, filepath)`

Class method to load a preprocessor from a file.

- **Parameters:**
  - `filepath` (str): Path to the saved preprocessor file.

- **Returns:**
  - DataPreprocessor: Loaded preprocessor object.

- **Raises:**
  - `FileNotFoundError`: If the specified file is not found.
  - `Exception`: For other errors during loading.

## Usage

1. Create a DataPreprocessor object:
   ```python
   preprocessor = DataPreprocessor("path/to/data.arff")
   ```

2. Prepare a JSON configuration file with preprocessing steps.

3. Fit and transform the data:
   ```python
   transformed_data = preprocessor.fit_transform(config_path="path/to/config.json")
   ```

4. Save the preprocessor for later use:
   ```python
   preprocessor.save("path/to/save/preprocessor.joblib")
   ```

5. Load a saved preprocessor:
   ```python
   loaded_preprocessor = DataPreprocessor.load("path/to/saved/preprocessor.joblib")
   ```

## Configuration File Structure

The configuration file should be a JSON file with the following structure:

```json
{
  "defaults": {
    "numeric": {
      "imputer": "simple",
      "imputer_strategy": "median",
      "scaler": "standard"
    },
    "categorical": {
      "imputer": "simple",
      "imputer_strategy": "constant",
      "imputer_fill_value": "missing",
      "encoder": "onehot"
    }
  },
  "columns": {
    "column_name": {
      "type": "numeric",
      "imputer": "knn",
      "knn_neighbors": 5,
      "scaler": "minmax",
      "use_variance_threshold": true,
      "variance_threshold": 0.1,
      "use_pca": true,
      "pca_components": 0.95
    }
  }
}
```

The `defaults` section specifies default preprocessing steps for numeric and categorical columns. The `columns` section allows for column-specific configurations that override the defaults.

## Notes

- The class uses various scikit-learn preprocessing tools including imputers, scalers, encoders, and dimensionality reduction techniques.
- It supports both simple and KNN imputation, standard/minmax/robust scaling, one-hot and ordinal encoding, variance threshold feature selection, and PCA.
- The preprocessor automatically detects column types (numeric or categorical) if not specified in the configuration.
- Feature names are preserved and can be accessed via the `feature_names` attribute after transformation.
