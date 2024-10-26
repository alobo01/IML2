from classes.Reader import DataPreprocessor
from classes.SVM import SVM

# # 1. Load data from .joblib
loaded_preprocessor = DataPreprocessor().load("hepatitis_preprocessor.joblib")
train_data_preprocessed = loaded_preprocessor.transform()

# Separate features and labels for the training data
train_features = train_data_preprocessed.drop('Class', axis=1)
train_labels = train_data_preprocessed['Class']

# 2. Preprocess test data
test_data_preprocessed = loaded_preprocessor.transform(DataPreprocessor.load_arff("hepatitis.fold.000000.test.arff")[0])

# Separate features and labels for the test data
test_features = test_data_preprocessed.drop('Class', axis=1)
test_labels = test_data_preprocessed['Class']