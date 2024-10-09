import pandas as pd
from sklearn.metrics import accuracy_score
from Reader import DataPreprocessor
from KNN import KNNAlgorithm
from SVM import SVM

# # 1. Load and preprocess the training data
# train_preprocessor = DataPreprocessor('hepatitis.fold.000000.train.arff')
# train_preprocessor.fit(config_path='config.json')
# train_preprocessor.save("hepatitis_preprocessor.joblib")
loaded_preprocessor = DataPreprocessor().load("hepatitis_preprocessor.joblib")
train_data_preprocessed = loaded_preprocessor.transform()

# Separate features and labels
train_features = train_data_preprocessed.drop('Class', axis=1)
train_labels = train_data_preprocessed['Class']

# 2. Initialize and train the KNN model
knn = KNNAlgorithm(k=3) # You can adjust parameters as needed
knn.fit(train_features, train_labels)

test_data_preprocessed = loaded_preprocessor.transform(DataPreprocessor.load_arff("hepatitis.fold.000000.test.arff")[0])

# Separate features and labels
test_features = test_data_preprocessed.drop('Class', axis=1)
test_labels = test_data_preprocessed['Class']

# 4. Make predictions on the test data
predictions = knn.predict(test_features)

# 5. Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")