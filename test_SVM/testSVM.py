from classes.Reader import DataPreprocessor
from classes.SVM import SVM
import os
import numpy as np
from sklearn import svm

# Function to load data from ARFF files for a fold
def load_fold_data(fold_number, dataset_path):
    train_file = os.path.join(dataset_path, f'hepatitis.fold.{fold_number:06d}.train.arff')
    test_file = os.path.join(dataset_path, f'hepatitis.fold.{fold_number:06d}.test.arff')

    loaded_preprocessor = DataPreprocessor().load("hepatitis_preprocessor.joblib")
    train_data_preprocessed = loaded_preprocessor.transform(DataPreprocessor.load_arff(train_file)[0])
    test_data_preprocessed = loaded_preprocessor.transform(DataPreprocessor.load_arff(test_file)[0])

    # Separate features and labels for train and test data
    train_features = train_data_preprocessed.drop('Class', axis=1)
    train_labels = train_data_preprocessed['Class']
    test_features = test_data_preprocessed.drop('Class', axis=1)
    test_labels = test_data_preprocessed['Class']

    return train_features, train_labels, test_features, test_labels


# import the data using the data preprocessor from the reader

fold_num=np.arange(0,10,1)
x_train, y_train, x_test, y_test = load_fold_data(0,'..\\datasets\\hepatitis')
print(x_train[0:2])

clf = svm.SVC()
clf.fit(x_train, y_train)


