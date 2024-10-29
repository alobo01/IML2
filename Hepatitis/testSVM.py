import pandas as pd

from classes.Reader import DataPreprocessor
from classes.SVM import SVM
import os
import numpy as np
import pickle

# Function to load data from ARFF files for a fold
def load_fold_data(fold_number, dataset_path):
    train_file = os.path.join(dataset_path, f'hepatitis.fold.{fold_number:06d}.train.arff')
    test_file = os.path.join(dataset_path, f'hepatitis.fold.{fold_number:06d}.test.arff')

    loaded_preprocessor = DataPreprocessor().load("hepatitis_preprocessor.joblib")
    train_data_preprocessed = loaded_preprocessor.transform(DataPreprocessor.load_arff(train_file))
    test_data_preprocessed = loaded_preprocessor.transform(DataPreprocessor.load_arff(test_file))

    # Separate features and labels for train and test data
    train_features = train_data_preprocessed.drop('Class', axis=1)
    train_labels = train_data_preprocessed['Class']
    test_features = test_data_preprocessed.drop('Class', axis=1)
    test_labels = test_data_preprocessed['Class']

    return train_features, train_labels, test_features, test_labels

C_values=[0.1,1,10,100]
kernels=['linear','rbf','poly','sigmoid']

results=np.zeros((5,5),dtype=object)
results[0,1:]=C_values
results[1:,0]=kernels

# previous analysis to set the kernels and the C values
def previous_analysis():
    for j in range(4):
        for i in range(4):
            prev_accuracy = []
            for n in range(10):
                x_train, y_train, x_test, y_test = load_fold_data(n, '..\\datasets\\hepatitis')
                # Create an instance of the SVM class with the training data
                svm_classifier = SVM(train_data=x_train, train_labels=y_train, kernel=kernels[i], C=C_values[j], gamma='auto')
                # Train the SVM model
                svm_classifier.train()
                prev_accuracy.append(svm_classifier.evaluate(x_test,y_test)[0])
            results[i+1,j+1]=np.mean(np.array(prev_accuracy))
    return results

#prev_results=previous_analysis()
#np.savetxt("pre_analysis.txt", prev_results, fmt="%s", delimiter="      ")

# total analysis
metric_tags = ['accuracy', 'performance', 'roc_auc', 'recall', 'precision', 'f1_score', 'confusion_matrix']

def total_analysis(def_kernel, C_value):
    metrics = []
    for n in range(10):
        x_train, y_train, x_test, y_test = load_fold_data(n,'..\\datasets\\hepatitis')

        # Create an instance of the SVM class with the training data
        svm_classifier = SVM(train_data=x_train, train_labels=y_train, kernel=def_kernel, C=C_value, gamma='auto')

         # Train the SVM model
        svm_classifier.train()

        # Evaluation of the model on the test set
        metrics.append(svm_classifier.evaluate(x_test,y_test))
    return metrics

def save_results(metrics,filename,def_kernel,C_value):
    # Save results to a pickle file
    with open(f"{filename}.pkl", 'wb') as f:
        pickle.dump(metrics, f)

    with open(f'{filename}.txt', 'w') as file:
        metrics.insert(0, metric_tags)
        for row in metrics[1:]:
            # Join the elements of each row
            file.write('        '.join(map(str, row)) + '\n')
        file.write(f"model trained with {def_kernel} kernel and C={C_value}.")

    print(f"Results saved to {filename}.pkl and {filename}.txt")

filename = ['svm_results_linear','svm_results_rbf']
def_kernel = ['linear','rbf']
C_value=[0.1,10]

for i in range(2):
    save_results(total_analysis(def_kernel[i],C_value[i]),filename[i],def_kernel[i],C_value[i])
