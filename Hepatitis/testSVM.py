import pandas as pd
from classes.Reader import DataPreprocessor
from classes.SVM import SVM
import os
import numpy as np
import pickle
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# STEPS:
# 1. Compare 16 algorithms over the 10 folds of the dataset hepatitis
# 2. I substract the 2 better kernels with their respective C values according to their accuracies
# 3. I apply the Wilconxin test to obtain the best model
# 4. I study the accuracy, performance and f1 of the SELECTED model in
# the 10 initial folds + 10 reduced folds for the 3 methods of reduction (over 40 folds in total)
# 5. I apply the Fridman + Nemenyi test to get the best reduction method.


def load_fold_data(fold_number: int, dataset_path: str, reduction_method: Optional[str] = None) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load fold data with optional reduction method

    Args:
        fold_number: The fold number to load
        dataset_path: Base path to the dataset
        reduction_method: Optional reduction method ('EENTH', 'GCNN' or 'DROP3')

    Returns:
        Tuple of (train_features, train_labels, test_features, test_labels)
    """

    if reduction_method:
        # Path for reduced datasets
        train_path = os.path.join(dataset_path, 'ReducedFolds')
        train_file = os.path.join(train_path, f'hepatitis.fold.{fold_number:06d}.train.{reduction_method}.csv')

    else:
        # Path for normal datasets
        train_path = os.path.join(dataset_path, 'preprocessed_csvs')
        train_file = os.path.join(train_path, f'hepatitis.fold.{fold_number:06d}.train.csv')

    # Load reduced CSV files
    train_data = pd.read_csv(train_file)
    train_data = train_data.drop('Unnamed: 0', axis=1)

    test_file = os.path.join(dataset_path, 'preprocessed_csvs', f'hepatitis.fold.{fold_number:06d}.test.csv')
    test_data = pd.read_csv(test_file)
    test_data = test_data.drop('Unnamed: 0', axis=1)

    # Split features and labels
    train_features = train_data.drop('Class', axis=1)
    train_labels = train_data['Class']
    test_features = test_data.drop('Class', axis=1)
    test_labels = test_data['Class']

    return train_features, train_labels, test_features, test_labels

# Pre-analysis of the data to choose the best two kernels and their corresponding C values - using Hepatitis without reduction

# previous_analysis gives the mean accuracy for each case studied
def previous_analysis(dataset_path_f):
    # values analyzed
    c_values = [0.1, 1, 10, 100]
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    results = np.zeros((5, 5), dtype=object)
    results[0, 1:] = c_values
    results[1:, 0] = kernels
    metrics=[]

    for j in range(4):
        for i in range(4):
            prev_accuracy = np.zeros(10)
            for n in range(10):
                x_train, y_train, x_test, y_test = load_fold_data(n,dataset_path_f)
                # Create an instance of the SVM class with the training data
                svm_classifier = SVM(train_data=x_train, train_labels=y_train, kernel=kernels[i], C=c_values[j], gamma='auto')
                # Train the SVM model
                svm_classifier.train()
                evaluation=svm_classifier.evaluate(x_test,y_test)
                prev_accuracy[n]=evaluation[0]

                model = f"SVM, kernel={kernels[i]}, C={c_values[j]}"
                # Evaluation of the model on the test set

                metrics.append({
                    'Model': model,
                    'Dataset/Fold': f"Hepatitis/{n}",
                    'Reduction': None,
                    'Accuracy': evaluation[0],
                    'Time': evaluation[1],
                    'F1': evaluation[2],
                    'recall': evaluation[3]
                })
                # evaluation[3] is the recall
                print(f'Model {model}, for hepatitis/{n} trained and saved.')
            results[i+1,j+1]=np.mean(prev_accuracy)
            print('accuracy of ',kernels[i],' ',c_values[j],' retrieved')

    print('Previous analysis done in order to find the best parameters for the dataset.')
    return results, pd.DataFrame(metrics)

# given the results from previous analysis we get the two kernels which show higher mean accuracy in the fold
# of the dataset and the C values
def find_top_two(results_f):
    # Initialize arrays to store results
    kernel_tags = np.zeros(2, dtype='object')
    c_value_tags = np.zeros(2, dtype='float')

    # Create a copy of the data region (excluding tags)
    data_region = results_f[1:, 1:].copy()

    # Find first maximum
    max_index = np.argmax(data_region)
    max_coords = np.unravel_index(max_index, data_region.shape)

    # Store tags for first maximum
    kernel_tags[0] = results_f[max_coords[0] + 1, 0]  # Add 1 to account for excluded row
    c_value_tags[0] = results_f[0, max_coords[1] + 1]  # Add 1 to account for excluded column

    # Mask the first maximum
    data_region[max_coords] = np.min(data_region) - 1

    # Find second maximum with different kernel
    while True:
        max_index = np.argmax(data_region)
        max_coords = np.unravel_index(max_index, data_region.shape)
        current_kernel = results_f[max_coords[0] + 1, 0]

        # Check if kernel is different from first maximum
        if current_kernel != kernel_tags[0]:
            kernel_tags[1] = current_kernel
            c_value_tags[1] = results_f[0, max_coords[1] + 1]
            break

        # If same kernel, mask this value and continue
        data_region[max_coords] = np.min(data_region) - 1

        # Safety check to prevent infinite loop
        if np.all(data_region == np.min(data_region) - 1):
            raise ValueError("Could not find second maximum with different kernel")

    print(f'Optimal parameters combination found: ,{kernel_tags} and {c_value_tags}')
    return kernel_tags, c_value_tags

def find_top_five(results_f):
    # Initialize arrays to store results
    kernel_tags = np.zeros(5, dtype='object')
    c_value_tags = np.zeros(5, dtype='float')

    # Create a copy of the data region (excluding tags)
    data_region = results_f[1:, 1:].copy()

    print(f'Optimal parameters are: ')

    for n in range(5):
        # Find first maximum
        max_index = np.argmax(data_region)
        max_coords = np.unravel_index(max_index, data_region.shape)

        # Store tags for first maximum
        kernel_tags[n] = results_f[max_coords[0] + 1, 0]  # Add 1 to account for excluded row
        c_value_tags[n] = results_f[0, max_coords[1] + 1]  # Add 1 to account for excluded column

        # Mask the first maximum
        data_region[max_coords] = np.min(data_region) - 1

        print(f'{kernel_tags[n]} and {c_value_tags[n]}')

    return kernel_tags, c_value_tags

# here I should get between the best two models (kernel and C) the BEST one according to Wilconxin test

# insert function here



# Once the best model is known, I study the accuracy, performance and f1 score of that model

def total_analysis(kernel_def_f, c_value_def_f,dataset_path_ff):
    # Add reduction methods (None means original dataset)
    reduction_methods_f = [None, 'EENTH', 'GCNN', 'DROP3']
    metrics = []

    print(f"Testing configurations across 10 folds with different reduction methods.")
    for reduction_method in reduction_methods_f:
        reduction_desc = reduction_method if reduction_method else "None"

        for fold in range(10):
            for i in range(2):
                model = f"SVM, kernel={kernel_def_f[i]}, C={c_value_def_f[i]}"
                x_train, y_train, x_test, y_test = load_fold_data(fold, dataset_path_ff,reduction_method)
                # Create an instance of the SVM class with the training data
                svm_classifier = SVM(train_data=x_train,train_labels=y_train,kernel=kernel_def_f[i],C=c_value_def_f[i],
                                     gamma='auto')

                # Train the SVM model
                svm_classifier.train()

                # Evaluation of the model on the test set
                evaluation=svm_classifier.evaluate(x_test, y_test)

                metrics.append({
                    'Model': model,
                    'Dataset/Fold': f"Hepatitis/{fold}",
                    'Reduction': reduction_desc,
                    'Accuracy': evaluation[0],
                    'Time': evaluation[1],
                    'F1': evaluation[2]
                })
                # evaluation[3] is the recall
                print(f'Model {model}, for hepatitis/{fold} and {reduction_desc} trained and saved.')

    return pd.DataFrame(metrics)

# The Fridman + Nemenyi test is applied in order to obtain the best reduced method

# insert function here

if __name__ == "__main__":
    # Set the dataset path
    dataset_path = '..\\Hepatitis'

    # previous analysis of the data with the non_reduced hepatitis dataset
    #prev_results = previous_analysis(dataset_path)
    #kernel_def, c_value_def = find_top_two(prev_results[0])
    #np.savetxt("pre_analysis.txt", prev_results[0], fmt="%s", delimiter=" , ")
    #prev_results[1].to_csv('svm_hepatitis_results.csv')

    # previous analysis of the data with the non_reduced hepatitis datasets to find the 5 algorithms with better accuracy
    prev_results = previous_analysis(dataset_path)
    np.savetxt("pre_analysis.txt", prev_results[0], fmt="%s", delimiter=" , ")
    prev_results[1].to_csv('svm_hepatitis_results.csv', index=False)

    kernel_def, c_value_def = find_top_five(prev_results[0])

    # Run experiments
    #results_svm_hepatitis= total_analysis(kernel_def, c_value_def,dataset_path)

    # Save detailed results with the requested format
    #results_svm_hepatitis.to_csv('svm_hepatitis_results.csv', index=False)
    #results_svm_hepatitis_expanded.to_csv('svm_hepatitis_results_expanded.csv', index=False)
