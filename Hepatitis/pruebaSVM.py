import pandas as pd
from classes.Reader import DataPreprocessor
from classes.SVM import SVM
import os
import numpy as np
import pickle
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import statisticalAnalysis

# STEPS:
# 1. Compare 16 algorithms over the 10 folds of the dataset hepatitis
# 2. I substract the 5 better pairs of kernels and C values according to their accuracies
# 3. I apply the Friedman-Nemenyi test to obtain the best model
# 4. I study the accuracy, performance and f1 of the SELECTED model in
# the 10 initial folds + 10 reduced folds for the 3 methods of reduction (over 40 folds in total)
# 5. I apply the Friedman + Nemenyi test to get the best reduction method.

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

                model = f"SVM, kernel={kernels[i]}, C={c_values[j]:.1f}"
                # Evaluation of the model on the test set

                metrics.append({
                    'Model': model,
                    'Fold': n,
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

def filter_top_models(prev_results_dataframe, kernel_def, c_value_def):
    # Create an empty list to store the filtered rows
    filtered_rows = []

    # Iterate through the top 5 models
    for i in range(5):
        # Create the model string pattern
        model_pattern = f"SVM, kernel={kernel_def[i]}, C={c_value_def[i]:.1f}"

            # Filter rows matching this model
        matching_rows = prev_results_dataframe[prev_results_dataframe['Model'] == model_pattern]

            # Add matching rows to the list
        filtered_rows.append(matching_rows)

        # Concatenate all matching rows into a single DataFrame
    filtered_dataframe = pd.concat(filtered_rows, ignore_index=True)

    return filtered_dataframe

# 1. Compare 16 algorithms over the 10 folds of the dataset hepatitis

dataset_path = '..\\Hepatitis'
prev_results = previous_analysis(dataset_path)
np.savetxt("pre_analysis.txt", prev_results[0], fmt="%s", delimiter=" , ")

# 2. I substract the 5 better pairs of kernels and C values according to their accuracies

#prev_results[1].to_csv('svm_hepatitis_results.csv', index=False)
kernel_def, c_value_def = find_top_five(prev_results[0])
best_five_algo=filter_top_models(prev_results[1],kernel_def,c_value_def)
best_five_algo.to_csv('svm_hepatitis_results.csv', index=False)

# 3. I apply the Friedman-Nemenyi test to obtain the best model

# Selecting the columns and rearranging them with 'Fold' as the first column
df_1 = best_five_algo[['Fold', 'Model', 'Accuracy']]

# Apply tests
test_results_1 = statisticalAnalysis.select_and_apply_test(df_1)

# Plot with conclusions including Nemenyi test if applicable
statisticalAnalysis.plot_conclusions_with_nemenyi(df_1, test_results_1)
