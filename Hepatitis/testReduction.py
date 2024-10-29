import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, 
    confusion_matrix
)
import pickle
import time
import os
from concurrent.futures import ThreadPoolExecutor
from classes.ReductionKNN import ReductionKNN
from classes.Reader import DataPreprocessor
from classes.KNN import KNNAlgorithm


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


# Function to evaluate metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics


# Function to apply reduction and compute metrics for each fold
def process_fold(fold_number, dataset_path, method):
    print(f"Processing fold {fold_number} with method {method}")

    # Load the fold data
    train_features, train_labels, test_features, test_labels = load_fold_data(fold_number, dataset_path)

    # Initialize original KNN
    ogKNN = KNNAlgorithm(k=3)
    ogKNN.fit(train_features, train_labels)
    if method == 'None':
        model = ogKNN
        reduction_percentage = 100
        reduction_time = 0
        train_features_reduced = train_features
        train_labels_reduced = train_labels
    else:
        start = time.time()
        reduction_knn = ReductionKNN(ogKNN, ogKNN)
        reduced_data = reduction_knn.apply_reduction(pd.concat([train_features, train_labels], axis=1), method)
        reduction_time = time.time() - start
        reduced_data.to_csv(f"ReducedFolds/hepatitis.fold.{fold_number:06d}.train.{method}.csv")
        train_features_reduced = reduced_data.drop('Class', axis=1)
        train_labels_reduced = reduced_data['Class']
        reduction_percentage = 100 * (len(train_labels_reduced) / len(train_labels))
    
    
    
    # Fit the model and evaluate it
    #ogKNN.fit(train_features_reduced, train_labels_reduced)
    #metrics = evaluate_model(ogKNN, test_features, test_labels)
    metrics = {}
    metrics['reduction_percentage'] = reduction_percentage
    metrics['reduction_time'] = reduction_time

    return fold_number, metrics


# Main function to process all folds
def main(dataset_path):
    # Reduction methods to compare
    reduction_methods = ['DROP3', 'None', 'GCNN', 'EENTH']
    n_folds = 10

    # Initialize result storage
    results = {
    }

    # Iterate over methods
    for method in reduction_methods:
        print(f"Evaluating method: {method}")

        # Parallel execution using ThreadPoolExecutor for each fold
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_fold, fold_number, dataset_path, method)
                for fold_number in range(n_folds)
            ]

            # Collect results from each fold
            fold_metrics = []
            for future in futures:
                fold_number, metrics = future.result()
                results[(fold_number,method)] = metrics
                print(f"Completed fold {fold_number} for method {method}")





    filename = 'knn_reduction_comparison_results'
    # Save results to a pickle file
    with open(f"{filename}.pkl", 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to {filename}.pkl")


if __name__ == "__main__":
    dataset_path = '..\\datasets\\hepatitis'
    main(dataset_path)
