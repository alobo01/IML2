import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, precision_score, f1_score,
    confusion_matrix
)
import pickle
import time
import os
import multiprocessing
from classes.KNN import KNNAlgorithm, apply_weighting_method
from classes.Reader import DataPreprocessor


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

def get_weighted_features(weighted_train_features, method, k):
    if method in ['equal_weight', 'information_gain_weight']:
        return weighted_train_features[(method, None)]  # Ignore k, always return the same value
    else:
        return weighted_train_features[(method, k)]


# Function to evaluate metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    model_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return model_metrics


# Function to apply reduction and compute metrics for each fold
def process_model(fold_number, model_data, params):
    weighted_train_features, train_labels, test_features, test_labels = model_data
    k, dist_metric, vote_policy, weighting_method = params
    print(f"Processing fold {fold_number} with configuration:\n"
        f" - k: {k}\n"
        f" - Distance Metric: {dist_metric}\n"
        f" - Voting Policy: {vote_policy}\n"
        f" - Weighting Method: {weighting_method}\n")

    # Create the KNNAlgorithm object with the corresponding configuration
    knn = KNNAlgorithm(k=k, distance_metric=dist_metric, voting_policy=vote_policy, weighting_method=weighting_method)

    # Fit the model and evaluate it
    knn.fit(weighted_train_features, train_labels)
    model_metrics = evaluate_model(knn, test_features, test_labels)

    model_metrics['k'] = k
    model_metrics['distance_metric'] = dist_metric
    model_metrics['voting_policy'] = vote_policy
    model_metrics['weighting_method'] = weighting_method

    return fold_number, model_metrics


# Main function to process all folds
def main(dataset_path):
    # Distance metrics, voting policies and weighting methods to test
    distance_metrics = ['euclidean_distance', 'manhattan_distance', 'clark_distance']
    voting_policies = ['majority_class', 'inverse_distance_weighted', 'shepard']
    weighting_methods = ['equal_weight', 'information_gain_weight', 'reliefF_weight']

    # Prepare parameter sets
    params_list = [(k, dist_metric, vote_policy, weighting_method)
                   for k in [1, 3, 5, 7]
                   for dist_metric in distance_metrics
                   for vote_policy in voting_policies
                   for weighting_method in weighting_methods]

    n_folds = 10

    # Initialize result storage
    metrics = []

    results = {
        'k': [],
        'distance_metric' : [],
        'voting_policy' : [],
        'weighting_method' : [],
        'accuracy': [],
        'roc_auc': [],
        'recall': [],
        'precision': [],
        'f1_score': [],
        'confusion_matrix': [],
    }

    for fold_number in range(n_folds):

        # Load the fold data
        train_features, train_labels, test_features, test_labels = load_fold_data(fold_number, dataset_path)

        weighted_train_features = {
            ('equal_weight', None): apply_weighting_method(train_features, train_labels, 'equal_weight'),
            ('information_gain_weight', None): apply_weighting_method(train_features, train_labels,
                                                                      'information_gain_weight')
        }

        # The reliefF weighting method depends on the value of k, while the other 2 do not
        for k in [1, 3, 5, 7]:
            weighted_train_features[('reliefF_weight', k)] = apply_weighting_method(train_features, train_labels,
                                                                                    'reliefF_weight', k)

        def collect_metrics(result):
            fold_number, model_metrics = result
            metrics.append(model_metrics)
            k, dist_metric, vote_policy, weighting_method = params
            print(f"Completed fold {fold_number} for configuration:"
                  f" - k: {k}\n"
                  f" - Distance Metric: {dist_metric}\n"
                  f" - Voting Policy: {vote_policy}\n"
                  f" - Weighting Method: {weighting_method}\n")

        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=8) as pool:
            for params in params_list:
                k, dist_metric, vote_policy, weighting_method = params
                model_data = (get_weighted_features(weighted_train_features, weighting_method, k),
                    train_labels, test_features, test_labels)
                pool.apply_async(process_model, args=(fold_number, model_data, params), callback=collect_metrics)
            pool.close()
            pool.join()

    for params in params_list:
        k, dist_metric, vote_policy, weighting_method = params

        # Gather the metrics of each model configuration across folds
        model_metrics = [m for m in metrics if m['k'] == k
                                            and m['distance_metric'] == dist_metric
                                            and m['voting_policy'] == vote_policy
                                            and m['weighting_method'] == weighting_method]

        # Calculate mean metrics over the folds
        mean_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in model_metrics]),
            'roc_auc': np.mean([m['roc_auc'] for m in model_metrics]),
            'recall': np.mean([m['recall'] for m in model_metrics]),
            'precision': np.mean([m['precision'] for m in model_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in model_metrics]),
            'confusion_matrix': np.mean([m['confusion_matrix'] for m in model_metrics], axis=0)
        }

        # Save metrics for this method
        results['k'].append(k)
        results['distance_metric'].append(dist_metric)
        results['voting_policy'].append(vote_policy)
        results['weighting_method'].append(weighting_method)
        for metric, value in mean_metrics.items():
            results[metric].append(value)

    filename = 'knn_comparison_results'
    # Save results to a pickle file
    with open(f"{filename}.pkl", 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to {filename}.pkl")


if __name__ == "__main__":
    dataset_path = '..\\datasets\\hepatitis'  # Example path
    main(dataset_path)
