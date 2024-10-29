import pandas as pd
import numpy as np
from classes.KNN import KNNAlgorithm, apply_weighting_method
from classes.Reader import DataPreprocessor
import itertools
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import time
from sklearn.metrics import f1_score


def load_fold_data(fold_number, dataset_path):
    train_file = os.path.join(dataset_path, f'hepatitis.fold.{fold_number:06d}.train.arff')
    test_file = os.path.join(dataset_path, f'hepatitis.fold.{fold_number:06d}.test.arff')
    loaded_preprocessor = DataPreprocessor().load("hepatitis_preprocessor.joblib")
    train_data_preprocessed = loaded_preprocessor.transform(DataPreprocessor.load_arff(train_file))
    test_data_preprocessed = loaded_preprocessor.transform(DataPreprocessor.load_arff(test_file))

    train_features = train_data_preprocessed.drop('Class', axis=1)
    train_labels = train_data_preprocessed['Class']
    test_features = test_data_preprocessed.drop('Class', axis=1)
    test_labels = test_data_preprocessed['Class']

    return train_features, train_labels, test_features, test_labels


def evaluate_knn_configuration(
        weighted_train_features: pd.DataFrame,
        train_labels: pd.Series,
        weighted_test_features: pd.DataFrame,
        test_labels: pd.Series,
        config: Dict
) -> Tuple[float, float, float]:
    """
    Evaluate a single KNN configuration and return accuracy, training/evaluation time, and F1 score
    """
    knn = KNNAlgorithm(
        k=config['k'],
        distance_metric=config['distance_metric'],
        weighting_method='equal_weight',
        voting_policy=config['voting_policy']
    )

    # Start timing
    start_time = time.time()

    # Train and predict
    knn.fit(weighted_train_features, train_labels)
    predictions = knn.predict(weighted_test_features)

    # End timing
    total_time = time.time() - start_time

    # Calculate metrics
    accuracy = knn.score(weighted_test_features, test_labels)
    f1 = f1_score(test_labels, predictions, average='weighted')

    return accuracy, total_time, f1


def get_weighted_features(train_features: pd.DataFrame,
                          train_labels: pd.Series,
                          test_features: pd.DataFrame,
                          weighting_method: str,
                          k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply feature weighting to both train and test features
    """
    weighted_train = apply_weighting_method(train_features, train_labels, weighting_method, k)
    return weighted_train, test_features


def run_experiments(dataset_path: str):
    config_space = {
        'k': [1, 3, 5, 7],
        'distance_metric': ['euclidean_distance', 'manhattan_distance', 'clark_distance'],
        'weighting_method': ['equal_weight', 'information_gain_weight', 'reliefF_weight'],
        'voting_policy': ['majority_class', 'inverse_distance_weighted', 'shepard']
    }

    results = []
    detailed_results = []  # For storing per-fold results

    print(f"Testing configurations across 10 folds...")
    for fold in tqdm(range(10), desc="Folds"):
        train_features, train_labels, test_features, test_labels = load_fold_data(fold, dataset_path)

        for weighting_method in config_space['weighting_method']:
            for k in config_space['k']:
                weighted_train, weighted_test = get_weighted_features(
                    train_features, train_labels, test_features, weighting_method, k
                )

                for distance_metric, voting_policy in itertools.product(
                        config_space['distance_metric'],
                        config_space['voting_policy']
                ):
                    config = {
                        'k': k,
                        'distance_metric': distance_metric,
                        'weighting_method': weighting_method,
                        'voting_policy': voting_policy
                    }

                    # Evaluate configuration
                    accuracy, train_time, f1 = evaluate_knn_configuration(
                        weighted_train, train_labels,
                        weighted_test, test_labels,
                        config
                    )

                    # Create model name string
                    model_name = f"KNN, {k}, {distance_metric}, {weighting_method}, {voting_policy}"

                    # Store detailed (per-fold) result
                    detailed_results.append({
                        'Model': model_name,
                        'Dataset/Fold': f"Hepatitis/{fold}",
                        'Accuracy': accuracy,
                        'Time': train_time,
                        'F1': f1
                    })

                    # Store aggregated result
                    results.append({
                        **config,
                        'fold': fold,
                        'accuracy': accuracy,
                        'time': train_time,
                        'f1': f1
                    })

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)

    # Compute mean and std across folds for each configuration
    final_results = results_df.groupby(
        ['k', 'distance_metric', 'weighting_method', 'voting_policy']
    ).agg({
        'accuracy': ['mean', 'std'],
        'time': 'mean',
        'f1': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    final_results.columns = ['k', 'distance_metric', 'weighting_method', 'voting_policy',
                             'mean_accuracy', 'std_accuracy', 'mean_time',
                             'mean_f1', 'std_f1']

    return final_results, pd.DataFrame(detailed_results)


def plot_results(results: pd.DataFrame):
    # Plot 1: K values vs accuracy for different distance metrics
    plt.figure(figsize=(12, 6))
    for metric in results['distance_metric'].unique():
        metric_data = results[results['distance_metric'] == metric]
        mean_scores = metric_data.groupby('k')['mean_accuracy'].mean()
        std_scores = metric_data.groupby('k')['std_accuracy'].mean()
        plt.errorbar(mean_scores.index, mean_scores.values, yerr=std_scores.values,
                     label=metric, marker='o')

    plt.title('Performance by K Value and Distance Metric\nHepatitis Dataset')
    plt.xlabel('K Value')
    plt.ylabel('Mean Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: Heatmap of voting policy vs weighting method
    plt.figure(figsize=(10, 6))
    pivot_table = results.pivot_table(
        values='mean_accuracy',
        index='voting_policy',
        columns='weighting_method',
        aggfunc='mean'
    )
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Mean Accuracy by Voting Policy and Weighting Method\nHepatitis Dataset')
    plt.tight_layout()
    plt.show()

    # Plot 3: Box plot of accuracies by weighting method
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='weighting_method', y='mean_accuracy', data=results)
    plt.title('Accuracy Distribution by Weighting Method\nHepatitis Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def analyze_top_configurations(results: pd.DataFrame, top_n: int = 5):
    top_configs = results.nlargest(top_n, 'mean_accuracy')

    print("\nTop", top_n, "Configurations for Hepatitis Dataset:")
    for idx, row in top_configs.iterrows():
        print(f"\nRank {idx + 1}")
        print(f"Mean Accuracy: {row['mean_accuracy']:.4f} (±{row['std_accuracy']:.4f})")
        print(f"Mean F1 Score: {row['mean_f1']:.4f} (±{row['std_f1']:.4f})")
        print(f"Mean Training Time: {row['mean_time']:.4f} seconds")
        print(f"Configuration:")
        print(f"  k: {row['k']}")
        print(f"  Distance Metric: {row['distance_metric']}")
        print(f"  Weighting Method: {row['weighting_method']}")
        print(f"  Voting Policy: {row['voting_policy']}")


def statistical_analysis(results: pd.DataFrame):
    print("\nStatistical Analysis:")

    print("\nBest parameters by category (averaged over other parameters):")

    # Best k
    k_stats = results.groupby('k')['mean_accuracy'].agg(['mean', 'std']).round(4)
    best_k = k_stats['mean'].idxmax()
    print(f"\nBest k: {best_k} (accuracy: {k_stats.loc[best_k, 'mean']:.4f} ± {k_stats.loc[best_k, 'std']:.4f})")

    # Best distance metric
    metric_stats = results.groupby('distance_metric')['mean_accuracy'].agg(['mean', 'std']).round(4)
    best_metric = metric_stats['mean'].idxmax()
    print(
        f"Best distance metric: {best_metric} (accuracy: {metric_stats.loc[best_metric, 'mean']:.4f} ± {metric_stats.loc[best_metric, 'std']:.4f})")

    # Best weighting method
    weight_stats = results.groupby('weighting_method')['mean_accuracy'].agg(['mean', 'std']).round(4)
    best_weight = weight_stats['mean'].idxmax()
    print(
        f"Best weighting method: {best_weight} (accuracy: {weight_stats.loc[best_weight, 'mean']:.4f} ± {weight_stats.loc[best_weight, 'std']:.4f})")

    # Best voting policy
    vote_stats = results.groupby('voting_policy')['mean_accuracy'].agg(['mean', 'std']).round(4)
    best_vote = vote_stats['mean'].idxmax()
    print(
        f"Best voting policy: {best_vote} (accuracy: {vote_stats.loc[best_vote, 'mean']:.4f} ± {vote_stats.loc[best_vote, 'std']:.4f})")


if __name__ == "__main__":
    # Set the dataset path
    dataset_path = '..\\datasets\\hepatitis'

    # Run experiments
    aggregated_results, detailed_results = run_experiments(dataset_path)

    # Save detailed results with the requested format
    detailed_results.to_csv('knn_hepatitis_results.csv', index=False)

    # Plot results
    plot_results(aggregated_results)

    # Analyze top configurations
    analyze_top_configurations(aggregated_results)

    # Perform statistical analysis
    statistical_analysis(aggregated_results)