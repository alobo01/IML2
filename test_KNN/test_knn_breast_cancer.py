import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from classes.KNN import KNNAlgorithm, apply_weighting_method
import itertools
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def evaluate_knn_configuration(
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        test_features: pd.DataFrame,
        test_labels: pd.Series,
        config: Dict
) -> float:
    """
    Evaluate a single KNN configuration and return its accuracy
    """
    knn = KNNAlgorithm(
        k=config['k'],
        distance_metric=config['distance_metric'],
        weighting_method=config['weighting_method'],
        voting_policy=config['voting_policy']
    )

    knn.fit(train_features, train_labels)
    return knn.score(test_features, test_labels)


def run_experiments():
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Configuration space
    config_space = {
        'k': [1, 3, 5, 7],
        'distance_metric': ['euclidean_distance', 'manhattan_distance', 'clark_distance'],
        'weighting_method': ['equal_weight', 'information_gain_weight', 'reliefF_weight'],
        'voting_policy': ['majority_class', 'inverse_distance_weighted', 'shepard']
    }

    # Generate all possible configurations
    configurations = [dict(zip(config_space.keys(), v))
                      for v in itertools.product(*config_space.values())]

    # Initialize results storage
    results = []

    # Prepare cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()

    # Run experiments
    print(f"Testing {len(configurations)} different configurations...")
    for config in tqdm(configurations):
        fold_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Scale features
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns
            )

            # Evaluate configuration
            score = evaluate_knn_configuration(
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                config
            )
            fold_scores.append(score)

        # Store results
        results.append({
            **config,
            'mean_accuracy': np.mean(fold_scores),
            'std_accuracy': np.std(fold_scores)
        })

    return pd.DataFrame(results)


def plot_results(results: pd.DataFrame):
    # Plot 1: K values vs accuracy for different distance metrics
    plt.figure(figsize=(12, 6))
    for metric in results['distance_metric'].unique():
        metric_data = results[results['distance_metric'] == metric]
        mean_scores = metric_data.groupby('k')['mean_accuracy'].mean()
        std_scores = metric_data.groupby('k')['std_accuracy'].mean()
        plt.errorbar(mean_scores.index, mean_scores.values, yerr=std_scores.values,
                     label=metric, marker='o')

    plt.title('Performance by K Value and Distance Metric')
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
    plt.title('Mean Accuracy by Voting Policy and Weighting Method')
    plt.tight_layout()
    plt.show()


def analyze_top_configurations(results: pd.DataFrame, top_n: int = 5):
    # Sort by mean accuracy and get top configurations
    top_configs = results.nlargest(top_n, 'mean_accuracy')

    print("\nTop", top_n, "Configurations:")
    for idx, row in top_configs.iterrows():
        print(f"\nRank {idx + 1}")
        print(f"Mean Accuracy: {row['mean_accuracy']:.4f} (±{row['std_accuracy']:.4f})")
        print(f"Configuration:")
        print(f"  k: {row['k']}")
        print(f"  Distance Metric: {row['distance_metric']}")
        print(f"  Weighting Method: {row['weighting_method']}")
        print(f"  Voting Policy: {row['voting_policy']}")


if __name__ == "__main__":
    # Run experiments
    results = run_experiments()

    # Plot results
    plot_results(results)

    # Analyze top configurations
    analyze_top_configurations(results)

    # Save results
    results.to_csv('knn_experiment_results.csv', index=False)