import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV and prepare aggregated statistics for base KNN analysis
    """
    # Load results
    results = pd.DataFrame(pd.read_csv(csv_path))

    # Extract configuration parameters from Model column
    results[['Algorithm', 'k', 'distance_metric', 'weighting_method', 'voting_policy']] = \
        results['Model'].str.split(', ', expand=True)

    # Create aggregated results
    aggregated_results = results.groupby(
        ['k', 'distance_metric', 'weighting_method', 'voting_policy']
    ).agg({
        'Accuracy': ['mean', 'std'],
        'Time': 'mean',
        'F1': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    aggregated_results.columns = [
        'k', 'distance_metric', 'weighting_method', 'voting_policy',
        'mean_accuracy', 'std_accuracy', 'mean_time',
        'mean_f1', 'std_f1'
    ]

    return results, aggregated_results


def create_plots_folder(base_path: str):
    """Create folder for plots if it doesn't exist"""
    Path(base_path).mkdir(parents=True, exist_ok=True)


def plot_k_vs_accuracy(results: pd.DataFrame, plots_path: str):
    """Plot K values vs accuracy"""
    plt.figure(figsize=(12, 6))
    mean_scores = results.groupby('k')['mean_accuracy'].mean()
    std_scores = results.groupby('k')['std_accuracy'].mean()

    plt.errorbar(mean_scores.index, mean_scores.values, yerr=std_scores.values,
                 marker='o', color='blue', capsize=5)

    plt.title('Performance by K Value\nHepatitis Dataset')
    plt.xlabel('K Value')
    plt.ylabel('Mean Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'k_vs_accuracy.png'), bbox_inches='tight', dpi=300)
    plt.close()


def plot_heatmap(results: pd.DataFrame, plots_path: str):
    """Plot heatmap of voting policy vs weighting method"""
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
    plt.savefig(os.path.join(plots_path, 'voting_weighting_heatmap.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


def analyze_top_configurations(results: pd.DataFrame, top_n: int = 5):
    """Analyze and print top configurations"""
    top_configs = results.nlargest(top_n, 'mean_accuracy')

    print(f"\nTop {top_n} KNN Configurations:")

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
    """Perform and print statistical analysis of parameters"""
    print("\nStatistical Analysis of Parameters:")
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


def create_performance_plots(results: pd.DataFrame, aggregated_results: pd.DataFrame, plots_path: str):
    """Create additional performance analysis plots"""
    # Time vs Accuracy scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(aggregated_results['mean_time'], aggregated_results['mean_accuracy'], alpha=0.6)
    plt.xlabel('Mean Training+Testing Time (seconds)')
    plt.ylabel('Mean Accuracy')
    plt.title('Time-Accuracy Trade-off')
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'time_vs_accuracy.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # F1 vs Accuracy scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(aggregated_results['mean_f1'], aggregated_results['mean_accuracy'], alpha=0.6)
    plt.xlabel('Mean F1 Score')
    plt.ylabel('Mean Accuracy')
    plt.title('F1 Score vs Accuracy Correlation')
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'f1_vs_accuracy.png'), bbox_inches='tight', dpi=300)
    plt.close()


def main():
    # Paths
    csv_path = 'knn_base_results.csv'
    plots_path = '..\\Hepatitis\\plots_and_tables\\knn_base_analysis'

    # Create plots folder
    create_plots_folder(plots_path)

    # Load and prepare data
    results, aggregated_results = load_and_prepare_data(csv_path)

    # Generate plots
    plot_k_vs_accuracy(aggregated_results, plots_path)
    plot_heatmap(aggregated_results, plots_path)
    create_performance_plots(results, aggregated_results, plots_path)

    # Print analyses
    analyze_top_configurations(aggregated_results)
    statistical_analysis(aggregated_results)


if __name__ == "__main__":
    main()