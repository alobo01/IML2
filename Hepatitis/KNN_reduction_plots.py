import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV and prepare aggregated statistics for reduction analysis
    """
    # Load results
    results = pd.DataFrame(pd.read_csv(csv_path))

    # Extract configuration parameters and reduction method
    results[['Algorithm', 'k', 'distance_metric', 'weighting_method', 'voting_policy', 'reduction']] = \
        results['Model'].str.split(', ', expand=True)

    # Create aggregated results
    aggregated_results = results.groupby(['reduction']).agg({
        'Accuracy': ['mean', 'std'],
        'Time': 'mean',
        'F1': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    aggregated_results.columns = [
        'reduction', 'mean_accuracy', 'std_accuracy', 'mean_time',
        'mean_f1', 'std_f1'
    ]

    return results, aggregated_results


def create_plots_folder(base_path: str):
    """Create folder for plots if it doesn't exist"""
    Path(base_path).mkdir(parents=True, exist_ok=True)


def plot_reduction_accuracy_comparison(results: pd.DataFrame, plots_path: str):
    """Plot comparison of reduction methods' accuracies"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='reduction', y='Accuracy', data=results)
    plt.title('Accuracy Distribution by Reduction Method\nHepatitis Dataset')
    plt.xlabel('Reduction Method')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'reduction_accuracy_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()


def plot_time_comparison(aggregated_results: pd.DataFrame, plots_path: str):
    """Plot time comparison across reduction methods"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='reduction', y='mean_time', data=aggregated_results)
    plt.title('Average Execution Time by Reduction Method\nHepatitis Dataset')
    plt.xlabel('Reduction Method')
    plt.ylabel('Mean Execution Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'reduction_time_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()


def calculate_storage_percentages(sample_counts_df: pd.DataFrame):
    """
    Calculate the average percentage of storage reduction for each reduction method compared to the "None" reduction.
    """
    storage_percentages = {'NONE': 100}

    for reduction_method in sample_counts_df['Reduction Method'].unique():
        if reduction_method == "NONE":
            continue

        reduction_samples = sample_counts_df[sample_counts_df['Reduction Method'] == reduction_method]['Training Samples']
        none_samples = sample_counts_df[sample_counts_df['Reduction Method'] == "NONE"]['Training Samples']

        avg_reduction_samples = reduction_samples.mean()
        avg_none_samples = none_samples.mean()

        storage_percentage = avg_reduction_samples / avg_none_samples * 100
        storage_percentages[reduction_method] = storage_percentage

    return storage_percentages

def plot_storage_comparison(sample_counts: pd.DataFrame, plots_path: str):

    storage_percentages = calculate_storage_percentages(sample_counts)

    # Plot the storage percentages
    plt.figure(figsize=(8, 6))
    plt.bar(storage_percentages.keys(), storage_percentages.values())
    plt.xlabel('Reduction Method')
    plt.ylabel('Storage Percentage (%)')
    plt.title('Storage Percentages per Reduction Method')
    plt.grid()
    plt.savefig(os.path.join(plots_path, 'storage_percentage_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()


# def analyze_reduction_methods(aggregated_results: pd.DataFrame):
#     """Analyze and print statistics for each reduction method"""
#     print("\nReduction Methods Analysis:")
#
#     for _, row in aggregated_results.iterrows():
#         print(f"\nReduction Method: {row['reduction']}")
#         print(f"Mean Accuracy: {row['mean_accuracy']:.4f} (±{row['std_accuracy']:.4f})")
#         print(f"Mean F1 Score: {row['mean_f1']:.4f} (±{row['std_f1']:.4f})")
#         print(f"Mean Execution Time: {row['mean_time']:.4f} seconds")


def create_comparison_plots(results: pd.DataFrame, plots_path: str):
    """Create additional comparison plots"""
    # Accuracy vs Time trade-off
    plt.figure(figsize=(12, 8))
    for reduction in results['reduction'].unique():
        reduction_data = results[results['reduction'] == reduction]
        plt.scatter(reduction_data['Time'], reduction_data['Accuracy'],
                    alpha=0.6, label=reduction)
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Time-Accuracy Trade-off by Reduction Method')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'reduction_time_accuracy_tradeoff.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # F1 vs Accuracy correlation
    plt.figure(figsize=(12, 8))
    for reduction in results['reduction'].unique():
        reduction_data = results[results['reduction'] == reduction]
        plt.scatter(reduction_data['F1'], reduction_data['Accuracy'],
                    alpha=0.6, label=reduction)
    plt.xlabel('F1 Score')
    plt.ylabel('Accuracy')
    plt.title('F1 Score vs Accuracy Correlation by Reduction Method')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'reduction_f1_accuracy_correlation.png'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_efficiency_comparison(results: pd.DataFrame, sample_counts: pd.DataFrame, plots_path: str):
    """
    Plot accuracy per training sample for each reduction method.
    This shows how efficiently each method uses its training samples to achieve accuracy.
    """
    # Calculate average accuracy for each reduction method
    avg_accuracy = results.groupby('reduction')['Accuracy'].mean()

    # Calculate average number of training samples for each reduction method
    avg_samples = sample_counts.groupby('Reduction Method')['Training Samples'].mean()

    # Calculate efficiency (accuracy per sample)
    efficiency = {}
    for method in avg_accuracy.index:
        # Match the method name in sample_counts (which uses uppercase)
        samples_method = method.upper()
        if samples_method in avg_samples.index:
            # Multiply by 100 to make the values more readable
            efficiency[method] = (avg_accuracy[method] * 100) / avg_samples[samples_method]

    # Create the plot
    plt.figure(figsize=(10, 6))
    methods = list(efficiency.keys())
    efficiencies = list(efficiency.values())

    bars = plt.bar(methods, efficiencies)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2e}',
                 ha='center', va='bottom')

    plt.title('Accuracy per Training Sample by Reduction Method\nHepatitis Dataset')
    plt.xlabel('Reduction Method')
    plt.ylabel('Efficiency (Accuracy % per Training Sample)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join(plots_path, 'reduction_efficiency_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


# def statistical_comparison(results: pd.DataFrame):
#     """Perform statistical comparison between reduction methods"""
#     print("\nStatistical Comparison of Reduction Methods:")
#
#     # Overall rankings
#     print("\nOverall Rankings (averaged across all configurations):")
#     rankings = results.groupby('reduction').agg({
#         'Accuracy': ['mean', 'std'],
#         'Time': 'mean',
#         'F1': ['mean', 'std']
#     }).round(4)
#
#     print("\nBy Accuracy:")
#     accuracy_ranking = rankings['Accuracy']['mean'].sort_values(ascending=False)
#     for idx, (reduction, acc) in enumerate(accuracy_ranking.items(), 1):
#         std = rankings.loc[reduction, ('Accuracy', 'std')]
#         print(f"{idx}. {reduction}: {acc:.4f} (±{std:.4f})")
#
#     print("\nBy Execution Time:")
#     time_ranking = rankings['Time']['mean'].sort_values()
#     for idx, (reduction, time) in enumerate(time_ranking.items(), 1):
#         print(f"{idx}. {reduction}: {time:.4f} seconds")


def main():
    # Paths
    csv_path = 'knn_reduction_results.csv'
    counts_path = 'knn_reduction_counts.csv'
    plots_path = '..\\Hepatitis\\plots_and_tables\\knn_reduction'

    # Create plots folder
    create_plots_folder(plots_path)

    # Load and prepare data
    results, aggregated_results = load_and_prepare_data(csv_path)
    sample_counts = pd.DataFrame(pd.read_csv(counts_path))

    # Generate plots
    plot_reduction_accuracy_comparison(results, plots_path)
    plot_time_comparison(aggregated_results, plots_path)
    create_comparison_plots(results, plots_path)
    plot_storage_comparison(sample_counts, plots_path)
    plot_efficiency_comparison(results, sample_counts, plots_path)

    # Print analyses
    # analyze_reduction_methods(aggregated_results)
    # statistical_comparison(results)


if __name__ == "__main__":
    main()