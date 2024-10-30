import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV and prepare aggregated statistics
    """
    # Load detailed results
    results = pd.DataFrame(pd.read_csv(csv_path))

    # Extract configuration parameters from Model column and reduction method
    results[['Algorithm', 'k', 'distance_metric', 'weighting_method', 'voting_policy', 'reduction']] = \
        results['Model'].str.split(', ', expand=True)

    # Create aggregated results
    aggregated_results = results.groupby(
        ['k', 'distance_metric', 'weighting_method', 'voting_policy', 'reduction']
    ).agg({
        'Accuracy': ['mean', 'std'],
        'Time': 'mean',
        'F1': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    aggregated_results.columns = [
        'k', 'distance_metric', 'weighting_method', 'voting_policy', 'reduction',
        'mean_accuracy', 'std_accuracy', 'mean_time',
        'mean_f1', 'std_f1'
    ]

    return results, aggregated_results


def create_plots_folder(base_path: str):
    """
    Create folder for plots if it doesn't exist
    """
    Path(base_path).mkdir(parents=True, exist_ok=True)


def plot_k_vs_accuracy_by_reduction(results: pd.DataFrame, plots_path: str):
    """
    Plot K values vs accuracy for different reduction methods
    """
    plt.figure(figsize=(12, 6))
    colors = {'NONE': 'gray', 'EENTH': 'red', 'GCNN': 'green', 'DROP3': 'blue'}

    for reduction in results['reduction'].unique():
        reduction_data = results[results['reduction'] == reduction]
        mean_scores = reduction_data.groupby('k')['mean_accuracy'].mean()
        std_scores = reduction_data.groupby('k')['std_accuracy'].mean()
        plt.errorbar(mean_scores.index, mean_scores.values, yerr=std_scores.values,
                     label=reduction, marker='o', color=colors[reduction])

    plt.title('Performance by K Value and Reduction Method\nHepatitis Dataset')
    plt.xlabel('K Value')
    plt.ylabel('Mean Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'k_vs_accuracy_by_reduction.png'), bbox_inches='tight', dpi=300)
    plt.close()


def plot_reduction_comparison_boxplot(results: pd.DataFrame, plots_path: str):
    """
    Plot box plot comparing accuracies across reduction methods
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='reduction', y='mean_accuracy', data=results)
    plt.title('Accuracy Distribution by Reduction Method\nHepatitis Dataset')
    plt.xlabel('Reduction Method')
    plt.ylabel('Mean Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'reduction_comparison_boxplot.png'), bbox_inches='tight', dpi=300)
    plt.close()


def plot_time_comparison_by_reduction(results: pd.DataFrame, plots_path: str):
    """
    Plot time comparison across reduction methods
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='reduction', y='mean_time', data=results)
    plt.title('Average Execution Time by Reduction Method\nHepatitis Dataset')
    plt.xlabel('Reduction Method')
    plt.ylabel('Mean Execution Time (seconds)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'time_comparison_by_reduction.png'), bbox_inches='tight', dpi=300)
    plt.close()


def plot_heatmap(results: pd.DataFrame, plots_path: str):
    """
    Plot heatmap of voting policy vs weighting method for each reduction method
    """
    for reduction in results['reduction'].unique():
        reduction_data = results[results['reduction'] == reduction]
        plt.figure(figsize=(10, 6))
        pivot_table = reduction_data.pivot_table(
            values='mean_accuracy',
            index='voting_policy',
            columns='weighting_method',
            aggfunc='mean'
        )
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title(f'Mean Accuracy by Voting Policy and Weighting Method\n{reduction} Dataset')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, f'voting_weighting_heatmap_{reduction}.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()


def analyze_top_configurations(results: pd.DataFrame, top_n: int = 5):
    """
    Analyze and print top configurations for each reduction method
    """
    for reduction in results['reduction'].unique():
        reduction_data = results[results['reduction'] == reduction]
        top_configs = reduction_data.nlargest(top_n, 'mean_accuracy')

        print(f"\nTop {top_n} Configurations for {reduction} Dataset:")

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
    """
    Perform and print statistical analysis for each reduction method
    """
    for reduction in results['reduction'].unique():
        reduction_data = results[results['reduction'] == reduction]

        print(f"\nStatistical Analysis for {reduction} Dataset:")
        print("\nBest parameters by category (averaged over other parameters):")

        # Best k
        k_stats = reduction_data.groupby('k')['mean_accuracy'].agg(['mean', 'std']).round(4)
        best_k = k_stats['mean'].idxmax()
        print(f"\nBest k: {best_k} (accuracy: {k_stats.loc[best_k, 'mean']:.4f} ± {k_stats.loc[best_k, 'std']:.4f})")

        # Best distance metric
        metric_stats = reduction_data.groupby('distance_metric')['mean_accuracy'].agg(['mean', 'std']).round(4)
        best_metric = metric_stats['mean'].idxmax()
        print(
            f"Best distance metric: {best_metric} (accuracy: {metric_stats.loc[best_metric, 'mean']:.4f} ± {metric_stats.loc[best_metric, 'std']:.4f})")

        # Best weighting method
        weight_stats = reduction_data.groupby('weighting_method')['mean_accuracy'].agg(['mean', 'std']).round(4)
        best_weight = weight_stats['mean'].idxmax()
        print(
            f"Best weighting method: {best_weight} (accuracy: {weight_stats.loc[best_weight, 'mean']:.4f} ± {weight_stats.loc[best_weight, 'std']:.4f})")

        # Best voting policy
        vote_stats = reduction_data.groupby('voting_policy')['mean_accuracy'].agg(['mean', 'std']).round(4)
        best_vote = vote_stats['mean'].idxmax()
        print(
            f"Best voting policy: {best_vote} (accuracy: {vote_stats.loc[best_vote, 'mean']:.4f} ± {vote_stats.loc[best_vote, 'std']:.4f})")


def create_additional_plots(results: pd.DataFrame, aggregated_results: pd.DataFrame, plots_path: str):
    """
    Create additional insightful plots with reduction method comparisons
    """
    # 1. Time vs Accuracy scatter plot by reduction method
    plt.figure(figsize=(12, 8))
    for reduction in aggregated_results['reduction'].unique():
        reduction_data = aggregated_results[aggregated_results['reduction'] == reduction]
        plt.scatter(reduction_data['mean_time'], reduction_data['mean_accuracy'],
                    alpha=0.6, label=reduction)
    plt.xlabel('Mean Training+Testing Time (seconds)')
    plt.ylabel('Mean Accuracy')
    plt.title('Time-Accuracy Trade-off by Reduction Method')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'time_vs_accuracy_by_reduction.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # 2. F1 vs Accuracy scatter plot by reduction method
    plt.figure(figsize=(12, 8))
    for reduction in aggregated_results['reduction'].unique():
        reduction_data = aggregated_results[aggregated_results['reduction'] == reduction]
        plt.scatter(reduction_data['mean_f1'], reduction_data['mean_accuracy'],
                    alpha=0.6, label=reduction)
    plt.xlabel('Mean F1 Score')
    plt.ylabel('Mean Accuracy')
    plt.title('F1 Score vs Accuracy Correlation by Reduction Method')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'f1_vs_accuracy_by_reduction.png'), bbox_inches='tight', dpi=300)
    plt.close()


def create_custom_pairplot(data: pd.DataFrame, plots_path: str):
    """
    Create a custom pairplot matrix for model hyperparameters showing:
    - Diagonal: Histograms of accuracies per parameter value
    - Lower triangle: Heatmaps of average accuracies
    - Upper triangle: Heatmaps of average times

    Labels are shown only once at the bottom and left of the matrix.
    """
    save_path = os.path.join(plots_path, 'hyperparameter_pairplot_matrix.png')

    # Parameters to analyze
    params = ['k', 'distance_metric', 'weighting_method', 'voting_policy', 'reduction']
    n_params = len(params)

    # Create figure
    fig, axes = plt.subplots(n_params, n_params, figsize=(20, 20))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Color maps
    accuracy_cmap = 'YlOrRd'
    time_cmap = 'YlGnBu'

    data['Time'] = data['Time'].multiply(100)

    # Process each pair of parameters
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            param1 = params[i]
            param2 = params[j]

            # Only show labels on bottom and left edges of the matrix
            if i == n_params - 1:  # Bottom row
                ax.set_xlabel(param2)
                xlabels = True
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])
                xlabels = False

            if j == 0:  # Leftmost column
                ax.set_ylabel(param1)
                ylabels = True
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
                ylabels = False

            if i == j:  # Diagonal - Histograms
                # Group by the parameter and calculate mean accuracy for each value
                param_data = data.groupby(param1)['Accuracy'].mean().reset_index()
                unique_values = param_data[param1].unique()

                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_values)))
                for idx, value in enumerate(unique_values):
                    value_data = param_data[param_data[param1] == value]['Accuracy']
                    ax.hist([value] * len(value_data), weights=value_data,
                            alpha=0.7, color=colors[idx])

                ax.set_title(f'Accuracy Distribution')

            elif i < j:  # Upper triangle - Time heatmaps
                pivot_data = data.pivot_table(
                    values='Time',
                    index=param1,
                    columns=param2,
                    aggfunc='mean'
                )

                sns.heatmap(pivot_data, ax=ax, xticklabels=xlabels, yticklabels=ylabels, cmap=time_cmap,
                            annot=True, fmt='.2f', cbar=False)
                ax.set_title(f'Average Time')

            else:  # Lower triangle - Accuracy heatmaps
                pivot_data = data.pivot_table(
                    values='Accuracy',
                    index=param1,
                    columns=param2,
                    aggfunc='mean'
                )

                sns.heatmap(pivot_data, ax=ax, xticklabels=xlabels, yticklabels=ylabels, cmap=accuracy_cmap,
                            annot=True, fmt='.3f', cbar=False)
                ax.set_title(f'Average Accuracy')

            # Rotate x-axis labels for better readability
            # Only apply rotation to bottom row
            if i == n_params - 1:
                ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)

    plt.suptitle('Hyperparameter Relationships Matrix\nAccuracy and Time Analysis',
                 fontsize=16, y=1.02)

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    # Paths
    csv_path = 'knn_hepatitis_results.csv'
    plots_path = '..\\Hepatitis\\plots_and_tables'

    # Create plots folder
    create_plots_folder(plots_path)

    # Load and prepare data
    results, aggregated_results = load_and_prepare_data(csv_path)

    # Generate all plots
    plot_k_vs_accuracy_by_reduction(aggregated_results, plots_path)
    plot_reduction_comparison_boxplot(aggregated_results, plots_path)
    plot_time_comparison_by_reduction(aggregated_results, plots_path)
    plot_heatmap(aggregated_results, plots_path)
    create_additional_plots(results, aggregated_results, plots_path)

    # Create the custom pairplot
    create_custom_pairplot(results, plots_path)

    # Print analyses to console
    analyze_top_configurations(aggregated_results)
    statistical_analysis(aggregated_results)


if __name__ == "__main__":
    main()