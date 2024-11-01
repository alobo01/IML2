import pandas as pd
import numpy as np
from scipy import stats
from scikit_posthocs import posthoc_nemenyi_friedman
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_model_performance(csv_path,output_path):
    """
    Analyze model performance using Friedman and Nemenyi tests.

    Parameters:
    csv_path (str): Path to the CSV file containing model performance data

    Returns:
    tuple: (top_models_df, friedman_result, nemenyi_matrix)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Calculate average accuracy for each model
    avg_performance = df.groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
    top_models = avg_performance.head(8).index.tolist()

    # Filter data for Top models
    top_df = df[df['Model'].isin(top_models)]

    # Create a pivot table for the Friedman test
    # Rows are datasets/folds, columns are models
    pivot_df = top_df.pivot(
        index='Dataset/Fold',
        columns='Model',
        values='Accuracy'
    )

    # Perform Friedman test
    friedman_statistic, friedman_p_value = stats.friedmanchisquare(
        *[pivot_df[model] for model in top_models]
    )

    # Create summary DataFrame for Top models
    summary_stats = pd.DataFrame({
        'Mean Accuracy': top_df.groupby('Model')['Accuracy'].mean(),
        'Std Accuracy': top_df.groupby('Model')['Accuracy'].std(),
        'Mean F1': top_df.groupby('Model')['F1'].mean(),
        'Mean Time': top_df.groupby('Model')['Time'].mean()
    }).round(4)

    summary_stats = summary_stats.loc[top_models]  # Preserve order
    # Print results
    print("\nTop Models Summary Statistics:")
    print(summary_stats)
    print("\nFriedman Test Results:")
    print(f"Statistic: {friedman_statistic:.4f}")
    print(f"p-value: {friedman_p_value:.4f}")

    if friedman_p_value >= 0.1:
        # Create figure with subplots
        fig, ax1 = plt.subplots(1, 1, figsize=(7,6))

        # Plot mean accuracy with error bars
        models = summary_stats.index
        means = summary_stats['Mean Accuracy']
        stds = summary_stats['Std Accuracy']

        #ax1.bar(models, means)
        ax1.errorbar(models, means, yerr=stds, fmt='o',
                     markerfacecolor = 'black',
                     ecolor = 'blue', capsize=5, capthick=2)
        ax1.set_title('Mean Accuracy of Top Models')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=70, ha='right')
        ax1.set_ylabel('Accuracy')
        # Set y-axis limits from 0.5 to 1
        ax1.set_ylim(0.5, 1)

        # Add a grid to the subplot
        ax1.grid(True)

        plt.tight_layout()
        plt.show()
        fig.savefig(output_path)

        print(f"p-value of {friedman_p_value} was obtained after performing the friedman_statistic test.")

        return False

    else:
        # Perform Nemenyi post-hoc test only when we discard the null hypothesis
        nemenyi_result = posthoc_nemenyi_friedman(pivot_df)
        return summary_stats, (friedman_statistic, friedman_p_value), nemenyi_result


def visualize_results(summary_stats, friedman_result, nemenyi_matrix):
    """
    Create visualizations for the statistical analysis results.
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot mean accuracy with error bars
    models = summary_stats.index
    means = summary_stats['Mean Accuracy']
    stds = summary_stats['Std Accuracy']

    ax1.bar(models, means)
    ax1.errorbar(models, means, yerr=stds, fmt='none', color='black', capsize=5)
    ax1.set_title('Mean Accuracy of Top Models')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')

    # Plot Nemenyi test results as a heatmap
    sns.heatmap(nemenyi_matrix, annot=True, cmap='RdYlGn_r', ax=ax2)
    ax2.set_title('Nemenyi Test p-values\n(lower values indicate significant differences)')

    plt.tight_layout()
    return fig


def main(csv_path, output_path=None):
    """
    Main function to run the analysis and save results.
    """
    # Perform analysis
    if not analyze_model_performance(csv_path):
        print(' ')
    else:
        summary_stats, friedman_result, nemenyi_matrix = analyze_model_performance(csv_path)

        print("\nNemenyi Test Results (p-values):")
        print(nemenyi_matrix.round(4))

        # Create and save visualizations
        fig = visualize_results(summary_stats, friedman_result, nemenyi_matrix)
        if output_path:
            fig.savefig(output_path)
        plt.show()


# Example usage:
#if __name__ == "__main__":
#    # Replace with your CSV file path
#    csv_path = "svm_base_results.csv"
#    output_path = "plots_and_tables\\svm_base\\statistical_analysis_results.png"
#    main(csv_path, output_path)