import pandas as pd
import numpy as np
from scipy import stats
from scikit_posthocs import posthoc_nemenyi_friedman
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List


class KNNHyperparameterAnalyzer:
    def __init__(self, csv_path: str, alpha: float):
        """Initialize the analyzer with the CSV data."""
        self.df = pd.read_csv(csv_path)
        self.parse_model_column()
        self.alpha = alpha

    def parse_model_column(self):
        """Parse the Model column into separate hyperparameter columns."""
        # Split the Model column into its components
        hyperparams = self.df['Model'].str.split(',', expand=True)

        # Clean and rename columns
        self.df['k'] = hyperparams[1].str.strip().astype(int)
        self.df['distance_metric'] = hyperparams[2].str.strip()
        self.df['weighting_method'] = hyperparams[3].str.strip()
        self.df['voting_policy'] = hyperparams[4].str.strip()

    def create_pivot_table(self, group_col: str) -> pd.DataFrame:
        """Create a pivot table for statistical analysis."""
        return self.df.pivot_table(
            values='Accuracy',
            index='Dataset/Fold',
            columns=group_col,
            aggfunc='first'
        )

    def perform_friedman_test(self, pivot_df: pd.DataFrame) -> Tuple[float, float]:
        """Perform Friedman test on the pivot table."""
        return stats.friedmanchisquare(*[pivot_df[col] for col in pivot_df.columns])

    def perform_nemenyi_test(self, pivot_df: pd.DataFrame) -> pd.DataFrame:
        """Perform Nemenyi post-hoc test."""
        return posthoc_nemenyi_friedman(pivot_df)

    def perform_bonferroni_test(self, pivot_df: pd.DataFrame, control: str) -> pd.DataFrame:
        """
        Perform Bonferroni-corrected Wilcoxon signed-rank tests against a control.
        """
        control_data = pivot_df[control]
        results = {}
        n_comparisons = len(pivot_df.columns) - 1  # Subtract control

        for column in pivot_df.columns:
            if column != control:
                statistic, p_value = stats.wilcoxon(
                    control_data,
                    pivot_df[column],
                    alternative='two-sided'
                )
                # Apply Bonferroni correction
                adjusted_p = min(p_value * n_comparisons, 1.0)
                results[column] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'adjusted_p': adjusted_p
                }

        return pd.DataFrame(results).T

    def analyze_hyperparameter(self,
                               param_name: str,
                               test_type: str = 'nemenyi',
                               control: str = None) -> Dict:
        """
        Analyze a specific hyperparameter using either Nemenyi or Bonferroni test.
        """
        pivot_df = self.create_pivot_table(param_name)
        friedman_stat, friedman_p = self.perform_friedman_test(pivot_df)

        if test_type == 'nemenyi':
            post_hoc = self.perform_nemenyi_test(pivot_df)
        else:  # bonferroni
            post_hoc = self.perform_bonferroni_test(pivot_df, control)

        # Calculate summary statistics
        summary = self.df.groupby(param_name)['Accuracy'].agg(['mean', 'std']).round(4)

        return {
            'summary': summary,
            'friedman_result': (friedman_stat, friedman_p),
            'post_hoc': post_hoc
        }

    def visualize_results(self,
                          param_name: str,
                          results: Dict,
                          test_type: str = 'nemenyi',
                          control: str = None) -> plt.Figure:
        """Create visualizations for the analysis results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot mean accuracy with error bars
        summary = results['summary']
        ax1.bar(range(len(summary)), summary['mean'])
        ax1.errorbar(range(len(summary)), summary['mean'],
                     yerr=summary['std'], fmt='none', color='black', capsize=5)
        ax1.set_xticks(range(len(summary)))
        ax1.set_xticklabels(summary.index, rotation=45, ha='right')
        ax1.set_title(f'Mean Accuracy by {param_name}')
        ax1.set_ylabel('Accuracy')

        # Plot post-hoc test results
        if test_type == 'nemenyi':
            sns.heatmap(results['post_hoc'], annot=True, cmap='RdYlGn_r', ax=ax2)
            ax2.set_title('Nemenyi Test p-values\n(lower values indicate significant differences)')
        else:  # bonferroni
            post_hoc = results['post_hoc']
            ax2.bar(range(len(post_hoc)), -np.log10(post_hoc['adjusted_p']))
            ax2.axhline(y=-np.log10(self.alpha), color='r', linestyle='--',
                        label=f'p={self.alpha} threshold')
            ax2.set_xticks(range(len(post_hoc)))
            ax2.set_xticklabels(post_hoc.index, rotation=45, ha='right')
            ax2.set_title('Bonferroni Test Results\n(-log10 adjusted p-value)')
            ax2.set_ylabel('-log10(adjusted p-value)')
            ax2.legend()

        plt.tight_layout()
        return fig


def main(csv_path: str, output_dir: str = None):
    """Main function to run all analyses."""
    analyzer = KNNHyperparameterAnalyzer(csv_path, alpha = 0.1)

    # Define analyses to perform
    analyses = [
        ('k', 'nemenyi', None),
        ('distance_metric', 'nemenyi', None),
        ('weighting_method', 'bonferroni', 'equal_weight')
    ]

    results = {}
    for param_name, test_type, control in analyses:
        print(f"\nAnalyzing {param_name}...")
        results[param_name] = analyzer.analyze_hyperparameter(param_name, test_type, control)

        # Print results
        print(f"\n{param_name} Summary Statistics:")
        print(results[param_name]['summary'])
        print(f"\nFriedman Test Results:")
        stat, p = results[param_name]['friedman_result']
        print(f"Statistic: {stat:.4f}")
        print(f"p-value: {p:.4f}")

        print(f"\nPost-hoc Test Results ({test_type}):")
        print(results[param_name]['post_hoc'].round(4))

        # Create and save visualization
        fig = analyzer.visualize_results(param_name, results[param_name], test_type, control)
        if output_dir:
            fig.savefig(f"{output_dir}/{param_name}_analysis.png")
        plt.close()

    return results


if __name__ == "__main__":
    # Replace with your CSV file path
    csv_path = "knn_base_results.csv"
    output_dir = "plots_and_tables\\knn_base\\hyperparameter_analysis"
    results = main(csv_path, output_dir)