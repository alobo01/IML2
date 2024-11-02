import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class ReductionMethodAnalyzer:
    def __init__(self, csv_path: str, alpha: float):
        """Initialize the analyzer with the CSV data."""
        self.df = pd.read_csv(csv_path)
        self.parse_model_column()
        self.alpha = alpha

    def parse_model_column(self):
        """Parse the Model column to extract the reduction method."""
        # Extract the reduction method from the last part of the Model string
        self.df['reduction_method'] = self.df['Model'].str.split(',').str[-1].str.strip()
        # Remove parentheses if present
        self.df['reduction_method'] = self.df['reduction_method'].str.strip('()')

    def create_pivot_table(self) -> pd.DataFrame:
        """Create a pivot table for statistical analysis."""
        return self.df.pivot_table(
            values=['Accuracy', 'F1'],  # Added F1 score analysis
            index='Dataset/Fold',
            columns='reduction_method',
            aggfunc='first'
        )

    def perform_friedman_test(self, data: pd.DataFrame) -> tuple[float, float]:
        """Perform Friedman test on the data."""
        return stats.friedmanchisquare(*[data[col] for col in data.columns])

    def perform_bonferroni_test(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform Bonferroni-corrected Wilcoxon signed-rank tests against NONE reduction."""
        control_data = data['NONE']
        results = {}
        n_comparisons = len(data.columns) - 1

        for method in data.columns:
            if method != 'NONE':
                effect_size = np.median(data[method] - control_data)
                statistic, p_value = stats.wilcoxon(
                    control_data,
                    data[method],
                    alternative='two-sided'
                )
                diff_percentage = ((data[method].mean() - control_data.mean())
                                   / control_data.mean() * 100)
                adjusted_p = min(p_value * n_comparisons, 1.0)

                results[method] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'adjusted_p': adjusted_p,
                    'effect_size': effect_size,
                    'diff_percentage': diff_percentage
                }

        return pd.DataFrame(results).T

    def analyze_reduction_methods(self) -> dict:
        """Analyze the impact of different reduction methods."""
        pivot_df = self.create_pivot_table()

        # Separate accuracy and F1 scores
        accuracy_pivot = pivot_df['Accuracy']
        f1_pivot = pivot_df['F1']

        # Perform Friedman test for both metrics
        accuracy_friedman = self.perform_friedman_test(accuracy_pivot)
        f1_friedman = self.perform_friedman_test(f1_pivot)

        # Calculate summary statistics for both metrics
        summary_accuracy = self.df.groupby('reduction_method')['Accuracy'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(4)

        summary_f1 = self.df.groupby('reduction_method')['F1'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(4)

        # Calculate time statistics
        time_stats = self.df.groupby('reduction_method')['Time'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(4)

        # Perform post-hoc tests if significant differences are found
        accuracy_significant = accuracy_friedman[1] < self.alpha
        f1_significant = f1_friedman[1] < self.alpha

        accuracy_post_hoc = self.perform_bonferroni_test(accuracy_pivot) if accuracy_significant else None
        f1_post_hoc = self.perform_bonferroni_test(f1_pivot) if f1_significant else None

        return {
            'accuracy_summary': summary_accuracy,
            'f1_summary': summary_f1,
            'time_stats': time_stats,
            'accuracy_friedman': accuracy_friedman,
            'f1_friedman': f1_friedman,
            'accuracy_post_hoc': accuracy_post_hoc,
            'f1_post_hoc': f1_post_hoc,
            'accuracy_significant': accuracy_significant,
            'f1_significant': f1_significant
        }

    def visualize_results(self, results: dict) -> tuple[plt.Figure, plt.Figure]:
        """Create visualizations for the analysis results."""
        # Figure 1: Accuracy and F1 scores comparison
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot accuracy
        summary_acc = results['accuracy_summary']
        ax1.bar(range(len(summary_acc)), summary_acc['mean'])
        ax1.errorbar(range(len(summary_acc)), summary_acc['mean'],
                     yerr=summary_acc['std'], fmt='none', color='black', capsize=5)
        ax1.set_xticks(range(len(summary_acc)))
        ax1.set_xticklabels(summary_acc.index, rotation=45, ha='right')
        ax1.set_title('Mean Accuracy by Reduction Method')
        ax1.set_ylabel('Accuracy')

        # Plot F1 scores
        summary_f1 = results['f1_summary']
        ax2.bar(range(len(summary_f1)), summary_f1['mean'])
        ax2.errorbar(range(len(summary_f1)), summary_f1['mean'],
                     yerr=summary_f1['std'], fmt='none', color='black', capsize=5)
        ax2.set_xticks(range(len(summary_f1)))
        ax2.set_xticklabels(summary_f1.index, rotation=45, ha='right')
        ax2.set_title('Mean F1 Score by Reduction Method')
        ax2.set_ylabel('F1 Score')

        plt.tight_layout()

        # Figure 2: Execution times
        fig2, ax3 = plt.subplots(figsize=(8, 6))
        time_stats = results['time_stats']
        ax3.bar(range(len(time_stats)), time_stats['mean'])
        ax3.errorbar(range(len(time_stats)), time_stats['mean'],
                     yerr=time_stats['std'], fmt='none', color='black', capsize=5)
        ax3.set_xticks(range(len(time_stats)))
        ax3.set_xticklabels(time_stats.index, rotation=45, ha='right')
        ax3.set_title('Mean Execution Time by Reduction Method')
        ax3.set_ylabel('Time (seconds)')

        plt.tight_layout()

        return fig1, fig2

    def generate_report(self, results: dict, output_path: str):
        """Generate a detailed text report of the statistical analysis results."""
        with open(output_path, 'w') as f:
            f.write("Statistical Analysis Report - Reduction Methods\n")
            f.write("=" * 50 + "\n\n")

            # Accuracy Statistics
            f.write("Accuracy Statistics\n")
            f.write("-----------------\n")
            f.write(results['accuracy_summary'].to_string())
            f.write("\n\n")

            # F1 Score Statistics
            f.write("F1 Score Statistics\n")
            f.write("------------------\n")
            f.write(results['f1_summary'].to_string())
            f.write("\n\n")

            # Execution Time Statistics
            f.write("Execution Time Statistics\n")
            f.write("-----------------------\n")
            f.write(results['time_stats'].to_string())
            f.write("\n\n")

            # Friedman Test Results
            f.write("Friedman Test Results\n")
            f.write("--------------------\n")
            f.write("Accuracy:\n")
            stat, p = results['accuracy_friedman']
            f.write(f"Test Statistic: {stat:.4f}\n")
            f.write(f"P-value: {p:.4f}\n\n")

            f.write("F1 Score:\n")
            stat, p = results['f1_friedman']
            f.write(f"Test Statistic: {stat:.4f}\n")
            f.write(f"P-value: {p:.4f}\n")
            f.write(f"Significance level (alpha): {self.alpha}\n\n")

            # Post-hoc results
            if results['accuracy_significant']:
                f.write("Post-hoc Test Results for Accuracy (Bonferroni)\n")
                f.write("------------------------------------------\n")
                f.write(results['accuracy_post_hoc'].round(4).to_string())
                f.write("\n\n")

            if results['f1_significant']:
                f.write("Post-hoc Test Results for F1 Score (Bonferroni)\n")
                f.write("------------------------------------------\n")
                f.write(results['f1_post_hoc'].round(4).to_string())


def main(csv_path: str, output_dir: str = None, alpha: float = 0.05):
    """Main function to run the analysis."""
    analyzer = ReductionMethodAnalyzer(csv_path, alpha)
    results = analyzer.analyze_reduction_methods()

    # Print results to console
    print("\nAccuracy Statistics:")
    print(results['accuracy_summary'])

    print("\nF1 Score Statistics:")
    print(results['f1_summary'])

    print("\nExecution Time Statistics:")
    print(results['time_stats'])

    print("\nFriedman Test Results (Accuracy):")
    stat, p = results['accuracy_friedman']
    print(f"Statistic: {stat:.4f}")
    print(f"p-value: {p:.4f}")

    print("\nFriedman Test Results (F1 Score):")
    stat, p = results['f1_friedman']
    print(f"Statistic: {stat:.4f}")
    print(f"p-value: {p:.4f}")

    if output_dir:
        # Create visualizations
        fig1, fig2 = analyzer.visualize_results(results)
        fig1.savefig(f"{output_dir}/reduction_metrics_analysis.png")
        fig2.savefig(f"{output_dir}/reduction_time_analysis.png")
        analyzer.generate_report(results, f"{output_dir}/reduction_analysis.txt")
        plt.close('all')

    return results


if __name__ == "__main__":
    csv_path = "svm_mushroom_results_reduced.csv"
    output_dir = "plots_and_tables/svm_reduction"
    results = main(csv_path, output_dir)