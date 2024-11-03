import pandas as pd
from scipy import stats
import numpy as np
from datetime import datetime


class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.buffer = []

    def write(self, text):
        self.buffer.append(str(text))

    def save(self):
        with open(self.filename, 'w') as f:
            f.write('\n'.join(self.buffer))


def load_and_prepare_data(svm_file, knn_file, metric):
    """
    Load and prepare the data from both CSV files for a specific metric
    """
    # Read the CSV files
    svm_data = pd.read_csv(svm_file)
    knn_data = pd.read_csv(knn_file)

    # Extract values for the specified metric
    svm_values = svm_data[metric].values
    knn_values = knn_data[metric].values

    return svm_values, knn_values


def perform_wilcoxon_test(svm_values, knn_values, metric_name, logger):
    """
    Perform Wilcoxon signed-rank test and log results for a specific metric
    """
    # Calculate mean values
    svm_mean = np.mean(svm_values)
    knn_mean = np.mean(knn_values)

    # Calculate std deviation
    svm_std = np.std(svm_values)
    knn_std = np.std(knn_values)

    logger.write(f"\nStatistical results for {metric_name}:")
    logger.write("-" * 40)
    logger.write(f"\nMean {metric_name}:")
    logger.write(f"SVM: {svm_mean:.4f} +- {svm_std:.4f}")
    logger.write(f"KNN: {knn_mean:.4f} +- {knn_std:.4f}")

    # Check if there's enough variation in the data
    if np.ptp(np.concatenate([svm_values, knn_values])) < 0.1:
        logger.write(f"Skipping Wilcoxon test due to low {metric_name} variation.")
        return None

    # Perform Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(svm_values, knn_values)
    logger.write(f"\nWilcoxon Signed-Rank Test Results for {metric_name}:")
    logger.write("-" * 40)
    logger.write(f"Statistic: {statistic:.4f}")
    logger.write(f"P-value: {p_value:.4f}")

    # Interpret results
    alpha = 0.1
    logger.write("\nInterpretation:")
    if p_value < alpha:
        logger.write(f"There is a significant difference between the models (p < {alpha})")
        if svm_mean > knn_mean:
            logger.write(f"SVM performed significantly better than KNN for {metric_name}")
        else:
            logger.write(f"KNN performed significantly better than SVM for {metric_name}")
    else:
        logger.write(f"There is no significant difference between the models for {metric_name} (p >= {alpha})")
    logger.write("\n" + "=" * 60 + "\n")


def main():
    # File paths
    svm_file = "svm_hepatitis_results_best1.csv"
    knn_file = "top_knn_results.csv"

    # Create timestamp for the results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"wilcoxon_results.txt"

    # Initialize logger
    logger = Logger(results_file)

    # Write header information
    logger.write("Wilcoxon Test Results to compare the best SVM algorithm and the best KNN algorithm for Hepatitis")
    logger.write("===================")
    logger.write(f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.write(f"SVM data file: {svm_file}")
    logger.write(f"KNN data file: {knn_file}")
    logger.write("\n")

    # Metrics to analyze
    metrics = ['Accuracy', 'Time', 'F1']

    try:
        # Perform analysis for each metric
        for metric in metrics:
            logger.write(f"\nAnalyzing {metric}...")
            svm_values, knn_values = load_and_prepare_data(svm_file, knn_file, metric)
            perform_wilcoxon_test(svm_values, knn_values, metric, logger)

        # Save all results to file
        logger.save()
        print(f"Results have been saved to: {results_file}")

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logger.write(error_msg)
        logger.save()
        print(error_msg)
        print(f"Error details have been saved to: {results_file}")


if __name__ == "__main__":
    main()