import pandas as pd
from scipy import stats
import numpy as np


def load_and_prepare_data(svm_file, knn_file):
    """
    Load and prepare the data from both CSV files
    """
    # Read the CSV files
    svm_data = pd.read_csv(svm_file)
    knn_data = pd.read_csv(knn_file)

    # Extract accuracy values
    svm_accuracy = svm_data['Accuracy'].values
    knn_accuracy = knn_data['Accuracy'].values

    return svm_accuracy, knn_accuracy


def perform_wilcoxon_test(svm_accuracy, knn_accuracy):
    """
    Perform Wilcoxon signed-rank test and print results
    """
    # Perform Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(svm_accuracy, knn_accuracy)

    # Calculate mean accuracies
    svm_mean = np.mean(svm_accuracy)
    knn_mean = np.mean(knn_accuracy)

    # Print results
    print("\nWilcoxon Signed-Rank Test Results:")
    print("---------------------------------")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print("\nMean Accuracies:")
    print(f"SVM: {svm_mean:.4f}")
    print(f"KNN: {knn_mean:.4f}")

    # Interpret results
    alpha = 0.05
    print("\nInterpretation:")
    if p_value < alpha:
        print(f"There is a significant difference between the models (p < {alpha})")
        if svm_mean > knn_mean:
            print("SVM performed significantly better than KNN")
        else:
            print("KNN performed significantly better than SVM")
    else:
        print(f"There is no significant difference between the models (p >= {alpha})")


def main():
    # File paths
    svm_file = "svm_hepatitis_results_best1.csv"
    knn_file = "top_knn_results.csv"

    try:
        # Load and prepare data
        svm_accuracy, knn_accuracy = load_and_prepare_data(svm_file, knn_file)

        # Perform statistical test
        perform_wilcoxon_test(svm_accuracy, knn_accuracy)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()