import pandas as pd
from scipy import stats
from scikit_posthocs import posthoc_nemenyi_friedman
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(filename):
    df = pd.read_csv(filename)
    # Extract method names (everything after the last comma)
    df['Method'] = df['Model'].apply(lambda x: x.split(',')[-1].strip())
    # Pivot the data to get methods as columns and folds as rows
    accuracy_matrix = df.pivot(index='Dataset/Fold',
                               columns='Method',
                               values='Accuracy')
    return df, accuracy_matrix


def perform_friedman_test(accuracy_matrix):
    statistic, p_value = stats.friedmanchisquare(*[accuracy_matrix[col] for col in accuracy_matrix.columns])
    return statistic, p_value


def perform_nemenyi_test(accuracy_matrix):
    return posthoc_nemenyi_friedman(accuracy_matrix)


def create_boxplot(df, filename):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Method', y='Accuracy', data=df)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots_and_tables/svm_reduction/{filename}_boxplot.png")
    plt.close()


def create_heatmap(nemenyi_matrix, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(nemenyi_matrix, annot=True, cmap='RdYlBu', center=0.5)
    plt.tight_layout()
    plt.savefig(f"plots_and_tables/svm_reduction/{filename}_heatmap.png")
    plt.close()


def write_results(filename, friedman_stat, friedman_p, accuracy_matrix, nemenyi_matrix=None):
    with open(f"plots_and_tables/svm_reduction/{filename}_analysis.txt", 'w') as f:
        f.write("Statistical Analysis of SVM Results\n")
        f.write("==================================\n\n")

        # Write summary statistics
        f.write("Summary Statistics:\n")
        f.write("-----------------\n")
        summary = accuracy_matrix.agg(['mean', 'std', 'min', 'max']).round(4)
        f.write(summary.to_string())
        f.write("\n\n")

        # Write Friedman test results
        f.write("Friedman Test Results:\n")
        f.write("---------------------\n")
        f.write(f"Statistic: {friedman_stat:.4f}\n")
        f.write(f"P-value: {friedman_p:.4f}\n\n")

        if nemenyi_matrix is not None:
            f.write("Nemenyi Test Results:\n")
            f.write("--------------------\n")
            f.write("P-values matrix:\n")
            f.write(nemenyi_matrix.round(4).to_string())
            f.write("\n\n")

        # Write interpretation
        f.write("Interpretation:\n")
        f.write("--------------\n")
        if friedman_p < 0.05:
            f.write("The Friedman test shows significant differences between methods (p < 0.05).\n")
            if nemenyi_matrix is not None:
                f.write("The Nemenyi test was performed to identify specific differences between methods.\n")
                sig_pairs = []
                for i in range(len(nemenyi_matrix.columns)):
                    for j in range(i + 1, len(nemenyi_matrix.columns)):
                        if nemenyi_matrix.iloc[i, j] < 0.05:
                            sig_pairs.append(f"{nemenyi_matrix.columns[i]} vs {nemenyi_matrix.columns[j]}")
                if sig_pairs:
                    f.write("\nSignificant differences found between:\n")
                    for pair in sig_pairs:
                        f.write(f"- {pair}\n")
        else:
            f.write("The Friedman test shows no significant differences between methods (p >= 0.05).\n")


def main():
    # Load and prepare data
    df, accuracy_matrix = load_and_prepare_data('svm_mushroom_results_reduced.csv')

    # Perform Friedman test
    friedman_stat, friedman_p = perform_friedman_test(accuracy_matrix)

    # If Friedman test is significant, perform Nemenyi test
    nemenyi_matrix = None
    if friedman_p < 0.05:
        nemenyi_matrix = perform_nemenyi_test(accuracy_matrix)

    # Create visualizations
    create_boxplot(df, 'svm_results')
    if nemenyi_matrix is not None:
        create_heatmap(nemenyi_matrix, 'svm_results')

    # Write results to file
    write_results('svm_results', friedman_stat, friedman_p, accuracy_matrix, nemenyi_matrix)


if __name__ == "__main__":
    main()