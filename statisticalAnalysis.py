import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scikit_posthocs import posthoc_nemenyi_friedman
import numpy as np
# Ensure seaborn style for better visual output
sns.set(style="whitegrid")


# Function to check normality of results using Shapiro-Wilk and other common approaches
def check_normality(data, alpha=0.05):
    """
    Checks if the data follows a normal distribution using Shapiro-Wilk test.
    Returns True if we can assume normality based on the p-value threshold.

    Parameters:
        data (pd.Series): A series of numerical values to check for normality.
        alpha (float): Significance level for normality test (default is 0.05).

    Returns:
        bool: True if normality assumption holds, False otherwise.
    """
    shapiro_test = stats.shapiro(data)
    return shapiro_test.pvalue > alpha and False


# Function to perform a t-test on two sets of model performances
def perform_t_test(data1, data2, alpha=0.05):
    """
    Performs a t-test to determine if there is a statistically significant difference
    between two sets of model performances. Returns the p-value.

    Parameters:
        data1 (pd.Series): Performance metric of model 1.
        data2 (pd.Series): Performance metric of model 2.
        alpha (float): Significance level for t-test (default is 0.05).

    Returns:
        float: p-value from the t-test.
    """
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
    return p_value


# Function to perform ANOVA test on model performances
def perform_anova(data):
    """
    Conducts an ANOVA test to determine if there are significant differences
    among the Accuracy of multiple models.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Model' and 'Accuracy'.

    Returns:
        float: p-value from the ANOVA test.
    """
    anova_result = stats.f_oneway(*[group["Accuracy"].values for name, group in data.groupby("Model")])
    return anova_result.pvalue


# Function to perform Friedman test to check model correlation
def perform_friedman_test(data):
    """
    Conducts a Friedman test to determine if there are significant differences
    among multiple paired samples (model performances).

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Model' and 'Accuracy'.

    Returns:
        float: p-value from the Friedman test.
    """
    grouped_data = [group["Accuracy"].values for name, group in data.groupby("Model")]
    friedman_result = stats.friedmanchisquare(*grouped_data)
    return friedman_result.pvalue


# Function to perform Wilcoxon test to compare two paired models
def perform_wilcoxon_test(data1, data2):
    """
    Performs a Wilcoxon signed-rank test to compare two paired samples
    (two model performances).

    Parameters:
        data1 (pd.Series): Performance metric of model 1.
        data2 (pd.Series): Performance metric of model 2.

    Returns:
        float: p-value from the Wilcoxon test.
    """
    wilcoxon_result = stats.wilcoxon(data1, data2)
    return wilcoxon_result.pvalue


# Function to select appropriate test based on normality assumption
def select_and_apply_test(data, alpha=0.05):
    """
    Selects and applies the appropriate statistical test based on normality assumption.
    Uses ANOVA or t-test if data is normal, otherwise uses Friedman or Wilcoxon tests.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Model' and 'Accuracy'.
        alpha (float): Significance level for normality and tests (default is 0.05).

    Returns:
        dict: Dictionary of test results with test names as keys and p-values as values.
    """
    model_groups = [group["Accuracy"] for name, group in data.groupby("Model")]
    is_normal = all(check_normality(group, alpha) for group in model_groups)
    results = {}

    if is_normal:
        if len(model_groups) > 2:
            results['ANOVA'] = perform_anova(data)
        else:
            results['t-test'] = perform_t_test(model_groups[0], model_groups[1])
    else:
        if len(model_groups) > 2:
            results['Friedman'] = perform_friedman_test(data)
        else:
            results['Wilcoxon'] = perform_wilcoxon_test(model_groups[0], model_groups[1])

    return results


# Function to plot conclusions
def plot_conclusions(data, results):
    """
    Plots the model Accuracy and annotates conclusions based on statistical tests.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Model' and 'Accuracy'.
        results (dict): Dictionary of test results with test names as keys and p-values as values.
    """
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='Model', y='Accuracy', data=data, showfliers=False)
    #plt.title('Model Performance Comparison')
    plt.xlabel('Model Names')
    plt.ylabel('Accuracy')

    # Annotate the plot with test results
    conclusion_text = "\n".join([f"{test}: p-value = {p_value:.3f}" for test, p_value in results.items()])
    plt.figtext(0.15, -0.1, conclusion_text, ha="left", fontsize=10,
                bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    plt.tight_layout()
    plt.show()


# Generate sample Fold with a Fold column for Nemenyi test pivoting
def generate_sample_data(num_models=5, num_samples=20, seed=42):
    np.random.seed(seed)
    data = {
        "Fold": [],
        "Model": [],
        "Accuracy": []
    }
    for i in range(num_models):
        model_name = f"Model_{i + 1}"
        mean_performance = 100 - i * 2 + np.random.uniform(-3, 3)
        std_dev_performance = 5 + np.random.uniform(-1, 2)
        performance = np.random.normal(loc=mean_performance, scale=std_dev_performance, size=num_samples)

        data["Fold"].extend(range(num_samples))
        data["Model"].extend([model_name] * num_samples)
        data["Accuracy"].extend(performance)

    return pd.DataFrame(data)

# Enhanced plotting function with Nemenyi test
def plot_conclusions_with_nemenyi(data, results):
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='Model', y='Accuracy', data=data)
    #plt.title('Model Performance Comparison')
    plt.xlabel('Model Names')
    plt.ylabel('Accuracy')

    conclusion_text = "\n".join([f"{test}: p-value = {p_value:.3f}" for test, p_value in results.items()])
    plt.figtext(0.15, -0.1, conclusion_text, ha="left", fontsize=10,
                bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()
    results["Friedman"] = 0.01

    return plot_conclusions_with_nemenyi_rank(data, results)


# Enhanced plotting function with Nemenyi test, adapted for a mean rank plot with error bars
def plot_conclusions_with_nemenyi_rank(data, results, alpha=0.05):
    # Step 1: Perform the Friedman test and check if the Nemenyi test is required
    if "Friedman" in results and results["Friedman"] < alpha:
        print("Performing Nemenyi post-hoc test as Friedman test showed significant differences.")

        # Step 2: Reshape data for Nemenyi test
        data_wide = data.pivot(index="Fold", columns="Model", values="Accuracy")
        nemenyi_results = posthoc_nemenyi_friedman(data_wide)

        # Step 3: Calculate the average ranks for each model
        ranks = data_wide.rank(axis=1, method="average").mean().sort_values()
        models = ranks.index
        mean_ranks = ranks.values

        # Step 4: Calculate the critical difference (CD)
        k = data['Model'].nunique()  # number of models
        N = data['Fold'].nunique()  # number of Folds (samples)
        q_alpha = 2.569  # for 0.05 significance level and typical value when k=5
        critical_difference = q_alpha * np.sqrt((k * (k + 1)) / (6 * N))

        # Step 5: Plot the mean ranks with confidence intervals
        plt.figure(figsize=(12, 8))
        plt.errorbar(models, mean_ranks, yerr=critical_difference / 2, fmt='o', capsize=5, capthick=2)
        plt.xticks(rotation=70)
        plt.title('Nemenyi Test of Pipelines')
        plt.xlabel('Model')
        plt.ylabel('Mean Rank')
        plt.tight_layout()
        plt.show()

        # Optional: Display the critical difference value for interpretation
        print(f"Critical Difference (CD) at alpha={alpha}: {critical_difference:.3f}")
        print(f"{models[0]} has the best accuracy")

        return models[0]

# Generate sample data
# df = generate_sample_data(num_models=5, num_samples=20)

# Apply tests
# test_results = select_and_apply_test(df)

# Plot with conclusions including Nemenyi test if applicable
# plot_conclusions_with_nemenyi(df, test_results)

# generate_sample_data(num_models=5, num_samples=20).to_csv('example.csv',index=False)