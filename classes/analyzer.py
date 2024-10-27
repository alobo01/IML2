import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
from itertools import combinations

matplotlib.use('Agg')  # Set backend to Agg for better file saving


def save_dataframe_description_analysis(df, folder_name="plots_and_tables"):
    # Create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Descriptive Statistics in Markdown format
    desc_stats = df.describe()
    desc_stats_md = tabulate(desc_stats, headers="keys", tablefmt="github")

    # Missing Values Analysis
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=True)
    if not missing_values.empty:
        missing_values_df = missing_values.to_frame(name="Missing Values Count")
        missing_values_md = tabulate(missing_values_df, headers="keys", tablefmt="github")
    else:
        missing_values_md = "No missing values found."

    # Identify Numerical and Categorical Columns
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    numerical_columns_md = ", ".join(numerical_columns) if numerical_columns else "No numerical columns found."
    categorical_columns_md = ", ".join(
        map(str, categorical_columns)) if categorical_columns else "No categorical columns found."

    # Identify the class column (assumed to be 'Class' if present, otherwise last column)
    class_column = 'Class' if 'Class' in df.columns else df.columns[-1]

    # Class Distribution Analysis
    class_distribution = df[class_column].value_counts()
    class_distribution_df = class_distribution.to_frame(name="Count")
    class_distribution_md = tabulate(class_distribution_df, headers="keys", tablefmt="github")

    # Writing to Markdown file
    with open(os.path.join(folder_name, "descriptive_statistics.md"), "w") as f:
        f.write("# Descriptive Statistics\n\n")
        f.write(desc_stats_md)
        f.write("\n\n# Missing Values Analysis\n\n")
        f.write(missing_values_md)
        f.write("\n\n# Column Types\n\n")
        f.write("**Numerical Columns**:\n\n")
        f.write(numerical_columns_md)
        f.write("\n\n**Categorical Columns**:\n\n")
        f.write(categorical_columns_md)
        f.write("\n\n# Class Distribution\n\n")
        f.write(class_distribution_md)

    print("Descriptive statistics, missing values analysis, column types, and class distribution saved in Markdown format.")
    print(f"All characteristics of the DataFrame saved in '{folder_name}'.")


def analyze_feature_importance(df, n_top_features=5, class_column=None, folder_name="plots_and_tables"):
    """
    Calculate entropy and information gain for all features in relation to the class column,
    test feature combinations using KNN, and save results in markdown format.

    Parameters:
    df (pandas.DataFrame): Input DataFrame
    n_top_features (int): Number of top features to return
    class_column (str): Name of the class column. If None, uses last column
    folder_name (str): Folder to save the markdown file

    Returns:
    list: Names of the n most influential features
    """

    # Create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Identify class column if not provided
    if class_column is None:
        class_column = df.columns[-1]

    # Initialize dictionary to store results
    entropy_results = {}

    # Calculate class entropy
    le = LabelEncoder()
    class_labels = le.fit_transform(df[class_column])
    class_probabilities = np.unique(class_labels, return_counts=True)[1] / len(class_labels)
    class_entropy = entropy(class_probabilities, base=2)

    # Calculate information gain for each feature
    features = [col for col in df.columns if col != class_column]

    for feature in features:
        # Handle both numerical and categorical features
        if df[feature].dtype in ['int64', 'float64']:
            # For numerical features, use binning
            bins = min(10, len(df[feature].unique()))  # Maximum 10 bins
            df[f'{feature}_binned'] = pd.qcut(df[feature], q=bins, duplicates='drop')
            feature_values = df[f'{feature}_binned']
            df = df.drop(f'{feature}_binned', axis=1)
        else:
            feature_values = df[feature]

        # Calculate conditional entropy
        conditional_entropy = 0
        feature_value_counts = feature_values.value_counts()

        for value in feature_values.unique():
            value_mask = feature_values == value
            value_proportion = sum(value_mask) / len(df)

            # Calculate class distribution for this feature value
            value_class_counts = np.unique(class_labels[value_mask], return_counts=True)[1]
            value_class_probabilities = value_class_counts / sum(value_class_counts)

            # Add weighted entropy
            conditional_entropy += value_proportion * entropy(value_class_probabilities, base=2)

        # Calculate information gain
        information_gain = class_entropy - conditional_entropy
        entropy_results[feature] = {
            'information_gain': information_gain,
            'feature_entropy': entropy(feature_value_counts / len(df), base=2),
            'conditional_entropy': conditional_entropy
        }

    # Sort features by information gain
    sorted_features = sorted(entropy_results.items(),
                             key=lambda x: x[1]['information_gain'],
                             reverse=True)

    # Get top n features
    top_features = [feature for feature, _ in sorted_features[:n_top_features]]

    # Prepare results for markdown
    results_df = pd.DataFrame([
        {
            'Feature': feature,
            'Information Gain': f"{values['information_gain']:.4f}",
            'Feature Entropy': f"{values['feature_entropy']:.4f}",
            'Conditional Entropy': f"{values['conditional_entropy']:.4f}"
        }
        for feature, values in sorted_features
    ])

    # Test feature combinations with KNN
    knn_results = []
    plt.figure(figsize=(12, 8))

    # Prepare scaler
    scaler = StandardScaler()

    # Test different feature combinations
    for n_features in range(2, n_top_features + 1):
        # Get all combinations of n features from top features
        feature_combinations = combinations(top_features, n_features)

        for feature_set in feature_combinations:
            # Prepare data
            X = df[list(feature_set)]
            y = class_labels

            # Handle categorical features
            for col in X.columns:
                if X[col].dtype == 'object':
                    X.loc[:, col] = LabelEncoder().fit_transform(X[col])

            # Scale features
            X = scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train KNN
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train, y_train)

            # Get predictions and probabilities
            y_pred_proba = knn.predict_proba(X_test)[:, 1]

            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.plot(fpr, tpr, label=f'{n_features} features (AUC = {roc_auc:.2f})')

            # Store results
            knn_results.append({
                'Features': ', '.join(feature_set),
                'N Features': n_features,
                'PR AUC': f"{pr_auc:.4f}",
                'ROC AUC': f"{roc_auc:.4f}"
            })

    # Finalize ROC plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Feature Combinations')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(folder_name, "roc_curves.png"))
    plt.close()

    # Save results to markdown
    with open(os.path.join(folder_name, "feature_entropy_analysis.md"), "w") as f:
        f.write("# Feature Entropy Analysis\n\n")
        f.write("## Class Entropy\n\n")
        f.write(f"Base class entropy: {class_entropy:.4f}\n\n")
        f.write("## Feature Information Gain Rankings\n\n")
        f.write(tabulate(results_df, headers='keys', tablefmt='github'))
        f.write("\n\n## Top Features\n\n")
        for i, (feature, values) in enumerate(sorted_features[:n_top_features], 1):
            f.write(f"{i}. **{feature}** (Information Gain: {values['information_gain']:.4f})\n")

        f.write("\n\n## KNN Classification Results\n\n")
        knn_df = pd.DataFrame(knn_results)
        f.write(tabulate(knn_df, headers='keys', tablefmt='github'))
        f.write("\n\n## ROC Curves\n\n")
        f.write("![ROC Curves](roc_curves.png)\n")

    print(f"Feature entropy analysis saved in '{folder_name}/feature_entropy_analysis.md'")

    # Return the top n feature names
    return top_features


def save_feature_distributions_by_class(df, folder_name="plots_and_tables"):
    """
    Create and save distribution plots for features by class, optimized for imbalanced datasets.

    Parameters:
    -----------
    df : pandas.DataFrame
        The complete dataset containing features and class labels
    folder_name : str
        The name of the folder where plots will be saved (default: "plots_and_tables")
    """

    # Set style for better-looking plots
    plt.style.use('default')  # Using default style instead of seaborn

    # Create folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Create a subfolder for the distribution plots
    plots_folder = os.path.join(folder_name, "feature_distributions")
    os.makedirs(plots_folder, exist_ok=True)

    # Identify the class column (assumed to be 'Class' if present, otherwise last column)
    class_column = 'Class' if 'Class' in df.columns else df.columns[-1]

    # Calculate class distribution
    class_dist = df[class_column].value_counts()
    total_samples = len(df)
    class_percentages = (class_dist / total_samples * 100).round(2)

    # Get features (all columns except the class column)
    features = df.drop(columns=[class_column]).columns

    # Color palette for different classes - using distinct colors
    colors = ['#1f77b4', '#d62728']  # Blue and Red for better contrast

    try:
        for feature in features:
            # Create a figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Distribution of {feature} by Class\nClass Distribution: ' +
                         ', '.join([f'Class {k}: {v}%' for k, v in class_percentages.items()]))

            if df[feature].dtype in ['int64', 'float64']:
                # Left subplot: Normalized histogram (density)
                for idx, class_value in enumerate(sorted(df[class_column].unique())):
                    subset = df[df[class_column] == class_value]
                    data = subset[feature].dropna()

                    # Calculate histogram
                    hist, bins = np.histogram(data, bins='auto', density=True)
                    bin_centers = (bins[:-1] + bins[1:]) / 2

                    # Plot smoothed line
                    ax1.plot(bin_centers, hist, label=f'Class {class_value}',
                             color=colors[idx], alpha=0.7)
                    ax1.fill_between(bin_centers, hist, alpha=0.3, color=colors[idx])

                ax1.set_title('Density Plot (Normalized)')
                ax1.set_xlabel(feature)
                ax1.set_ylabel('Density')

                # Right subplot: Absolute counts with transparency
                for idx, class_value in enumerate(sorted(df[class_column].unique())):
                    subset = df[df[class_column] == class_value]
                    ax2.hist(subset[feature].dropna(),
                             bins=min(30, len(subset[feature].unique())),
                             alpha=0.5,
                             label=f'Class {class_value}',
                             color=colors[idx])
                ax2.set_title('Absolute Counts')
                ax2.set_xlabel(feature)
                ax2.set_ylabel('Count')

            else:
                # For categorical features, create bar plots
                # Left subplot: Normalized proportions
                width = 0.35  # Width of bars
                x = np.arange(len(df[feature].unique()))

                for idx, class_value in enumerate(sorted(df[class_column].unique())):
                    subset = df[df[class_column] == class_value]
                    props = subset[feature].value_counts(normalize=True)
                    props = props.reindex(sorted(df[feature].unique()), fill_value=0)
                    ax1.bar(x + idx * width,
                            props.values,
                            width,
                            alpha=0.7,
                            label=f'Class {class_value}',
                            color=colors[idx])

                ax1.set_xticks(x + width / 2)
                ax1.set_xticklabels(sorted(df[feature].unique()))
                ax1.set_title('Proportions within Each Class')
                ax1.set_ylabel('Proportion')

                # Right subplot: Absolute counts
                for idx, class_value in enumerate(sorted(df[class_column].unique())):
                    subset = df[df[class_column] == class_value]
                    counts = subset[feature].value_counts()
                    counts = counts.reindex(sorted(df[feature].unique()), fill_value=0)
                    ax2.bar(x + idx * width,
                            counts.values,
                            width,
                            alpha=0.7,
                            label=f'Class {class_value}',
                            color=colors[idx])

                ax2.set_xticks(x + width / 2)
                ax2.set_xticklabels(sorted(df[feature].unique()))
                ax2.set_title('Absolute Counts')
                ax2.set_ylabel('Count')

                # Rotate labels if categorical
                ax1.tick_params(axis='x', rotation=45)
                ax2.tick_params(axis='x', rotation=45)

            # Add legends and grid
            ax1.legend(title='Class')
            ax2.legend(title='Class')
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)

            # Adjust layout
            plt.tight_layout()

            # Save the plot
            plot_filename = os.path.join(plots_folder, f'{feature}_distribution.png')
            try:
                plt.savefig(plot_filename,
                            format='png',
                            dpi=300,
                            bbox_inches='tight',
                            facecolor='white',
                            edgecolor='none')
            except Exception as e:
                print(f"Error saving plot for feature {feature}: {str(e)}")

            # Clean up
            plt.close('all')

    except Exception as e:
        print(f"Error during plot generation: {str(e)}")

    print(f"Feature distribution plots saved in '{plots_folder}'")
    print(f"Class distribution information saved in '{plots_folder}/class_distribution.txt'")
