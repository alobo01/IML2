from classes.Reader import DataPreprocessor
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Set backend to Agg for better file saving


def save_dataframe_description_analysis(df, folder_name="plots_and_tables"):
    # Create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Descriptive Statistics in Markdown format
    desc_stats = df.describe()
    desc_stats_md = tabulate(desc_stats, headers="keys", tablefmt="github")

    # Missing Values Analysis
    missing_values = df.isnull().sum() + (df == '?').sum()
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
