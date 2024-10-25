from classes.Reader import DataPreprocessor
import os
from tabulate import tabulate


def save_dataframe_characteristics(df, folder_name="plots_and_tables"):
    # Create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Descriptive Statistics in Markdown format
    desc_stats = df.describe()
    desc_stats_md = tabulate(desc_stats, headers="keys", tablefmt="github")

    # Missing Values Analysis
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]  # Only show columns with missing values
    if not missing_values.empty:
        missing_values_df = missing_values.to_frame(name="Missing Values Count")
        missing_values_md = tabulate(missing_values_df, headers="keys", tablefmt="github")
    else:
        missing_values_md = "No missing values found."

    # Identify Numerical and Categorical Columns
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    numerical_columns_md = ", ".join(numerical_columns) if numerical_columns else "No numerical columns found."
    categorical_columns_md = ", ".join(categorical_columns) if categorical_columns else "No categorical columns found."

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

    print("Descriptive statistics, missing values analysis, and column types saved in Markdown format.")
    print(f"All characteristics of the DataFrame saved in '{folder_name}'.")


# Usage example:
reader = DataPreprocessor("../datasets/hepatitis/hepatitis.fold.000000.train.arff")
complete_df = reader.get_whole_dataset_as_df(
    reader,
    "../datasets/hepatitis/hepatitis.fold.000000.test.arff"
)

save_dataframe_characteristics(complete_df)



