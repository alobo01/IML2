import pandas as pd


def select_top_model(input_csv, output_csv):
    # Read the input CSV
    df = pd.read_csv(input_csv)

    # Group by Model and calculate mean accuracy
    model_avg_accuracy = df.groupby('Model')['Accuracy'].mean().reset_index()

    # Find the model with the highest average accuracy
    top_model = model_avg_accuracy.loc[model_avg_accuracy['Accuracy'].idxmax(), 'Model']

    # Filter the original dataframe to keep only entries of the top model
    top_model_entries = df[df['Model'] == top_model]

    # Save the filtered entries to a new CSV
    top_model_entries.to_csv(output_csv, index=False)

    print(f"Top performing model: {top_model}")
    print(f"Average accuracy: {model_avg_accuracy['Accuracy'].max():.4f}")

    return top_model_entries

# Example usage
select_top_model('knn_base_results.csv', 'top_knn_results.csv')