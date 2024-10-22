import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

def read_pickle_and_generate_report(pickle_file, output_html='results_report.html'):
    # Load results from pickle file
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)

    # Convert the results dictionary into a pandas DataFrame
    df = pd.DataFrame(results)

    # Start writing to the HTML file
    with open(output_html, 'w') as f_html:
        # Write an introductory section
        f_html.write('<h1>Experiment Results Report</h1>')
        f_html.write(
            '<p>This report summarizes the results of the experiment, including accuracy, ROC-AUC, recall, precision, F1-score, and reduction percentage across different methods.</p>')

        # Generate and insert tables into the HTML
        f_html.write('<h2>Summary of Results</h2>')
        f_html.write(df.to_html(index=False))  # Write the table as HTML

        # Plot the metrics for each method
        metrics = ['accuracy', 'roc_auc', 'recall', 'precision', 'f1_score', 'reduction_percentage']

        #df["method"] = df["method"].apply(str)

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='method', y=metric, data=df)
            plt.title(f'{metric.capitalize()} Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot as image in memory (to embed in HTML)
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            # Embed the image in the HTML
            f_html.write(f'<h2>{metric.capitalize()} Comparison</h2>')
            f_html.write(f'<img src="data:image/png;base64,{img_base64}" />')

            plt.close()

    print(f"Report saved as {output_html}")


#Pedro

#Bruno

#read_pickle_and_generate_report('knn_comparison_results.pkl',output_html="results_knn_report.html")


#Mari

#Antonio

read_pickle_and_generate_report('knn_reduction_comparison_results.pkl',output_html="results_reduction_report.html")