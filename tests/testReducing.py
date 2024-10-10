import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from KNN import KNNAlgorithm
import pandas as pd
from ReductionKNN import ReductionKNN  # Assuming the previous code is in ReductionKNN.py


def generate_dataset():
    """
    Generates a 2D dataset with three classes for demonstration purposes.

    Returns:
        DataFrame: A DataFrame containing the generated dataset.
    """
    # Generate two interleaving half circles
    X1, y1 = make_moons(n_samples=300, noise=0.1, random_state=42)
    y1[y1 == 0] = 2  # Change label 1 to 2

    # Generate a blob for the third class
    X2, y2 = make_blobs(n_samples=200, centers=1, cluster_std=0.6, random_state=42)

    # Combine the datasets
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # Create a DataFrame
    df = pd.DataFrame(X, columns=['X', 'Y'])
    df['Label'] = y

    return df


def plot_dataset(ax, data, title):
    """
    Plots a 2D dataset with different colors for each class.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        data (DataFrame): The dataset to plot.
        title (str): The title for the plot.
    """
    colors = ['r', 'g', 'b']
    for label in data['Label'].unique():
        subset = data[data['Label'] == label]
        ax.scatter(subset['X'], subset['Y'], c=colors[int(label)], label=f'Class {label}', alpha=0.6)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


def main():
    # Generate the dataset
    data = generate_dataset()

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_data[['X', 'Y']].to_numpy(), train_data['Label'].to_numpy())

    # Initialize the ReductionKNN object
    reduction_knn = ReductionKNN(KNeighborsClassifier(n_neighbors=1))

    # Set up the plot
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Comparison of KNN Reduction Methods', fontsize=16)

    # Plot the original dataset


    # Apply each reduction method and plot the results
    for i, method in enumerate(['GCNN', 'RENN', 'IB2']):
        reduced_data = reduction_knn.apply_reduction(train_data, method)
        plot_dataset(axs[i, 0], train_data, 'Original Dataset')
        plot_dataset(axs[i, 1], reduced_data, f'{method} Reduced Dataset')

        # Print the reduction in dataset size
        print(f"{method} reduced the dataset from {len(train_data)} to {len(reduced_data)} points")

        # Evaluate and print the accuracies
        accuracies = reduction_knn.evaluate(test_data)
        print(f"{method} - Original Accuracy: {accuracies['original_accuracy']:.4f}, "
              f"Reduced Accuracy: {accuracies['reduced_accuracy']:.4f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()