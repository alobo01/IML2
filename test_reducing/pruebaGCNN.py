import os
import numpy as np
import matplotlib.pyplot as plt
from classes.KNN import KNNAlgorithm
from classes.ReductionKNN import ReductionKNN
from sklearn.preprocessing import LabelEncoder
from matplotlib import colormaps
from scipy.io import arff
import pandas as pd


def generate_circular_dataset(n_inner=50, n_outer=200, inner_radius=1.0, outer_radius=4.0, noise=0.2):
    """
    Generates a 2D dataset with a small circle (inner class) surrounded by a circular crown (outer class).

    Args:
        n_inner (int): Number of samples in the inner circle (minority class).
        n_outer (int): Number of samples in the outer crown (majority class).
        inner_radius (float): Radius of the inner circle.
        outer_radius (float): Mean radius of the outer crown.
        noise (float): Standard deviation of noise added to the points.

    Returns:
        DataFrame: Features (X, Y) of the dataset.
        Series: Labels of the dataset.
    """
    # Generate points for the inner circle (Class B, minority class)
    angles_inner = 2 * np.pi * np.random.rand(n_inner)
    inner_x = inner_radius * np.cos(angles_inner) + np.random.normal(0, noise, n_inner)
    inner_y = inner_radius * np.sin(angles_inner) + np.random.normal(0, noise, n_inner)
    inner_points = np.column_stack((inner_x, inner_y))
    inner_labels = np.array(['B'] * n_inner)

    # Generate points for the outer crown (Class A, majority class)
    angles_outer = 2 * np.pi * np.random.rand(n_outer)
    outer_radii = np.random.normal(outer_radius, noise, n_outer)  # Random radius around outer_radius
    outer_x = outer_radii * np.cos(angles_outer)
    outer_y = outer_radii * np.sin(angles_outer)
    outer_points = np.column_stack((outer_x, outer_y))
    outer_labels = np.array(['A'] * n_outer)

    # Combine the points into a DataFrame
    features = np.vstack((inner_points, outer_points))
    labels = np.hstack((inner_labels, outer_labels))

    # Convert to DataFrame and Series
    features_df = pd.DataFrame(features, columns=['X', 'Y'])
    labels_series = pd.Series(labels, name='Label')

    unique_classes = np.unique(labels_series.values)
    cmap = colormaps['prism']
    color_dict = {cls: cmap(i / len(unique_classes)) for i, cls in enumerate(unique_classes)}
    # Plot the dataset for visualization
    plt.figure(figsize=(8, 8))
    plt.scatter(inner_x, inner_y, color=color_dict['B'], label='Inner Circle (Class B)', alpha=0.6)
    plt.scatter(outer_x, outer_y, color=color_dict['A'], label='Outer Crown (Class A)', alpha=0.6)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Circular Dataset: Inner Circle Surrounded by Crown")
    plt.show()

    return features_df, labels_series

features, labels = generate_circular_dataset()
X, y = features.values, labels.values

# Set up color mapping for each class
unique_classes = np.unique(y)
cmap = colormaps['prism']
color_dict = {cls: cmap(i / len(unique_classes)) for i, cls in enumerate(unique_classes)}

# Initialize KNN and reduction algorithms
ogknn = KNNAlgorithm(k=3)
ogknn.fit(features, labels)
knn = KNNAlgorithm(k=3)
knn.fit(features, labels)
reduction = ReductionKNN(ogknn, knn)

# Define rho values and output directory
rho_values = [0.8, 0.2, 0, 0.4, 0.6, 1]
output_dir = 'gcnn'
os.makedirs(output_dir, exist_ok=True)

# Iterate over each rho value to apply GCNN reduction and plot results
for rho in rho_values:
    keeped_indices = reduction.generalized_condensed_nearest_neighbor(features, labels, rho=rho, saveAnimation=True)
    X_gcnn, y_gcnn = features.iloc[keeped_indices].values, labels.iloc[keeped_indices].values

    # Plot original and reduced data
    plt.figure(figsize=(8, 8))
    for class_label in unique_classes:
        # Plot original data with lower opacity
        plt.scatter(X[y == class_label, 0], X[y == class_label, 1],
                    color=color_dict[class_label], edgecolor='k', alpha=0.3, s=20,
                    label=f'Original Class {class_label}')

        # Plot reduced data on top with higher opacity
        plt.scatter(X_gcnn[y_gcnn == class_label, 0], X_gcnn[y_gcnn == class_label, 1],
                    color=color_dict[class_label], edgecolor='k', alpha=0.8, s=60,
                    label=f'Selected Class {class_label}')

    plt.title(f'Comparison of Original and GCNN Reduction (rho={rho})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"comparison_plot_rho_{rho}.png"))
    plt.close()
