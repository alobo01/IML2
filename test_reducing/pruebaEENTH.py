import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

from classes.KNN import KNNAlgorithm
from classes.ReductionKNN import ReductionKNN

output_dir = "eenth"
os.makedirs(output_dir, exist_ok=True)

# Generate 5 blobs with centers around [0, 0] and combinations of [1, 1]
centers = [[0, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
X, y = make_blobs(n_samples=500, centers=centers, cluster_std=0.6, random_state=42)
data_df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
data_df["Label"] = y

features = data_df.drop("Label",axis=1)
labels = data_df["Label"]

# Save original plot
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30, edgecolor='k', alpha=0.6)
plt.title("Original Generated Blobs with Mixed Centers")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig(os.path.join(output_dir, "original_blobs.png"))
plt.close()

ogknn = KNNAlgorithm(k=5)
ogknn.fit(features,labels)
knn = KNNAlgorithm(k=5)
knn.fit(features,labels)
reduction = ReductionKNN(ogknn,knn)

# Values of mu to evaluate
mu_values = [0.15, 0.25, 0.45, 0.65, 0.85]

# Generate and save filtered plots for each mu
for mu in mu_values:
    # Apply the reduction algorithm for the current mu value
    reducedIndices = reduction.editing_algorithm_estimating_class_probabilities_and_threshold(features, labels, mu=mu)
    reducedDF = data_df.loc[reducedIndices]
    X_filtered, y_filtered = reducedDF.drop("Label", axis=1).values, reducedDF["Label"].values

    # Plot the filtered dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_filtered, cmap='viridis', s=30, edgecolor='k', alpha=0.6)
    plt.title(f"Filtered Blobs after Applying EENTH Algorithm (mu={mu})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(os.path.join(output_dir, f"filtered_blobs_mu_{mu}.png"))
    plt.close()