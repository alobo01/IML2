import numpy as np
import matplotlib.pyplot as plt

from classes.KNN import KNNAlgorithm
from classes.ReductionKNN import ReductionKNN
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from matplotlib import colormaps
from scipy.io import arff
import pandas as pd
# Generate synthetic dataset
#X, y = make_blobs(n_samples=300, centers=[[0, 0], [1, 1]], random_state=42, cluster_std=0.5)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Load the ARFF file
data, meta = arff.loadarff('grid.fold.000000.train.arff')
df = pd.DataFrame(data)
df["class"] = label_encoder.fit_transform(df["class"])

features, labels = df.drop("class", axis=1), df["class"]

X, y = df.drop("class", axis=1).values, df["class"].values


# Create a color dictionary from the colormap
unique_classes = np.unique(y)
cmap = colormaps['prism']
color_dict = {cls: cmap(i / len(unique_classes)) for i, cls in enumerate(unique_classes)}



def enn_filter(X, y, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    mask = y_pred == y
    return X[mask], y[mask]


# Apply ENN filtering
X_enn, y_enn = enn_filter(X, y, k=3)


def distance_to_nearest_enemy(X, y):
    distances = np.zeros(len(X))
    for idx in range(len(X)):
        X_enemy = X[y != y[idx]]
        dist = np.linalg.norm(X_enemy - X[idx], axis=1)
        distances[idx] = np.min(dist)
    return distances


dist_to_enemy = distance_to_nearest_enemy(X_enn, y_enn)
data = np.hstack((X_enn, y_enn.reshape(-1, 1), dist_to_enemy.reshape(-1, 1)))
data_sorted = data[np.argsort(-data[:, -1])]


def drop3(data, k=5, show_plots=True):
    X_s = data[:, :2]
    y_s = data[:, 2].astype(int)
    keep_indices = np.ones(len(X_s), dtype=bool)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_s)

    associates = [[] for _ in range(len(X_s))]
    distances, indices = nbrs.kneighbors(X_s)
    for idx, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            associates[neighbor].append(idx)


    unique_classes = np.unique(y_s)  # Get unique class labels

    # Initialize the class count tracker
    class_count = {class_label: np.sum(y_s == class_label) for class_label in unique_classes}

    for idx in range(len(X_s)):
        if not keep_indices[idx]:
            keep_indices[idx] = False
            continue

        xi = X_s[idx].reshape(1, -1)
        yi = y_s[idx]
        temp_indices = keep_indices.copy()
        X_temp = X_s[temp_indices]
        y_temp = y_s[temp_indices]

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_temp, y_temp)

        associate_indices = associates[idx]
        if not associate_indices:
            # Check if removing this point would leave any class without representation
            if class_count[yi] > 1:  # Ensure at least one node per class
                keep_indices[idx] = False
                class_count[yi] -= 1  # Update class count after removal
            continue

        X_assoc = X_s[associate_indices]
        y_assoc = y_s[associate_indices]

        y_pred_with = knn.predict(X_assoc)
        correct_with = np.sum(y_pred_with == y_assoc)


        temp_indices[idx] = False
        X_temp = X_s[temp_indices]
        y_temp = y_s[temp_indices]

        knn.fit(X_temp, y_temp)
        y_pred_without = knn.predict(X_assoc)
        correct_without = np.sum(y_pred_without == y_assoc)

        # Show intermediate plots only if show_plots is True
        if show_plots:
            plt.figure(figsize=(8, 6))

            for class_label in unique_classes:
                plt.scatter(
                    X_s[y_s == class_label, 0], X_s[y_s == class_label, 1],
                    color=color_dict[class_label], edgecolor='k', alpha=0.5,
                    label=f'Class {class_label}'
                )

            plt.scatter(X_s[idx, 0], X_s[idx, 1], color='yellow', edgecolor='black', s=200, label='Evaluated Point')

            for i, class_label in enumerate(unique_classes):
                plt.scatter(
                    X_assoc[y_assoc == class_label, 0], X_assoc[y_assoc == class_label, 1],
                    color=color_dict[class_label], s=150, edgecolor='k'
                )

            plt.title(f'Correct with: {correct_with}, Correct without: {correct_without}')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.show()

        if correct_without >= correct_with:
            # Check if removing this point would leave any class without representation
            if class_count[yi] > 1:  # Ensure at least one node per class
                keep_indices[idx] = False
                class_count[yi] -= 1  # Update class count after removal

        if np.sum(keep_indices) == k:
            break

    return X_s[keep_indices], y_s[keep_indices]


# Apply DROP3 with optional intermediate plots
#X_drop3, y_drop3 = drop3(data_sorted, k=5, show_plots=False)
ogknn = KNNAlgorithm()
ogknn.fit(features,labels)
knn = KNNAlgorithm()
knn.fit(features,labels)
reduction = ReductionKNN(ogknn,knn)

reducedDF = reduction.apply_reduction(df,"DROP3")
X_drop3, y_drop3 = reducedDF.drop("class",axis=1).values, reducedDF["class"].values
# Original Data Plot
plt.figure(figsize=(8, 8))
for class_label in unique_classes:
    plt.scatter(X[y == class_label, 0], X[y == class_label, 1], color=color_dict[class_label], edgecolor='k', alpha=1, label=f'Class {class_label}')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# ENN Reduction Plot
plt.figure(figsize=(8, 8))
for class_label in unique_classes:
    plt.scatter(X_enn[y_enn == class_label, 0], X_enn[y_enn == class_label, 1], color=color_dict[class_label], edgecolor='k', alpha=1, label=f'Class {class_label}')
plt.title('After ENN Reduction')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# DROP3 Reduction Plot
plt.figure(figsize=(8, 8))
for class_label in unique_classes:
    plt.scatter(X_drop3[y_drop3 == class_label, 0], X_drop3[y_drop3 == class_label, 1], color=color_dict[class_label], edgecolor='k', alpha=1, s=100, label=f'Class {class_label}')
plt.title('After DROP3 Reduction')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Comparison of Original and DROP3 Reduction Plot
plt.figure(figsize=(8, 8))
for class_label in unique_classes:
    plt.scatter(X[y == class_label, 0], X[y == class_label, 1], color=color_dict[class_label], edgecolor='k', alpha=1, s=20, label=f'Original Class {class_label}')
    plt.scatter(X_drop3[y_drop3 == class_label, 0], X_drop3[y_drop3 == class_label, 1], color=color_dict[class_label], edgecolor='k', alpha=0.5, s=100, label=f'DROP3 Class {class_label}')
plt.title('Comparison of Original and DROP3 Reduction')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
print("para")