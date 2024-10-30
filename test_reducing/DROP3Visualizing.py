import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from matplotlib import colormaps
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Generate synthetic dataset
X, y = make_blobs(n_samples=500, centers=[[0, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]], random_state=42, cluster_std=0.5)

# Create a color dictionary from the colormap
unique_classes = np.unique(y)
cmap = colormaps['prism']
color_dict = {cls: cmap(i-(len(unique_classes)/2) / len(unique_classes)) for i, cls in enumerate(unique_classes)}

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ENN Filter function
def enn_filter(X, y, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    mask = y_pred == y
    return X[mask], y[mask]

# Distance to Nearest Enemy function
def distance_to_nearest_enemy(X, y):
    distances = np.zeros(len(X))
    for idx in range(len(X)):
        X_enemy = X[y != y[idx]]
        dist = np.linalg.norm(X_enemy - X[idx], axis=1)
        distances[idx] = np.min(dist)
    return distances

# DROP3 Reduction function
def drop3(data, k=3):
    X_s = data[:, :2]
    y_s = data[:, 2].astype(int)
    keep_indices = np.ones(len(X_s), dtype=bool)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_s)

    associates = [[] for _ in range(len(X_s))]
    distances, indices = nbrs.kneighbors(X_s)
    for idx, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            associates[neighbor].append(idx)

    temp_indicesOG = keep_indices.copy()
    for idx in range(len(X_s)):
        if not keep_indices[idx]:
            continue

        xi = X_s[idx].reshape(1, -1)
        yi = y_s[idx]

        X_temp = X_s[temp_indicesOG]
        y_temp = y_s[temp_indicesOG]

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_temp, y_temp)

        associate_indices = associates[idx]
        if not associate_indices:
            keep_indices[idx] = False
            continue

        X_assoc = X_s[associate_indices]
        y_assoc = y_s[associate_indices]

        y_pred_with = knn.predict(X_assoc)
        correct_with = np.sum(y_pred_with == y_assoc)

        temp_indices = temp_indicesOG.copy()
        temp_indices[idx] = False
        X_temp = X_s[temp_indices]
        y_temp = y_s[temp_indices]

        knn.fit(X_temp, y_temp)
        y_pred_without = knn.predict(X_assoc)
        correct_without = np.sum(y_pred_without == y_assoc)

        if correct_without >= correct_with:
            keep_indices[idx] = False

        if np.sum(keep_indices) == k:
            break

    return X_s[keep_indices], y_s[keep_indices]

# Timing ENN Reduction
start = time.time()
X_enn, y_enn = enn_filter(X_train, y_train, k=3)
enn_time = time.time() - start

# Preparing data for DROP3
dist_to_enemy = distance_to_nearest_enemy(X_enn, y_enn)
data_sorted = np.hstack((X_enn, y_enn.reshape(-1, 1), dist_to_enemy.reshape(-1, 1)))

# Timing DROP3 Reduction
start = time.time()
X_drop3, y_drop3 = drop3(data_sorted, k=3)
drop3_time = time.time() - start

# Classifier training and accuracy evaluation
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
total_accuracy = knn.score(X_test, y_test)

knn.fit(X_drop3, y_drop3)
drop3_accuracy = knn.score(X_test, y_test)

# Summary Statistics
print(f"Total Accuracy (Before Reduction): {total_accuracy:.4f}")
print(f"Accuracy (After DROP3): {drop3_accuracy:.4f}")
print(f"Execution Time (ENN Reduction): {enn_time:.4f} seconds")
print(f"Execution Time (DROP3 Reduction): {drop3_time:.4f} seconds")
print(f"Percentage of Points Remaining (ENN): {len(X_enn) / len(X_train) * 100:.2f}%")
print(f"Percentage of Points Remaining (DROP3): {len(X_drop3) / len(X_enn) * 100:.2f}%")

# Identify misclassified points in DROP3
knn.fit(X_drop3, y_drop3)
y_test_pred_drop3 = knn.predict(X_test)
misclassified = X_test[y_test_pred_drop3 != y_test]

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Before Reduction
axes[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=[color_dict[label] for label in y_train], edgecolor='k')
axes[0, 0].set_title("Before Reduction")
axes[0, 0].set_xlabel("Feature 1")
axes[0, 0].set_ylabel("Feature 2")

# After ENN
axes[0, 1].scatter(X_enn[:, 0], X_enn[:, 1], c=[color_dict[label] for label in y_enn], edgecolor='k')
axes[0, 1].set_title("After ENN Reduction")
axes[0, 1].set_xlabel("Feature 1")
axes[0, 1].set_ylabel("Feature 2")

# After DROP3
axes[1, 0].scatter(X_drop3[:, 0], X_drop3[:, 1], c=[color_dict[label] for label in y_drop3], edgecolor='k')
axes[1, 0].set_title("After DROP3 Reduction")
axes[1, 0].set_xlabel("Feature 1")
axes[1, 0].set_ylabel("Feature 2")

# All Data with Misclassified Points
axes[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=[color_dict[label] for label in y_test], edgecolor='k', alpha=0.5)
axes[1, 1].scatter(misclassified[:, 0], misclassified[:, 1], color=[color_dict[label] for label in y_test[y_test_pred_drop3 != y_test]], s=100, edgecolor='black', label='Misclassified')
axes[1, 1].set_title("All Data with Misclassified Points (DROP3)")
axes[1, 1].set_xlabel("Feature 1")
axes[1, 1].set_ylabel("Feature 2")
axes[1, 1].legend()

plt.tight_layout()
plt.show()
