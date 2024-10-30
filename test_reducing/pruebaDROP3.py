import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

# Generate synthetic dataset
X, y = make_blobs(n_samples=300, centers=[[0, 0], [1, 1]], random_state=42, cluster_std=0.5)

# Visualize the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
plt.title('Original Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

def enn_filter(X, y, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    mask = y_pred == y
    return X[mask], y[mask]

# Apply ENN filtering
X_enn, y_enn = enn_filter(X, y, k=3)

# Visualize the filtered dataset
plt.figure(figsize=(8, 6))
plt.scatter(X_enn[:, 0], X_enn[:, 1], c=y_enn, cmap='bwr', edgecolor='k')
plt.title('After ENN Filtering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

from sklearn.neighbors import NearestNeighbors

def distance_to_nearest_enemy(X, y):
    distances = np.zeros(len(X))
    for idx in range(len(X)):
        # Find instances of the opposite class
        X_enemy = X[y != y[idx]]
        # Compute distances to enemies
        dist = np.linalg.norm(X_enemy - X[idx], axis=1)
        distances[idx] = np.min(dist)
    return distances

# Compute distances
dist_to_enemy = distance_to_nearest_enemy(X_enn, y_enn)

# Combine data and distances
data = np.hstack((X_enn, y_enn.reshape(-1, 1), dist_to_enemy.reshape(-1, 1)))

# Sort instances by distance to nearest enemy (descending)
data_sorted = data[np.argsort(-data[:, -1])]


def drop3(data, k=3):
    # Initialize
    X_s = data[:, :2]
    y_s = data[:, 2]
    X_t = X_s.copy()
    y_t = y_s.copy()
    keep_indices = np.ones(len(X_s), dtype=bool)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_s)

    # Precompute associates
    # Q is associate of P if P \in N(Q)
    associates = [[] for _ in range(len(X_s))]
    distances, indices = nbrs.kneighbors(X_s)
    for idx, neighbors in enumerate(indices):
        # Exclude the point itself
        for neighbor in neighbors[1:]:
            associates[neighbor].append(idx)

    # Iterate over instances
    for idx in range(len(X_s)):
        if not keep_indices[idx]:
            continue

        # Current instance
        xi = X_s[idx].reshape(1, -1)
        yi = y_s[idx]

        # Temporarily remove instance
        temp_indices = keep_indices.copy()

        X_temp = X_s[temp_indices]
        y_temp = y_s[temp_indices]

        # Classifier without current instance
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_temp, y_temp)

        # Check associates
        associate_indices = associates[idx]
        if not associate_indices: # If it has no associates this point is not important for any decision
            # So correct_without >= correct_with always true
            keep_indices[idx] = False
            continue

        X_assoc = X_t[associate_indices]
        y_assoc = y_t[associate_indices]

        # Classification with and without the instance
        y_pred_with = knn.predict(X_assoc)
        correct_with = np.sum(y_pred_with == y_assoc)

        # Remove instance and reclassify
        temp_indices[idx] = False
        knn.fit(X_temp, y_temp)
        y_pred_without = knn.predict(X_assoc)
        correct_without = np.sum(y_pred_without == y_assoc)

        # Decide whether to remove the instance
        if correct_without >= correct_with:
            keep_indices[idx] = False

    # Return reduced dataset
    return X_s[keep_indices], y_s[keep_indices]

# Apply DROP3
X_drop3, y_drop3 = drop3(data_sorted, k=3)

# Visualize the final dataset
plt.figure(figsize=(8, 6))
plt.scatter(X_drop3[:, 0], X_drop3[:, 1], c=y_drop3, cmap='bwr', edgecolor='k')
plt.title('After DROP3 Reduction')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
