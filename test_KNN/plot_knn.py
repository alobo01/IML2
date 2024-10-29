import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from classes.KNN import KNNAlgorithm

# Generate a 2D toy dataset
X, y = make_blobs(n_samples=200, centers=4, n_features=2, random_state=420)
train_features = pd.DataFrame(X)
train_labels = pd.Series(y)

# Create 3 test samples
test_features = pd.DataFrame(np.random.uniform(-10, 10, size=(3, 2)))

# Initialize the KNN model
knn = KNNAlgorithm(k=3, distance_metric='euclidean_distance', weighting_method='equal_weight', voting_policy='majority_class')
knn.fit(train_features, train_labels)

# Get the neighbors for the test samples
neighbors = knn.get_neighbors(test_features, return_distances=True)
predictions = knn.predict(test_features)

# Plot the dataset and test samples with neighbors
plt.figure(figsize=(8, 6))
plt.scatter(train_features[0], train_features[1], c=train_labels, cmap='viridis', alpha=0.7)
plt.scatter(test_features[0], test_features[1], c=predictions, cmap='viridis', s=100, marker='x')

for (feats, dists), labels in neighbors:
    for feat, dist, label in zip(feats.values, dists, labels):
        plt.scatter(feat[0], feat[1], c=f'C{label}', s=80, alpha=0.7)

print(predictions)
for _, labels in neighbors:
    print(labels)

plt.title("KNN Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(['Train Samples', 'Test Samples'] + [f'Neighbor {i}' for i in range(3)], loc='upper left')
plt.show()