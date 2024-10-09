import pandas as pd
from KNN import KNNAlgorithm  # Assuming the KNN file is saved as KNN.py

# Create a dummy dataset
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [5, 4, 3, 2, 1],
    'Feature3': [2, 3, 4, 5, 6]
}
labels = [0, 1, 0, 1, 0]

train_data = pd.DataFrame(data)
train_labels = pd.Series(labels)

# Dummy test set
test_data = pd.DataFrame({
    'Feature1': [2.5, 4.5],
    'Feature2': [3.5, 1.5],
    'Feature3': [3.5, 5.5]
})

# Distance metrics and voting policies to test
distance_metrics = ['euclidean_distance', 'manhattan_distance', 'clark_distance']
voting_policies = ['majority_class', 'inverse_distance_weighted', 'shepard']

# Test KNN with k=1 and k=3 for each distance metric and voting policy
for k in [1, 3]:
    for dist_metric in distance_metrics:
        for vote_policy in voting_policies:
            knn = KNNAlgorithm(k=k, distance_metric=dist_metric, voting_policy=vote_policy)
            knn.fit(train_data, train_labels)
            predictions = knn.predict(test_data)
            print(f"k={k}, Distance Metric={dist_metric}, Voting Policy={vote_policy}")
            print(f"Predictions: {predictions}\n")
