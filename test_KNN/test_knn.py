import pandas as pd
import time
from sklearn.metrics import accuracy_score
from Reader import DataPreprocessor
from KNN import KNNAlgorithm

# # 0. Load from .arff and preprocess the training data, then save it to .joblib
# train_preprocessor = DataPreprocessor('hepatitis.fold.000000.train.arff')
# train_preprocessor.fit(config_path='config.json')
# train_preprocessor.save("hepatitis_preprocessor.joblib")

# # 1. Load data from .joblib
loaded_preprocessor = DataPreprocessor().load("hepatitis_preprocessor.joblib")
train_data_preprocessed = loaded_preprocessor.transform()

# Separate features and labels
train_features = train_data_preprocessed.drop('Class', axis=1)
train_labels = train_data_preprocessed['Class']

# # 2. Load and preprocess test data
test_data_preprocessed = loaded_preprocessor.transform(DataPreprocessor.load_arff("hepatitis.fold.000000.test.arff")[0])

# Separate features and labels
test_features = test_data_preprocessed.drop('Class', axis=1)
test_labels = test_data_preprocessed['Class']

# # 3. Set up the test
# Distance metrics and voting policies to test
distance_metrics = ['euclidean_distance', 'manhattan_distance', 'clark_distance']
voting_policies = ['majority_class', 'inverse_distance_weighted', 'shepard']

results = []
start_time = time.time()

# Test KNN with k = 1, 3, 5 and 7 for each distance metric and voting policy
for k in [1, 3, 5, 7]:
    for dist_metric in distance_metrics:
        for vote_policy in voting_policies:

            # Create the KNNAlgorithm object with the corresponding configuration
            knn = KNNAlgorithm(k=k, distance_metric=dist_metric, voting_policy=vote_policy)
            knn.fit(train_features, train_labels)

            # 4. Make predictions on the test data
            predictions = knn.predict(test_features)

            # 5. Evaluate the model
            accuracy = accuracy_score(test_labels, predictions)
            results.append({
                'k': k,
                'distance_metric': dist_metric,
                'voting_policy': vote_policy,
                'accuracy': accuracy
            })

elapsed_time = time.time() - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds\n")

# Convert the results list into a pandas DataFrame
results_df = pd.DataFrame(results)

# Find the configuration with the highest accuracy
best_result = results_df.loc[results_df['accuracy'].idxmax()]

# Print the configuration with the highest accuracy
print("Best Configuration:")
print(f"k: {best_result['k']}")
print(f"Distance Metric: {best_result['distance_metric']}")
print(f"Voting Policy: {best_result['voting_policy']}")
print(f"Accuracy: {best_result['accuracy']:.4f}\n")

# Display pivot tables for each value of k
for k in [1, 3, 5, 7]:
    print(f"Accuracy Table for k = {k}")
    pivot_table = results_df[results_df['k'] == k].pivot(index='distance_metric', columns='voting_policy', values='accuracy')
    print(pivot_table)
    print("\n")