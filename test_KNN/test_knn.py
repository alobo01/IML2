import multiprocessing
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from Reader import DataPreprocessor
from KNN import KNNAlgorithm

# 1. Load preprocessed training data from .joblib
loaded_preprocessor = DataPreprocessor().load("hepatitis_preprocessor.joblib")
train_data_preprocessed = loaded_preprocessor.transform()

# Separate features and labels for the training data
train_features = train_data_preprocessed.drop('Class', axis=1)
train_labels = train_data_preprocessed['Class']

# 2. Preprocess test data
test_data_preprocessed = loaded_preprocessor.transform(DataPreprocessor.load_arff("hepatitis.fold.000000.test.arff")[0])

# Separate features and labels for the test data
test_features = test_data_preprocessed.drop('Class', axis=1)
test_labels = test_data_preprocessed['Class']

# 3. Set up distance metrics and voting policies to test
distance_metrics = ['euclidean_distance', 'manhattan_distance', 'clark_distance']
voting_policies = ['majority_class', 'inverse_distance_weighted', 'shepard']

results = []
start_time = time.time()




# Function to evaluate the KNN model
def evaluate_knn(params):
    k, dist_metric, vote_policy, train_features, train_labels, test_features, test_labels = params
    print(
        "Executing configuration:\n"
        f" - k: {k}\n"
        f" - Distance Metric: {dist_metric}\n"
        f" - Voting Policy: {vote_policy}\n"
    )
    # Create the KNNAlgorithm object with the corresponding configuration
    knn = KNNAlgorithm(k=k, distance_metric=dist_metric, voting_policy=vote_policy)
    knn.fit(train_features, train_labels)

    # Make predictions on the test data
    predictions = knn.predict(test_features)

    # Evaluate the model
    accuracy = accuracy_score(test_labels, predictions)

    return {
        'k': k,
        'distance_metric': dist_metric,
        'voting_policy': vote_policy,
        'accuracy': accuracy
    }

# 4. Prepare parameter sets for the pool
params_list = [(k, dist_metric, vote_policy, train_features, train_labels, test_features, test_labels)
               for k in [1, 3, 5, 7]
               for dist_metric in distance_metrics
               for vote_policy in voting_policies]

if __name__ == '__main__':

    print(f" - Number of Training Features: {len(train_features)}\n"
        f" - Number of Training Labels: {len(train_labels)}\n"
        f" - Number of Test Features: {len(test_features)}\n"
        f" - Number of Test Labels: {len(test_labels)}")

    # 5. Use a pool of 4 workers to parallelize the evaluation
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(evaluate_knn, params_list)


    elapsed_time = time.time() - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds\n")

    # 6. Convert results to a pandas DataFrame
    results_df = pd.DataFrame(results)

    # 7. Find the configuration with the highest accuracy
    best_result = results_df.loc[results_df['accuracy'].idxmax()]

    # Print the best configuration
    print("Best Configuration:")
    print(f"k: {best_result['k']}")
    print(f"Distance Metric: {best_result['distance_metric']}")
    print(f"Voting Policy: {best_result['voting_policy']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}\n")

    # 8. Display pivot tables for each value of k
    for k in [1, 3, 5, 7]:
        print(f"Accuracy Table for k = {k}")
        pivot_table = results_df[results_df['k'] == k].pivot(index='distance_metric', columns='voting_policy', values='accuracy')
        print(pivot_table)
        print("\n")
