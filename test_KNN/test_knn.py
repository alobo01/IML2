import multiprocessing
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from classes.Reader import DataPreprocessor
from classes.KNN import KNNAlgorithm, apply_weighting_method


# # 0. Load from .arff and preprocess the training data, then save it to .joblib
# train_preprocessor = DataPreprocessor('hepatitis.fold.000000.train.arff')
# train_preprocessor.fit(config_path='config.json')
# train_preprocessor.save("hepatitis_preprocessor.joblib")

# # 1. Load data from .joblib
loaded_preprocessor = DataPreprocessor().load("hepatitis_preprocessor.joblib")
train_data_preprocessed = loaded_preprocessor.transform()

# Separate features and labels for the training data
train_features = train_data_preprocessed.drop('Class', axis=1)
train_labels = train_data_preprocessed['Class']

# 2. Preprocess test data
test_data_preprocessed = loaded_preprocessor.transform(DataPreprocessor.load_arff("hepatitis.fold.000000.test.arff"))

# Separate features and labels for the test data
test_features = test_data_preprocessed.drop('Class', axis=1)
test_labels = test_data_preprocessed['Class']

# 3. Set up distance metrics, voting policies and weighting methods to test
distance_metrics = ['euclidean_distance', 'manhattan_distance', 'clark_distance']
voting_policies = ['majority_class', 'inverse_distance_weighted', 'shepard']
weighting_methods = ['equal_weight', 'information_gain_weight', 'reliefF_weight']

def get_weighted_features(weighted_train_features, method, k):
    if method in ['equal_weight', 'information_gain_weight']:
        return weighted_train_features[(method, None)]  # Ignore k, always return the same value
    else:
        return weighted_train_features[(method, k)]

# Function to evaluate the KNN model
def evaluate_knn(params):
    k, dist_metric, vote_policy, weighting_method, weighted_train_features, train_labels, test_features, test_labels = params
    print(
        "Executing configuration:\n"
        f" - k: {k}\n"
        f" - Distance Metric: {dist_metric}\n"
        f" - Voting Policy: {vote_policy}\n"
        f" - Weighting Method: {weighting_method}\n"
    )

    # Create the KNNAlgorithm object with the corresponding configuration
    knn = KNNAlgorithm(k=k, distance_metric=dist_metric, voting_policy=vote_policy, weighting_method=weighting_method)
    knn.fit(weighted_train_features, train_labels)

    # Make predictions on the test data
    predictions = knn.predict(test_features)

    # Evaluate the model
    accuracy = accuracy_score(test_labels, predictions)

    return {
        'k': k,
        'distance_metric': dist_metric,
        'voting_policy': vote_policy,
        'weighting_method': weighting_method,
        'accuracy': accuracy
    }

if __name__ == '__main__':

    print(f" - Number of Training Features: {len(train_features)}\n"
        f" - Number of Training Labels: {len(train_labels)}\n"
        f" - Number of Test Features: {len(test_features)}\n"
        f" - Number of Test Labels: {len(test_labels)}")

    start_preprocess_time = time.time()

    # Pre-process the weighted features for all weighting methods, to avoid repeating the calculations
    weighted_train_features = {
        ('equal_weight', None): apply_weighting_method(train_features, train_labels, 'equal_weight'),
        ('information_gain_weight', None): apply_weighting_method(train_features, train_labels,
                                                                  'information_gain_weight')
    }

    # The reliefF weighting method depends on the value of k, while the other 2 do not
    for k in [1, 3, 5, 7]:
        weighted_train_features[('reliefF_weight', k)] = apply_weighting_method(train_features, train_labels,
                                                                                'reliefF_weight', k)

    elapsed_preprocess_time = time.time() - start_preprocess_time
    print(f"Pre-processing time: {elapsed_preprocess_time:.2f} seconds\n")

    # 4. Prepare parameter sets for the pool

    params_list = [(k, dist_metric, vote_policy, weighting_method,
                    get_weighted_features(weighted_train_features, weighting_method, k),
                    train_labels, test_features, test_labels)
                   for k in [1, 3, 5, 7]
                   for dist_metric in distance_metrics
                   for vote_policy in voting_policies
                   for weighting_method in weighting_methods]

    # 5. Use a pool of 4 workers to parallelize the evaluation

    def collect_results(result):
        results.append(result)

    start_time = time.time()

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=8) as pool:
        results = []
        for params in params_list:
            pool.apply_async(evaluate_knn, args=(params,), callback=collect_results)
        pool.close()
        pool.join()


    elapsed_time = time.time() - start_time
    print(f"Evaluation time: {elapsed_time:.2f} seconds\n")

    # 6. Convert results to a pandas DataFrame
    results_df = pd.DataFrame(results)

    # 7. Find the configuration with the highest accuracy
    top_results = results_df.nlargest(10, 'accuracy')

    # 8. Print the top configurations
    for rank, (_, result) in enumerate(top_results.iterrows(), start=1):
        print(f"Top {rank} Configuration:")
        print(f"k: {result['k']}")
        print(f"Distance Metric: {result['distance_metric']}")
        print(f"Voting Policy: {result['voting_policy']}")
        print(f"Weighting Method: {result['weighting_method']}")
        print(f"Accuracy: {result['accuracy']:.6f}\n")
