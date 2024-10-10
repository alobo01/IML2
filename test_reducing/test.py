import pandas as pd
import time
import multiprocessing
from sklearn.metrics import accuracy_score
from Reader import DataPreprocessor
from KNN import KNNAlgorithm
from ReductionKNN import ReductionKNN


def reduction_worker(train_data, test_data, method):
    """
    Function to perform dataset reduction and evaluate performance.
    Args:
        train_data (DataFrame): Training data including features and labels.
        test_data (tuple): Tuple containing test features and labels.
        method (str): The reduction method to use.

    Returns:
        dict: Dictionary with execution times, accuracy values, and dataset sizes.
    """
    train_features, train_labels = train_data.drop('Class', axis=1), train_data['Class']
    test_features, test_labels = test_data

    # Initialize original KNN and reduction KNN
    ogKNN = KNNAlgorithm(k=3)
    ogKNN.fit(train_features, train_labels)
    reduction_knn = ReductionKNN(ogKNN, KNNAlgorithm(k=3))

    # Measure reduction time
    start_time = time.time()
    reduced_data = reduction_knn.apply_reduction(train_data, method)
    reduction_time = time.time() - start_time

    # Separate features and labels of reduced data
    reduced_train_features, reduced_train_labels = reduced_data.drop('Class', axis=1), reduced_data['Class']

    # Inference on original dataset
    original_predictions = ogKNN.predict(test_features)
    original_accuracy = accuracy_score(test_labels, original_predictions)

    # Fit and inference on reduced dataset
    reduction_knn.reducedKNN.fit(reduced_train_features, reduced_train_labels)
    start_inference = time.time()
    reduced_predictions = reduction_knn.reducedKNN.predict(test_features)
    inference_time = time.time() - start_inference
    reduced_accuracy = accuracy_score(test_labels, reduced_predictions)

    return {
        'Method': method,
        'Reduction Time (s)': round(reduction_time, 4),
        'Inference Time (s)': round(inference_time, 4),
        'Original Accuracy': round(original_accuracy, 4),
        'Reduced Accuracy': round(reduced_accuracy, 4),
        'Original Size': len(train_data),
        'Reduced Size': len(reduced_data),
    }


def main():
    # 1. Load preprocessed training data from .joblib
    loaded_preprocessor = DataPreprocessor().load("hepatitis_preprocessor.joblib")
    train_data_preprocessed = loaded_preprocessor.transform()

    # 2. Separate features and labels for the training data
    train_features = train_data_preprocessed.drop('Class', axis=1)
    train_labels = train_data_preprocessed['Class']
    train_data = pd.concat([train_features, train_labels], axis=1)  # Combine features and labels into a DataFrame

    # 3. Preprocess test data
    test_data_preprocessed = loaded_preprocessor.transform(
        DataPreprocessor.load_arff("hepatitis.fold.000000.test.arff")[0])

    # Separate features and labels for the test data
    test_features = test_data_preprocessed.drop('Class', axis=1)
    test_labels = test_data_preprocessed['Class']

    # 4. Create a pool of workers and apply reductions in parallel
    with multiprocessing.Pool(processes=3) as pool:
        # Define the tasks for each worker
        methods = ['GCNN', 'RENN', 'IB2']
        results = pool.starmap(
            reduction_worker, [(train_data, (test_features, test_labels), method) for method in methods]
        )

    # 5. Print the results in a table format
    print(f"{'Method':<10} {'Reduction Time (s)':<20} {'Inference Time (s)':<20} "
          f"{'Original Accuracy':<20} {'Reduced Accuracy':<20} "
          f"{'Original Size':<15} {'Reduced Size':<15}")
    print("=" * 125)
    for result in results:
        print(f"{result['Method']:<10} {result['Reduction Time (s)']:<20} {result['Inference Time (s)']:<20} "
              f"{result['Original Accuracy']:<20} {result['Reduced Accuracy']:<20} "
              f"{result['Original Size']:<15} {result['Reduced Size']:<15}")


if __name__ == "__main__":
    main()
