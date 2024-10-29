import argparse
from classes.ReductionKNN import ReductionKNN
from classes.KNNAlgorithm import KNNAlgorithm
from classes.SVM import SVM
from sklearn.metrics import accuracy_score
import numpy as np

# Argument parser setup
parser = argparse.ArgumentParser(description="Run dataset reduction and model training.")

# Dataset and Reduction method arguments
parser.add_argument("--dataset", choices=["Hepatitis", "Mushroom"], required=True, help="Dataset to use.")
parser.add_argument("--reduction_method", choices=["GCNN", "DROP3", "EENTH", "NONE"], default="NONE", help="Dataset reduction method to apply.")

# Model arguments
parser.add_argument("--model", choices=["KNN", "SVM"], required=True, help="Model type to use.")
parser.add_argument("--k", type=int, help="K value for KNN model.")
parser.add_argument("--distance_metric", choices=["euclidean_distance", "manhattan_distance", "clark_distance"], help="Distance metric for KNN.")
parser.add_argument("--voting_policy", choices=["majority_class", "inverse_distance_weighted", "shepard"], help="Voting policy for KNN.")
parser.add_argument("--weighting_method", choices=["equal_weight", "information_gain_weight", "reliefF_weight"], help="Weighting method for KNN.")

parser.add_argument("--C", type=float, choices=[0.1, 1, 10, 100], help="C value for SVM.")
parser.add_argument("--kernel", choices=["linear", "rbf", "poly", "sigmoid"], help="Kernel type for SVM.")

args = parser.parse_args()

# Dataset loading
dataset = args.dataset
print(f"Selected Dataset: {dataset}")
# Load the dataset, e.g., as a DataFrame: df = load_dataset(dataset)



# Apply reduction method if specified
reduction_method = args.reduction_method
print(f"Selected Reduction Method: {reduction_method}")
if reduction_method != "NONE":
    reduction_knn = ReductionKNN()
    reduced_df = reduction_knn.apply_reduction(df, reduction_method)
else:
    reduced_df = df

# Configure and train the selected model
if args.model == "KNN":
    print("Configuring KNN model...")
    if args.k and args.distance_metric and args.voting_policy and args.weighting_method:
        knn = KNNAlgorithm(k=args.k, distance_metric=args.distance_metric, voting_policy=args.voting_policy, weighting_method=args.weighting_method)
        knn.fit(reduced_df, train_labels)  # Assume train_labels are loaded
        predictions = knn.predict(test_features)  # Assume test_features are loaded
        accuracy = accuracy_score(test_labels, predictions)  # Assume test_labels are loaded
        print(f"KNN Model Accuracy: {accuracy}")
    else:
        print("Please specify k, distance metric, voting policy, and weighting method for KNN.")
elif args.model == "SVM":
    print("Configuring SVM model...")
    if args.C and args.kernel:
        svm_classifier = SVM(train_data=reduced_df, train_labels=train_labels, kernel=args.kernel, C=args.C, gamma='auto')
        svm_classifier.train()
        accuracy = svm_classifier.evaluate(test_features, test_labels)  # Assume test_labels are loaded
        print(f"SVM Model Accuracy: {accuracy[0]}")
    else:
        print("Please specify C and kernel for SVM.")
else:
    print("Invalid model selected.")
