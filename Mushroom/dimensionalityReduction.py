# Import necessary libraries
from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Step 1: Load and preprocess ARFF dataset
def load_and_preprocess_arff(filepath, class_column='class'):
    """Load ARFF data and preprocess it into a Pandas DataFrame."""
    # Load the ARFF dataset
    data, meta = arff.loadarff(filepath)

    # Convert ARFF data to a Pandas DataFrame
    df = pd.DataFrame(data)
    # Decode byte strings in the dataset
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')

    # Split the DataFrame into features and labels (class column)
    features = df.drop([class_column], axis=1)
    features_labels = df[class_column]

    return df, features, features_labels


# Step 2: Preprocess categorical data using Label Encoding
def label_encode_features(features):
    """Apply label encoding to all categorical features."""
    le = LabelEncoder()
    for col in features.columns:
        features[col] = le.fit_transform(features[col].astype(str))
    return features


# Step 3: Compute Negative Entropy for each feature value and class with verbose option
def compute_negative_entropy(df, features, class_column='class', verbose=False):
    """Compute negative entropy for each feature and return the results."""
    negative_entropy_results = {}

    for feature in features.columns:
        feature_entropy = {}
        for feature_value in df[feature].unique():
            # Filter dataset by the feature value
            feature_df = df[df[feature] == feature_value]

            # Get class distribution for this feature value
            class_distribution = feature_df[class_column].value_counts(normalize=True)

            # Compute entropy of the class distribution
            entropy = -np.sum(class_distribution * np.log(class_distribution))

            # Compute negative entropy
            negative_entropy = -entropy

            feature_entropy[feature_value] = negative_entropy

            # Print detailed entropy information if verbose is enabled
            if verbose:
                print(
                    f"Feature: {feature}, Value: {feature_value}, Class Distribution: {class_distribution.to_dict()}, Negative Entropy: {negative_entropy}")

        negative_entropy_results[feature] = feature_entropy

    return negative_entropy_results


# Step 4: Plot histograms for feature distribution by class
def plot_feature_distributions_by_class(df, features, class_column='class'):
    """Plot histograms for feature distributions, colored by class labels."""
    for feature in features.columns:
        # Create a histogram plot for each feature
        plt.figure(figsize=(10, 6))
        for class_value in df[class_column].unique():
            subset = df[df[class_column] == class_value]
            plt.hist(subset[feature], alpha=0.5, label=f'Class {class_value}', bins=10)

        plt.title(f'Feature: {feature} - Distribution by Class')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.legend(title='Class')
        plt.show()


# Step 5: Summarize and rank features by negative entropy
def summarize_and_sort_entropy(negative_entropy_results):
    """Summarize and sort features based on total negative entropy."""
    feature_entropy_summary = {}

    for feature, values in negative_entropy_results.items():
        # Summing up negative entropy for each feature across all values
        total_negative_entropy = sum(values.values())
        feature_entropy_summary[feature] = total_negative_entropy

    # Sort features by total negative entropy in descending order
    sorted_features = sorted(feature_entropy_summary.items(), key=lambda x: x[1], reverse=True)

    return sorted_features


# Step 6: Calculate number of classes (unique values) for each feature
def calculate_number_of_classes(df, features):
    """Calculate the number of unique classes for each feature."""
    num_classes_per_feature = {}
    for feature in features.columns:
        num_classes_per_feature[feature] = df[feature].nunique()
    return num_classes_per_feature


# Step 7: Feature selection and model training
def train_knn_with_top_features(df, features, features_labels, sorted_features, top_n=4):
    """Train a KNN model using the top N features based on negative entropy."""
    # Select the top N features
    top_features = [feature for feature, _ in sorted_features[:top_n]]
    print(f"Top {top_n} Features: {top_features}")

    # Extract the top N features
    selected_features = features[top_features]

    # Apply OneHotEncoder to the selected features
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(selected_features)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(encoded_features, features_labels, test_size=0.3,
                                                        random_state=42)

    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy of KNN model: {accuracy:.2f}")
    return accuracy


# Main execution flow
if __name__ == "__main__":
    # Load and preprocess the dataset
    filepath = '../datasets/mushroom/mushroom.fold.000000.train.arff'
    df, features, features_labels = load_and_preprocess_arff(filepath)

    # Encode the features
    features = label_encode_features(features)

    # Compute negative entropy with verbose output
    negative_entropy_results = compute_negative_entropy(df, features, verbose=True)

    # Plot histograms for feature distributions by class
    plot_feature_distributions_by_class(df, features)

    # Summarize and sort features by total negative entropy
    sorted_features = summarize_and_sort_entropy(negative_entropy_results)

    # Calculate and display the number of classes (unique values) for each feature
    num_classes = calculate_number_of_classes(df, features)
    print("\nNumber of unique classes per feature:")
    for feature, num_class in num_classes.items():
        print(f"Feature: {feature}, Number of unique classes: {num_class}")

    # Train a KNN model using the top 4 features
    train_knn_with_top_features(df, features, features_labels, sorted_features)
