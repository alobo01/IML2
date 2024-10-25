import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from classes.Reader import DataPreprocessor


# Function to compute PCA and plot results
def compute_pca_and_plot(train_features, train_labels, test_features, test_labels, fold_number):
    # Combine train and test features for PCA
    all_features = pd.concat([train_features, test_features], ignore_index=True)

    # Fit PCA on all data
    pca = PCA()
    pca.fit(all_features)

    # Transform train and test data
    train_pca = pca.transform(train_features)
    test_pca = pca.transform(test_features)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_

    # Plot cumulative explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(explained_variance), marker='o')
    plt.title(f'Fold {fold_number} - Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.show()

    # Plot first 2 principal components
    plt.figure(figsize=(8, 6))
    for class_label in np.unique(train_labels):
        indices = train_labels == class_label
        plt.scatter(train_pca[indices, 0], train_pca[indices, 1], label=f'Train Class {class_label}', alpha=0.7)
    for class_label in np.unique(test_labels):
        indices = test_labels == class_label
        plt.scatter(test_pca[indices, 0], test_pca[indices, 1], marker='x', label=f'Test Class {class_label}',
                    alpha=0.7)
    plt.title(f'Fold {fold_number} - PCA Plot (First 2 Components)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3D Plot of first 3 principal components
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for class_label in np.unique(train_labels):
        indices = train_labels == class_label
        ax.scatter(train_pca[indices, 0], train_pca[indices, 1], train_pca[indices, 2],
                   label=f'Train Class {class_label}', alpha=0.7)
    for class_label in np.unique(test_labels):
        indices = test_labels == class_label
        ax.scatter(test_pca[indices, 0], test_pca[indices, 1], test_pca[indices, 2], marker='x',
                   label=f'Test Class {class_label}', alpha=0.7)
    ax.set_title(f'Fold {fold_number} - PCA Plot (First 3 Components)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.legend()
    plt.show()


# Assuming you have the load_fold_data function defined as provided
def load_fold_data(fold_number, dataset_path):
    train_file = os.path.join(dataset_path, f'hepatitis.fold.{fold_number:06d}.train.arff')
    test_file = os.path.join(dataset_path, f'hepatitis.fold.{fold_number:06d}.test.arff')

    loaded_preprocessor = DataPreprocessor().load("hepatitis_preprocessor.joblib")
    train_data_preprocessed = loaded_preprocessor.transform(DataPreprocessor.load_arff(train_file)[0])
    test_data_preprocessed = loaded_preprocessor.transform(DataPreprocessor.load_arff(test_file)[0])

    # Separate features and labels for train and test data
    train_features = train_data_preprocessed.drop('Class', axis=1)
    train_labels = train_data_preprocessed['Class']
    test_features = test_data_preprocessed.drop('Class', axis=1)
    test_labels = test_data_preprocessed['Class']

    return train_features, train_labels, test_features, test_labels


# Main code to iterate over all folds
dataset_path = '..\\datasets\\hepatitis'
N_folds = 10  # Replace with the actual number of folds

for fold_number in range(1, N_folds + 1):
    train_features, train_labels, test_features, test_labels = load_fold_data(fold_number, dataset_path)
    compute_pca_and_plot(train_features, train_labels, test_features, test_labels, fold_number)
