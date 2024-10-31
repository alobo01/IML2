import heapq
from typing import List, Union, Any, Callable

from pandas import Series
from sklearn.feature_selection import mutual_info_classif
from sklearn_relief import ReliefF
import pandas as pd
import numpy as np
from functools import lru_cache

class KNNAlgorithm:
    """
    A k-Nearest Neighbors (kNN) Classifier supporting multiple voting techniques.
    """
    def __init__(self,
                 k: int = 3,
                 distance_metric: str = 'euclidean_distance',
                 weighting_method: str = 'equal_weight',
                 voting_policy: str = 'majority_class'):
        """
        Initialize the kNNAlgorithm with:
        - k: Number of neighbors to consider.
        - distance_metric: A function to calculate the distance between samples ('euclidean_distance', 'manhattan_distance', 'clark_distance').
        - weighting_method: Weighting method to apply to the distance ('equal_weight', 'information_gain_weight', 'reliefF_weight').
        - voting_policy: Voting technique to determine the class ('majority_class', 'inverse_distance_weighted', 'shepard').
        """
        self.k = k
        self.distance_metric = distance_metric
        self.distance = self.get_distance(distance_metric)
        self.weighting_method = weighting_method
        self.voting_policy = voting_policy
        self.vote = self.get_vote(voting_policy)
        self.train_features = None
        self.train_labels = None

    def fit(self, train_features: pd.DataFrame, train_labels: pd.Series):
        """
        Fit the kNN model with training data and labels.
        """
        self.train_labels = train_labels
        self.train_features = train_features

    def get_distance(self, distance_metric: str) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Choose the distance function based on the selected metric.
        """
        if distance_metric == 'euclidean_distance':
            return self.euclidean_distance
        elif distance_metric == 'manhattan_distance':
            return self.manhattan_distance
        elif distance_metric == 'clark_distance':
            return self.clark_distance
        else:
            raise ValueError(f"Unsupported distance function: {distance_metric}")

    def get_vote(self, voting_policy: str) -> Callable[[Any, Any], int]:
        """
        Choose the vote function based on the selected policy.
        """
        if voting_policy == 'majority_class':
            return self.majority_class_vote
        elif voting_policy == 'inverse_distance_weighted':
            return self.inverse_distance_weighted
        elif voting_policy == 'shepard':
            return self.shepard_vote
        else:
            raise ValueError(f"Unsupported voting policy: {voting_policy}")

    @staticmethod
    def euclidean_distance(vec1: pd.DataFrame, vec2: pd.DataFrame, memmap_file='distance_memmap.dat') -> pd.DataFrame:
        try:
            # Attempt to calculate distances directly in memory
            return np.sqrt(((vec1.values[:, np.newaxis] - vec2.values) ** 2).sum(axis=2))
        except MemoryError:
            print("MemoryError encountered, switching to disk-based computation.")

            # Define the shape of the result array for memory mapping
            shape = (vec1.shape[0], vec2.shape[0])

            # Use a memory-mapped file to handle large data
            with np.memmap(memmap_file, dtype='float32', mode='w+', shape=shape) as distance_memmap:
                for i in range(vec1.shape[0]):
                    distance_memmap[i, :] = np.sqrt(((vec1.values[i] - vec2.values) ** 2).sum(axis=1))

                # Convert memory-mapped data back to DataFrame
                return distance_memmap.copy()

    @staticmethod
    def manhattan_distance(vec1: pd.DataFrame, vec2: pd.DataFrame) -> pd.DataFrame:
        return np.abs(vec1.values[:, np.newaxis] - vec2.values).sum(axis=2)

    @staticmethod
    def clark_distance(vec1: pd.DataFrame, vec2: pd.DataFrame) -> pd.DataFrame:
        numerator = np.abs(vec1.values[:, np.newaxis] - vec2.values)
        denominator = vec1.values[:, np.newaxis] + vec2.values + 1e-10  # Avoid division by zero
        squared_ratio = (numerator / denominator) ** 2
        return np.nansum(squared_ratio, axis=2)  # Handle NaN values if any

    def get_neighbors(self, test_features: pd.DataFrame, custom_k=None, return_distances=False) -> list[
        tuple[Any, Any]]:
        """
        Identify the k nearest neighbors for each row in the test_features DataFrame
        using the custom distance methods.
        """
        if not custom_k:
            custom_k = self.k

        # Compute pairwise distances using self.distance()
        distance_matrix = self.distance(test_features, self.train_features)

        # For each test row, find the indices of the `custom_k` nearest train rows
        nearest_indices = np.argsort(distance_matrix, axis=1)[:, :custom_k]
        nearest_distances = np.take_along_axis(distance_matrix, nearest_indices, axis=1) if return_distances else None

        results = []
        for i, neighbors_idx in enumerate(nearest_indices):
            neighbors_features = self.train_features.iloc[neighbors_idx]
            neighbors_labels = self.train_labels.iloc[neighbors_idx]

            if return_distances:
                results.append(((neighbors_features, nearest_distances[i].tolist()), neighbors_labels))
            else:
                results.append((neighbors_features, neighbors_labels))

        return results

    def classify(self, test_features: pd.DataFrame) -> list[int]:
        """
        Classify each example in the test_features using k-nearest neighbors.
        """
        neighbors = self.get_neighbors(test_features, return_distances=True)
        return [self.vote(labels, distances) for ((features, distances), labels) in neighbors]

    def majority_class_vote(self, neighbors_labels, neighbors_distances) -> int:
        """
        Majority class voting: Return the most common class among the neighbors.
        """
        return neighbors_labels.mode()[0]

    def inverse_distance_weighted(self, neighbors_labels, neighbors_distances) -> int:
        """
        Inverse distance weighting: Weight the class labels by the inverse of their distances.
        """
        weights = 1 / (np.array(neighbors_distances) + 1e-5)
        class_vote = {}

        for i, (weight, label) in enumerate(zip(weights, neighbors_labels)):
            if label not in class_vote:
                class_vote[label] = 0
            class_vote[label] += weight

        return max(class_vote, key=class_vote.get)

    def shepard_vote(self, neighbors_labels, neighbors_distances) -> int:
        """
        Shepard's method: Use a power-based weighting.
        """
        weights = np.exp(-np.array(neighbors_distances))
        class_vote = {}

        for i, (weight, label) in enumerate(zip(weights, neighbors_labels)):
            if label not in class_vote:
                class_vote[label] = 0
            class_vote[label] += weight

        return max(class_vote, key=class_vote.get)

    def predict(self, test_features: pd.DataFrame) -> List[Union[int, float]]:
        """
        Predict the class labels for the test set.
        """
        return self.classify(test_features)

    def score(self, test_features: pd.DataFrame, test_labels: pd.Series) -> float:
        """
        Calculate the accuracy of the kNN classifier on the test data.
        """
        predictions = self.predict(test_features)
        correct_predictions = sum(pred == true for pred, true in zip(predictions, test_labels))
        accuracy = correct_predictions / len(test_labels)
        return accuracy


def apply_weighting_method(train_features: pd.DataFrame, train_labels: pd.Series, weighting_method : str, k : int = 1) -> pd.DataFrame:
    # Equal weights
    if weighting_method == 'equal_weight':
        weights = np.ones(train_features.shape[1])

    # Information Gain
    elif weighting_method == 'information_gain_weight':
        # Compute information gain for each feature
        weights = mutual_info_classif(train_features, train_labels)

    # ReliefF
    elif weighting_method == 'reliefF_weight':
        # Initialize ReliefF with the number of neighbors
        relief = ReliefF(k=k)
        relief.fit(train_features.values, train_labels)
        weights = relief.w_

    else:
        raise ValueError(f"Unsupported weighting method: {weighting_method}")

    return train_features.multiply(weights)