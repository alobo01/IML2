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

    def get_distance(self, distance_metric: str) -> Callable[[pd.Series, pd.Series], float]:
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

    def get_vote(self, voting_policy: str) -> Callable[[Any, Any, pd.Series], int]:
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
    #@lru_cache(maxsize=None)
    def euclidean_distance(vec1: pd.Series, vec2: pd.Series) -> float:
        """
        Euclidean distance metric.
        """
        return np.sqrt(np.sum((vec1 - vec2) ** 2))

    @staticmethod
    #@lru_cache(maxsize=None)
    def manhattan_distance(vec1: pd.Series, vec2: pd.Series) -> float:
        """
        Manhattan distance metric.
        """
        return np.sum(np.abs(vec1 - vec2))

    @staticmethod
    #@lru_cache(maxsize=None)
    def clark_distance(vec1: pd.Series, vec2: pd.Series) -> float:
        """
        Clark distance metric.
        """
        numerator = np.abs(vec1 - vec2)
        denominator = vec1 + vec2
        squared_ratio = (numerator / denominator) ** 2

        return np.sqrt(squared_ratio.sum())

    def get_neighbors(self, test_row: pd.Series, custom_k = None, return_distances = False) -> tuple[Any, Any]:
        """
        Identify the k nearest neighbors for a given test row.
        """
        if not custom_k:
            custom_k = self.k
        distances = [(index, self.distance(test_row, self.train_features.iloc[index])) for index in range(len(self.train_features))]
        #sorted_distances = sorted(distances, key=lambda x: x[1])
        all_distances = heapq.nsmallest(custom_k, distances, key=lambda x: x[1])
        if return_distances:
            neighbors_idx, distances = [], []
            for (idx,distance) in all_distances:
                neighbors_idx.append(idx)
                distances.append(distance)
            neighbors_features = self.train_features.iloc[neighbors_idx]
            neighbors_labels = self.train_labels.iloc[neighbors_idx]
            return (neighbors_features, distances), neighbors_labels

        else:
            neighbors_idx = [index for index, _ in all_distances]

            neighbors_features = self.train_features.iloc[neighbors_idx]
            neighbors_labels = self.train_labels.iloc[neighbors_idx]

            return neighbors_features, neighbors_labels

    def classify(self, test_row: pd.Series) -> Union[int, float]:
        """
        Classify a single example using k-nearest neighbors.
        """
        neighbors_features, neighbors_labels = self.get_neighbors(test_row)

        return self.vote(neighbors_labels, neighbors_features, test_row)

    def majority_class_vote(self, neighbors_labels, neighbors_features = None, test_row: pd.Series = None) -> int:
        """
        Majority class voting: Return the most common class among the neighbors.
        """
        return neighbors_labels.mode()[0]

    def inverse_distance_weighted(self, neighbors_labels, neighbors_features, test_row: pd.Series) -> int:
        """
        Inverse distance weighting: Weight the class labels by the inverse of their distances.
        """
        distances = [self.distance(test_row, row) for _, row in neighbors_features.iterrows()]
        weights = [1 / (d + 1e-5) for d in distances]
        class_vote = {}

        for i, weight in enumerate(weights):
            label = neighbors_labels.iloc[i]
            if label not in class_vote:
                class_vote[label] = 0
            class_vote[label] += weight

        max_vote_label = max(class_vote, key=class_vote.get)
        return max_vote_label

    def shepard_vote(self, neighbors_labels, neighbors_features, test_row: pd.Series) -> int:
        """
        Shepard's method: Use a power-based weighting.
        """
        distances = [self.distance(test_row, row) for _, row in neighbors_features.iterrows()]
        weights = [np.exp(-d) for d in distances]
        class_vote = {}

        for i, weight in enumerate(weights):
            label = neighbors_labels.iloc[i]
            if label not in class_vote:
                class_vote[label] = 0
            class_vote[label] += weight

        max_vote_label = max(class_vote, key=class_vote.get)
        return max_vote_label

    def predict(self, test_features: pd.DataFrame) -> List[Union[int, float]]:
        """
        Predict the class labels for the test set.
        """
        predictions = [self.classify(row) for _, row in test_features.iterrows()]
        return predictions

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