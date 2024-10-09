from typing import List, Union, Any
import pandas as pd
import numpy as np


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
        - weighting_method: Weighting method to apply to the distance ('equal_weight', 'OTHER1_weight', 'OTHER2_weight').
        - voting_policy: Voting technique to determine the class ('majority_class', 'inverse_distance_weighted', 'shepard').
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weighting_method = weighting_method
        self.voting_policy = voting_policy
        self.train_data = None
        self.train_labels = None

    def fit(self, train_data: pd.DataFrame, train_labels: pd.Series):
        """
        Fit the kNN model with training data and labels.
        """
        weights = self.get_weights(train_data)

        self.train_data = train_data.multiply(weights, axis=1)
        self.train_labels = train_labels

    def get_weights(self, train_data: pd.DataFrame) -> np.ndarray:
        if self.weighting_method == 'equal_weight':
            return np.ones(train_data.shape[1])
        if self.weighting_method == 'OTHER1_weight':
            return np.ones(train_data.shape[1])
        if self.weighting_method == 'OTHER2_weight':
            return np.ones(train_data.shape[1])
        else:
            raise ValueError(f"Unsupported weighting method: {self.weighting_method}")

    def get_distances(self, train_data: pd.DataFrame, test_row: pd.Series) -> list[tuple[int, float]]:
        """
        Calculate list of distances between vectors of 2 DataFrames.
        """
        if self.distance_metric == 'euclidean_distance':
            return [(int(index), self.euclidean_distance(test_row, row)) for index, row in train_data.iterrows()]
        elif self.distance_metric == 'manhattan_distance':
            return [(int(index), self.manhattan_distance(test_row, row)) for index, row in train_data.iterrows()]
        elif self.distance_metric == 'clark_distance':
            return [(int(index), self.clark_distance(test_row, row)) for index, row in train_data.iterrows()]
        else:
            raise ValueError(f"Unsupported distance function: {self.distance_metric}")

    def euclidean_distance(self, vec1: pd.Series, vec2: pd.Series) -> float:
        """
        Euclidean distance metric.
        """
        return np.sqrt(np.sum((vec1 - vec2) ** 2))

    def manhattan_distance(self, vec1: pd.Series, vec2: pd.Series) -> float:
        """
        Manhattan distance metric.
        """
        return np.sum(np.abs(vec1 - vec2))

    def clark_distance(self, vec1: pd.Series, vec2: pd.Series) -> float:
        """
        Clark distance metric.
        """
        numerator = np.abs(vec1 - vec2)
        denominator = vec1 + vec2
        squared_ratio = (numerator / denominator) ** 2

        # Handle divisions by zero (if vec1 + vec2 == 0, the result should be ignored in sum)
        squared_ratio = squared_ratio.replace([np.inf, -np.inf], np.nan).dropna()

        return np.sqrt(squared_ratio.sum())

    def get_neighbors(self, train_data: pd.DataFrame, train_labels: pd.Series, test_row: pd.Series) -> tuple[Any, Any]:
        """
        Identify the k nearest neighbors for a given test row.
        """
        distances = self.get_distances(train_data, test_row)
        sorted_distances = sorted(distances, key=lambda x: x[1])  # Sort by distance
        neighbors_idx = [index for index, _ in sorted_distances[:self.k]]  # Get the top k indices

        neighbors_data = train_data.iloc[neighbors_idx]
        neighbors_labels = train_labels.iloc[neighbors_idx]

        return neighbors_data, neighbors_labels # @todo TYPING?? Then, change it in voting policies as well

    def classify(self, train_data: pd.DataFrame, train_labels: pd.Series, test_row: pd.Series) -> Union[int, float]:
        """
        Classify a single example using k-nearest neighbors.
        """
        neighbors_data, neighbors_labels = self.get_neighbors(train_data, train_labels, test_row)

        if self.voting_policy == 'majority_class':
            return self.majority_class_vote(neighbors_labels)
        elif self.voting_policy == 'inverse_distance_weighted':
            return self.inverse_distance_weighted(neighbors_data, neighbors_labels, test_row)
        elif self.voting_policy == 'shepard':
            return self.shepard_vote(neighbors_data, neighbors_labels, test_row)
        else:
            raise ValueError(f"Unsupported voting policy: {self.voting_policy}")

    def majority_class_vote(self, neighbors_labels) -> int:
        """
        Majority class voting: Return the most common class among the neighbors.
        """
        return neighbors_labels.mode()[0]  # Get the most frequent label

    def inverse_distance_weighted(self, neighbors_data, neighbors_labels, test_row: pd.Series) -> float:
        """
        Inverse distance weighting: Weight the class labels by the inverse of their distances.
        """
        distances = self.get_distances(neighbors_data, test_row) # @todo This is already calculated. Maybe store and bring here
        # Avoid division by zero
        weights = [1 / (d + 1e-5) for _, d in distances]  # Add a small constant to avoid division by zero
        class_vote = {}

        for i, weight in enumerate(weights):
            label = neighbors_labels.iloc[i]
            if label not in class_vote:
                class_vote[label] = 0
            class_vote[label] += weight

        # Find the label with the highest weighted vote
        max_vote_label = max(class_vote, key=class_vote.get)

        return max_vote_label

    def shepard_vote(self, neighbors_data, neighbors_labels, test_row: pd.Series) -> float:
        """
        Shepard's method: Use a power-based weighting.
        """
        distances = self.get_distances(neighbors_data, test_row)
        weights = [np.exp(-d) for _, d in distances]  # Shepard's weighting (power of 2)
        class_vote = {}

        for i, weight in enumerate(weights):
            label = neighbors_labels.iloc[i]
            if label not in class_vote:
                class_vote[label] = 0
            class_vote[label] += weight

        # Find the label with the highest weighted vote
        max_vote_label = max(class_vote, key=class_vote.get)

        return max_vote_label

    def predict(self, test_data: pd.DataFrame) -> List[Union[int, float]]:
        """
        Predict the class labels for the test set.
        """
        test_data = test_data
        predictions = [self.classify(self.train_data, self.train_labels, row) for _, row in test_data.iterrows()]
        return predictions
