from pydantic import BaseModel, validator
from typing import Callable, List, Union
import pandas as pd
import numpy as np


class KNNAlgorithm:
    """
    A k-Nearest Neighbors (kNN) Classifier supporting multiple voting techniques.
    """
    def __init__(self,
                 k: int = 3,
                 distance_function: Callable[[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]], float] = None,
                 voting_policy: str = 'majority_class'):
        """
        Initialize the kNNAlgorithm with:
        - k: Number of neighbors to consider.
        - distance_function: A function to calculate the distance between two vectors.
        - voting_policy: Voting technique to determine the class ('majority_class', 'inverse_distance_weighting', 'shepard').
        """
        self.k = k
        self.distance_function = distance_function if distance_function else self.euclidean_distance
        self.voting_policy = voting_policy

    def euclidean_distance(self, vec1: Union[pd.Series, np.ndarray], vec2: Union[pd.Series, np.ndarray]) -> float:
        """
        Default Euclidean distance function.
        """
        return np.sqrt(np.sum((vec1 - vec2) ** 2))

    def get_neighbors(self, train_data: pd.DataFrame, test_row: pd.Series) -> List[int]:
        """
        Identify the k nearest neighbors for a given test row.
        """
        distances = [(index, self.distance_function(test_row, row)) for index, row in train_data.iterrows()]
        sorted_distances = sorted(distances, key=lambda x: x[1])  # Sort by distance
        neighbors = [index for index, _ in sorted_distances[:self.k]]  # Get the top k indices
        return neighbors

    def classify(self, train_data: pd.DataFrame, train_labels: pd.Series, test_row: pd.Series) -> Union[int, float]:
        """
        Classify a single example using k-nearest neighbors.
        """
        neighbors = self.get_neighbors(train_data, test_row)
        if self.voting_policy == 'majority_class':
            return self.majority_class_vote(train_labels, neighbors)
        elif self.voting_policy == 'inverse_distance_weighting':
            return self.inverse_distance_weighting(train_data, train_labels, test_row, neighbors)
        elif self.voting_policy == 'shepard':
            return self.shepard_vote(train_data, train_labels, test_row, neighbors)
        else:
            raise ValueError(f"Unsupported voting policy: {self.voting_policy}")

    def majority_class_vote(self, train_labels: pd.Series, neighbors: List[int]) -> int:
        """
        Majority class voting: Return the most common class among the neighbors.
        """
        neighbor_labels = train_labels.iloc[neighbors]
        return neighbor_labels.mode()[0]  # Get the most frequent label

    def inverse_distance_weighting(self, train_data: pd.DataFrame, train_labels: pd.Series, test_row: pd.Series, neighbors: List[int]) -> float:
        """
        Inverse distance weighting: Weight the class labels by the inverse of their distances.
        """
        distances = [self.distance_function(train_data.iloc[neighbor], test_row) for neighbor in neighbors]
        # Avoid division by zero
        weights = [1 / (d + 1e-5) for d in distances]  # Add a small constant to avoid division by zero
        weighted_labels = train_labels.iloc[neighbors] * weights
        return weighted_labels.sum() / sum(weights)

    def shepard_vote(self, train_data: pd.DataFrame, train_labels: pd.Series, test_row: pd.Series, neighbors: List[int]) -> float:
        """
        Shepard's method: Use a power-based weighting.
        """
        distances = [self.distance_function(train_data.iloc[neighbor], test_row) for neighbor in neighbors]
        weights = [(1 / (d + 1e-5)) ** 2 for d in distances]  # Shepard's weighting (power of 2)
        weighted_labels = train_labels.iloc[neighbors] * weights
        return weighted_labels.sum() / sum(weights)

    def fit(self, train_data: pd.DataFrame, train_labels: pd.Series):
        """
        Fit the kNN model with training data and labels.
        """
        self.train_data = KNNData(data=train_data).data
        self.train_labels = train_labels

    def predict(self, test_data: pd.DataFrame) -> List[Union[int, float]]:
        """
        Predict the class labels for the test set.
        """
        test_data = KNNData(data=test_data).data
        predictions = [self.classify(self.train_data, self.train_labels, row) for _, row in test_data.iterrows()]
        return predictions
