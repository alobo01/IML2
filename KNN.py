from pydantic import BaseModel, validator
from typing import List, Union, Any
import pandas as pd
import numpy as np


class KNNData(BaseModel):
    """
    Data model using pydantic to validate input dataframes.
    Expects a pandas DataFrame with numerical values.
    """
    data: pd.DataFrame

    @validator('data')
    def check_dataframe_numeric(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if not all([np.issubdtype(dt, np.number) for dt in v.dtypes]):
            raise ValueError("All columns in the DataFrame must be numerical")
        return v


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
        - distance_metric: A function to calculate the distance between samples ('euclidean_distance', 'manhattan_distance', 'OTHER_distance').
        - weighting_method: Weighting method to apply to the distance ('equal_weight', 'OTHER1_weight', 'OTHER2_weight').
        - voting_policy: Voting technique to determine the class ('majority_class', 'inverse_distance_weighting', 'shepard').
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weighting_method = weighting_method
        self.voting_policy = voting_policy

    def get_weights(self, vec: pd.Series) -> np.ndarray:
        if self.weighting_method == 'equal_weight':
            return np.ones(vec.size)
        if self.weighting_method == 'OTHER1_weight':
            return np.ones(vec.size)
        if self.weighting_method == 'OTHER2_weight':
            return np.ones(vec.size)
        else:
            raise ValueError(f"Unsupported weighting method: {self.weighting_method}")

    def euclidean_distance(self, vec1: Union[pd.Series, np.ndarray], vec2: Union[pd.Series, np.ndarray], weights) -> float:
        """
        Euclidean distance metric.
        """
        return np.sqrt(np.sum(weights * (vec1 - vec2) ** 2))

    def manhattan_distance(self, vec1: Union[pd.Series, np.ndarray], vec2: Union[pd.Series, np.ndarray], weights) -> float:
        """
        Manhattan distance metric.
        """
        return np.sum(weights * np.abs(vec1 - vec2))

    def OTHER_distance(self, vec1: Union[pd.Series, np.ndarray], vec2: Union[pd.Series, np.ndarray], weights) -> float:
        """
        OTHER distance metric.
        """
        return np.sqrt(np.sum(weights * (vec1 - vec2) ** 2))

    def get_distances(self, train_data: pd.DataFrame, test_row: pd.Series) -> list[tuple[int, float]]:
        """
        Calculate list of distances between vectors of 2 DataFrames.
        """
        weights = self.get_weights(test_row)

        if self.distance_metric == 'euclidean_distance':
            return [(int(index), self.euclidean_distance(test_row, row, weights)) for index, row in train_data.iterrows()]
        elif self.voting_policy == 'manhattan_distance':
            return [(int(index), self.manhattan_distance(test_row, row, weights)) for index, row in train_data.iterrows()]
        elif self.voting_policy == 'OTHER_distance':
            return [(int(index), self.OTHER_distance(test_row, row, weights)) for index, row in train_data.iterrows()]
        else:
            raise ValueError(f"Unsupported distance function: {self.distance_metric}")

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
        elif self.voting_policy == 'inverse_distance_weighting':
            return self.inverse_distance_weighting(neighbors_data, neighbors_labels, test_row)
        elif self.voting_policy == 'shepard':
            return self.shepard_vote(neighbors_data, neighbors_labels, test_row)
        else:
            raise ValueError(f"Unsupported voting policy: {self.voting_policy}")

    def majority_class_vote(self, neighbors_labels) -> int:
        """
        Majority class voting: Return the most common class among the neighbors.
        """
        return neighbors_labels.mode()[0]  # Get the most frequent label

    def inverse_distance_weighting(self, neighbors_data, neighbors_labels, test_row: pd.Series) -> float:
        """
        Inverse distance weighting: Weight the class labels by the inverse of their distances.
        """
        distances = self.get_distances(neighbors_data, test_row)
        # Avoid division by zero
        weights = [1 / (d + 1e-5) for _, d in distances]  # Add a small constant to avoid division by zero
        weighted_labels = neighbors_labels * weights
        return weighted_labels.sum() / sum(weights)

    def shepard_vote(self, neighbors_data, neighbors_labels, test_row: pd.Series) -> float:
        """
        Shepard's method: Use a power-based weighting.
        """
        distances = self.get_distances(neighbors_data, test_row)
        weights = [(1 / (d + 1e-5)) ** 2 for _, d in distances]  # Shepard's weighting (power of 2)
        weighted_labels = neighbors_labels * weights
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
