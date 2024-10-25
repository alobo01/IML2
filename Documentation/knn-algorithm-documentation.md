# KNNAlgorithm Class Documentation

## Table of Contents
1. [Overview](#overview)
2. [Class Definition](#class-definition)
3. [Constructor](#constructor)
4. [Methods](#methods)
   - [fit](#fit)
   - [get_distance](#get_distance)
   - [get_neighbors](#get_neighbors)
   - [classify](#classify)
   - [predict](#predict)
   - [score](#score)
5. [Static Methods](#static-methods)
   - [euclidean_distance](#euclidean_distance)
   - [manhattan_distance](#manhattan_distance)
   - [clark_distance](#clark_distance)
6. [Helper Methods](#helper-methods)
   - [majority_class_vote](#majority_class_vote)
   - [inverse_distance_weighted](#inverse_distance_weighted)
   - [shepard_vote](#shepard_vote)
7. [Usage Examples](#usage-examples)

## Overview

The `KNNAlgorithm` class implements a k-Nearest Neighbors (kNN) Classifier supporting multiple distance metrics, weighting methods, and voting techniques. It provides methods to fit the model, make predictions, and evaluate the classifier's performance.

## Class Definition

```python
class KNNAlgorithm:
    def __init__(self, k: int = 3, distance_metric: str = 'euclidean_distance', weighting_method: str = 'equal_weight', voting_policy: str = 'majority_class')
    def fit(self, train_features: pd.DataFrame, train_labels: pd.Series)
    def get_distance(self, vec1: pd.Series, vec2: pd.Series) -> float
    def get_neighbors(self, test_row: pd.Series) -> tuple[Any, Any]
    def classify(self, test_row: pd.Series) -> Union[int, float]
    def predict(self, test_features: pd.DataFrame) -> List[Union[int, float]]
    def score(self, test_features: pd.DataFrame, test_labels: pd.Series) -> float
```

## Constructor

### `__init__(self, k: int = 3, distance_metric: str = 'euclidean_distance', weighting_method: str = 'equal_weight', voting_policy: str = 'majority_class')`

Initializes the KNNAlgorithm class with the specified parameters.

#### Parameters:
- `k` (int, optional): Number of neighbors to consider. Defaults to 3.
- `distance_metric` (str, optional): Function to calculate the distance between samples. Defaults to 'euclidean_distance'.
- `weighting_method` (str, optional): Weighting method to apply to the distance. Defaults to 'equal_weight'.
- `voting_policy` (str, optional): Voting technique to determine the class. Defaults to 'majority_class'.

## Methods

### `fit`

```python
def fit(self, train_features: pd.DataFrame, train_labels: pd.Series)
```

Fits the kNN model with training data and labels.

#### Parameters:
- `train_features` (pd.DataFrame): The feature set of the training data.
- `train_labels` (pd.Series): The labels of the training data.

---

### `get_distance`

```python
def get_distance(self, vec1: pd.Series, vec2: pd.Series) -> float
```

Computes the distance between two vectors based on the selected metric.

#### Parameters:
- `vec1` (pd.Series): The first vector.
- `vec2` (pd.Series): The second vector.

#### Returns:
- float: The distance between the two vectors.

#### Raises:
- ValueError: If an unsupported distance function is specified.

---

### `get_neighbors`

```python
def get_neighbors(self, test_row: pd.Series) -> tuple[Any, Any]
```

Identifies the k nearest neighbors for a given test row.

#### Parameters:
- `test_row` (pd.Series): The test sample to find neighbors for.

#### Returns:
- tuple: A tuple containing the features and labels of the k nearest neighbors.

---

### `classify`

```python
def classify(self, test_row: pd.Series) -> Union[int, float]
```

Classifies a single example using k-nearest neighbors.

#### Parameters:
- `test_row` (pd.Series): The test sample to classify.

#### Returns:
- Union[int, float]: The predicted class label.

#### Raises:
- ValueError: If an unsupported voting policy is specified.

---

### `predict`

```python
def predict(self, test_features: pd.DataFrame) -> List[Union[int, float]]
```

Predicts the class labels for the test set.

#### Parameters:
- `test_features` (pd.DataFrame): The feature set of the test data.

#### Returns:
- List[Union[int, float]]: A list of predicted class labels.

---

### `score`

```python
def score(self, test_features: pd.DataFrame, test_labels: pd.Series) -> float
```

Calculates the accuracy of the kNN classifier on the test data.

#### Parameters:
- `test_features` (pd.DataFrame): The feature set of the test data.
- `test_labels` (pd.Series): The true labels of the test data.

#### Returns:
- float: The accuracy of the classifier.

## Static Methods

### `euclidean_distance`

```python
@staticmethod
def euclidean_distance(vec1: pd.Series, vec2: pd.Series) -> float
```

Calculates the Euclidean distance between two vectors.

---

### `manhattan_distance`

```python
@staticmethod
def manhattan_distance(vec1: pd.Series, vec2: pd.Series) -> float
```

Calculates the Manhattan distance between two vectors.

---

### `clark_distance`

```python
@staticmethod
def clark_distance(vec1: pd.Series, vec2: pd.Series) -> float
```

Calculates the Clark distance between two vectors.

## Helper Methods

### `majority_class_vote`

```python
def majority_class_vote(self, neighbors_labels) -> int
```

Implements majority class voting to determine the class label.

---

### `inverse_distance_weighted`

```python
def inverse_distance_weighted(self, neighbors_features, neighbors_labels, test_row: pd.Series) -> float
```

Implements inverse distance weighting to determine the class label.

---

### `shepard_vote`

```python
def shepard_vote(self, neighbors_features, neighbors_labels, test_row: pd.Series) -> float
```

Implements Shepard's method to determine the class label.

## Usage Examples

```python
import pandas as pd
from KNNAlgorithm import KNNAlgorithm

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Split into features and labels
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

# Split into training and test sets
train_features = features.sample(frac=0.8, random_state=42)
train_labels = labels.loc[train_features.index]
test_features = features.drop(train_features.index)
test_labels = labels.drop(train_features.index)

# Initialize KNN classifier
knn = KNNAlgorithm(k=5, distance_metric='euclidean_distance', voting_policy='majority_class')

# Fit the model
knn.fit(train_features, train_labels)

# Make predictions
predictions = knn.predict(test_features)

# Evaluate the model
accuracy = knn.score(test_features, test_labels)

print(f"KNN Accuracy: {accuracy}")
```