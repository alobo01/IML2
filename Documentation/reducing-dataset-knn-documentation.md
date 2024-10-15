# ReductionKNN Class Documentation

## Table of Contents
1. [Overview](#overview)
2. [Class Definition](#class-definition)
3. [Constructor](#constructor)
4. [Methods](#methods)
   - [apply_reduction](#apply_reduction)
   - [generalized_condensed_nearest_neighbor](#generalized_condensed_nearest_neighbor)
   - [repeated_edited_nearest_neighbor](#repeated_edited_nearest_neighbor)
   - [ib2](#ib2)
   - [evaluate](#evaluate)
5. [Usage Examples](#usage-examples)
6. [References](#references)

## Overview

The `ReductionKNN` class implements various data reduction techniques for K-Nearest Neighbor (KNN) classification. It provides methods to apply different reduction algorithms to a dataset before performing KNN classification, potentially improving efficiency and generalization.

The class supports three reduction methods:
1. Generalized Condensed Nearest Neighbor (GCNN)
2. Repeated Edited Nearest Neighbor (RENN)
3. Instance-Based Learning 2 (IB2)

## Class Definition

```python
class ReductionKNN:
    def __init__(self, original: KNNAlgorithm, reduced: KNNAlgorithm)
    def apply_reduction(self, data: DataFrame, reductionMethod: str) -> DataFrame
    def generalized_condensed_nearest_neighbor(self, features: DataFrame, labels: DataFrame, rho=0.5) -> list
    def repeated_edited_nearest_neighbor(self, features: DataFrame, labels: DataFrame, k=3) -> list
    def ib2(self, features: DataFrame, labels: DataFrame) -> list
    def evaluate(self, test_data: DataFrame) -> dict
```

## Constructor

### `__init__(self, original: KNNAlgorithm, reduced: KNNAlgorithm)`

Initializes the ReductionKNN class with the original and reduced KNN classifiers.

#### Parameters:
- `original` (KNNAlgorithm): The original KNN classifier object.
- `reduced` (KNNAlgorithm): The KNN classifier object to be used with the reduced dataset.

## Methods

### `apply_reduction`

```python
def apply_reduction(self, data: DataFrame, reductionMethod: str) -> DataFrame
```

Applies the selected reduction method to the dataset and returns the reduced dataset.

#### Parameters:
- `data` (DataFrame): The dataset to apply the reduction method to.
- `reductionMethod` (str): The name of the reduction method ("GCNN", "RENN", or "IB2").

#### Returns:
- DataFrame: The reduced dataset.

#### Raises:
- ValueError: If the reduction method is not recognized.

---

### `generalized_condensed_nearest_neighbor`

```python
def generalized_condensed_nearest_neighbor(self, features: DataFrame, labels: DataFrame, rho=0.5) -> list
```

Implements the Generalized Condensed Nearest Neighbor (GCNN) algorithm to reduce the dataset.

#### Parameters:
- `features` (DataFrame): The feature set of the dataset.
- `labels` (DataFrame): The labels of the dataset.
- `rho` (float, optional): The absorption threshold, ρ ∈ [0, 1). Defaults to 0.5.

#### Returns:
- list: Indices of the points in the reduced dataset.

---

### `repeated_edited_nearest_neighbor`

```python
def repeated_edited_nearest_neighbor(self, features: DataFrame, labels: DataFrame, k=3) -> list
```

Implements the Repeated Edited Nearest Neighbor (RENN) algorithm to reduce the dataset.

#### Parameters:
- `features` (DataFrame): The feature set of the dataset.
- `labels` (DataFrame): The labels of the dataset.
- `k` (int, optional): Number of nearest neighbors to check. Defaults to 3.

#### Returns:
- list: Indices of the points in the reduced dataset.

---

### `ib2`

```python
def ib2(self, features: DataFrame, labels: DataFrame) -> list
```

Implements the Instance-Based Learning 2 (IB2) algorithm to reduce the dataset.

#### Parameters:
- `features` (DataFrame): The feature set of the dataset.
- `labels` (DataFrame): The labels of the dataset.

#### Returns:
- list: Indices of the points in the reduced dataset.

---

### `evaluate`

```python
def evaluate(self, test_data: DataFrame) -> dict
```

Evaluates the performance of both the original and reduced KNN classifiers.

#### Parameters:
- `test_data` (DataFrame): The test dataset to evaluate on.

#### Returns:
- dict: A dictionary containing accuracy of the original and reduced KNNs.

## Usage Examples

```python
import pandas as pd
from KNN import KNNAlgorithm
from ReductionKNN import ReductionKNN

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Split into training and test sets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Initialize KNN classifiers
original_knn = KNNAlgorithm(k=5)
reduced_knn = KNNAlgorithm(k=5)

# Initialize ReductionKNN
reduction_knn = ReductionKNN(original_knn, reduced_knn)

# Apply GCNN reduction
reduced_data = reduction_knn.apply_reduction(train_data, "GCNN")

# Train the original KNN on the full dataset
original_knn.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])

# Evaluate both classifiers
results = reduction_knn.evaluate(test_data)

print(f"Original KNN Accuracy: {results['original_accuracy']}")
print(f"Reduced KNN Accuracy: {results['reduced_accuracy']}")
```

## References

1. Nikolaidis, K., Rodriguez, J. J., & Goulermas, J. Y. (2011). Generalized Condensed Nearest Neighbor for Prototype Reduction. In 2011 IEEE International Conference on Systems, Man, and Cybernetics (pp. 2885-2890). IEEE.

2. Wilson, D. L. (1972). Asymptotic Properties of Nearest Neighbor Rules Using Edited Data. IEEE Transactions on Systems, Man, and Cybernetics, SMC-2(3), 408-421.

3. Aha, D. W., Kibler, D., & Albert, M. K. (1991). Instance-based learning algorithms. Machine learning, 6(1), 37-66.

4. Wilson, D. R., & Martinez, T. R. (2000). Reduction techniques for instance-based learning algorithms. Machine learning, 38(3), 257-286.