import numpy as np
import random
from KNN import KNNAlgorithm
from pandas import DataFrame

def distance_matrix(X, Y):
    return np.sqrt(((X[:, np.newaxis] - Y) ** 2).sum(axis=2))

class ReductionKNN:
    """
    Class for applying different reduction techniques to a dataset before 
    performing K-Nearest Neighbor (KNN) classification.

    Attributes:
        originalKNN: The original KNN classifier.
        reducedKNN: The reduced KNN classifier after applying a reduction method.
    """

    def __init__(self, bestKNN: KNNAlgorithm):
        """
        Initializes the ReductionKNN class with the original KNN classifier.

        Args:
            bestKNN (KNNAlgorithm): The KNN classifier object.
        """
        self.originalKNN = bestKNN
        self.reducedKNN = None

    def apply_reduction(self, data: DataFrame, reductionMethod: str):
        """
        Applies the selected reduction method to the dataset and returns the reduced dataset.

        Args:
            data (DataFrame): The dataset to apply the reduction method to.
            reductionMethod (str): The name of the reduction method ("GCNN", "RENN", or "IB2").

        Returns:
            DataFrame: The reduced dataset.
        """
        features = data.iloc[:, :-1]
        labels = data.iloc[:, -1]

        if reductionMethod == "GCNN":
            reduced_indices = self.generalized_condensed_nearest_neighbor(features, labels)
        elif reductionMethod == "RENN":
            reduced_indices = self.repeated_edited_nearest_neighbor(features, labels)
        elif reductionMethod == "IB2":
            reduced_indices = self.ib2(features, labels)
        else:
            raise ValueError(f"Reduction method {reductionMethod} not recognized.")

        reduced_data = data.iloc[reduced_indices]
        self.reducedKNN = self.originalKNN
        self.reducedKNN.fit(reduced_data.iloc[:, :-1], reduced_data.iloc[:, -1])

        return reduced_data

    def generalized_condensed_nearest_neighbor(self, features: DataFrame, labels: DataFrame, rho=0.5):
        """
        The Generalized Condensed Nearest Neighbor (GCNN) algorithm reduces the dataset by absorbing samples
        based on a generalized absorption criterion.

        Citation:
        Nikolaidis, K., Rodriguez, J. J., & Goulermas, J. Y. (2011). Generalized Condensed Nearest 
        Neighbor for Prototype Reduction. In 2011 IEEE International Conference on Systems, Man, 
        and Cybernetics (pp. 2885-2890). IEEE.

        Args:
            features (DataFrame): The feature set of the dataset.
            labels (DataFrame): The labels of the dataset.
            rho (float): The absorption threshold, ρ ∈ [0, 1].

        Returns:
            list: Indices of the points in the reduced dataset.
        """
        features_array = np.array(features)
        labels_array = np.array(labels)

        # Initialize prototypes for each class
        prototypes = []
        prototype_labels = []
        prototype_indices = []

        unique_labels = np.unique(labels_array)
        for lbl in unique_labels:
            class_indices = np.where(labels_array == lbl)[0]
            rand_index = random.choice(class_indices)
            prototypes.append(features_array[rand_index])
            prototype_labels.append(labels_array[rand_index])
            prototype_indices.append(rand_index)

        prototypes = np.array(prototypes)
        prototype_labels = np.array(prototype_labels)

        # Compute δn: minimum distance between points of different classes
        dist_matrix = distance_matrix(features_array, features_array)
        delta_n = np.min([dist_matrix[i, j] for i in range(len(features_array)) for j in range(len(features_array)) if
                          labels_array[i] != labels_array[j]])

        # Iteratively absorb points
        absorbed = np.full(len(features_array), False)
        absorbed[prototype_indices] = True

        while not np.all(absorbed):
            for i in range(len(features_array)):
                if not absorbed[i]:
                    p_idx = np.argmin(
                        [np.linalg.norm(features_array[i] - proto) for proto, lbl in zip(prototypes, prototype_labels)
                         if lbl == labels_array[i]])
                    q_idx = np.argmin(
                        [np.linalg.norm(features_array[i] - proto) for proto, lbl in zip(prototypes, prototype_labels)
                         if lbl != labels_array[i]])

                    p = prototypes[p_idx]
                    q = prototypes[q_idx]

                    # Absorption criteria
                    if np.linalg.norm(features_array[i] - q) - np.linalg.norm(features_array[i] - p) > rho * delta_n:
                        prototypes = np.vstack([prototypes, features_array[i]])
                        prototype_labels = np.append(prototype_labels, labels_array[i])
                        prototype_indices.append(i)
                    absorbed[i] = True

        return prototype_indices

    def repeated_edited_nearest_neighbor(self, features: DataFrame, labels: DataFrame, k=3):
        """
        Repeated Edited Nearest Neighbor (RENN) applies the ENN algorithm iteratively until all instances 
        remaining have a majority of their k nearest neighbors with the same class.

        Citation:
        Wilson, D. L. (1972). Asymptotic Properties of Nearest Neighbor Rules Using Edited Data. 
        IEEE Transactions on Systems, Man, and Cybernetics, SMC-2(3), 408-421.
        As cited in: Wilson, D. R., & Martinez, T. R. (2000). Reduction techniques for instance-based learning algorithms. 
        Machine learning, 38(3), 257-286.

        Args:
            features (DataFrame): The feature set of the dataset.
            labels (DataFrame): The labels of the dataset.
            k (int): Number of nearest neighbors to check (default = 3).

        Returns:
            list: Indices of the points in the reduced dataset.
        """
        features_array = np.array(features)
        labels_array = np.array(labels)

        absorbed = np.full(len(features_array), True)
        changed = True

        while changed:
            changed = False
            knn = self.originalKNN
            knn.fit(features_array[absorbed], labels_array[absorbed])

            for i in range(len(features_array)):
                if absorbed[i]:
                    neighbors = knn.kneighbors([features_array[i]], n_neighbors=k, return_distance=False)
                    neighbor_labels = labels_array[absorbed][neighbors[0]]

                    # If majority of k neighbors disagree, remove the point
                    if np.sum(neighbor_labels == labels_array[i]) <= k // 2:
                        absorbed[i] = False
                        changed = True

        return np.where(absorbed)[0].tolist()

    def ib2(self, features: DataFrame, labels: DataFrame):
        """
        The IB2 algorithm is incremental: it starts with S initially empty, and each 
        instance in T is added to S if it is not classified correctly by the instances already in S.

        Citation:
        Aha, D. W., Kibler, D., & Albert, M. K. (1991). Instance-based learning algorithms. 
        Machine learning, 6(1), 37-66.
        As cited in: Wilson, D. R., & Martinez, T. R. (2000). Reduction techniques for instance-based learning algorithms. 
        Machine learning, 38(3), 257-286.

        Args:
            features (DataFrame): The feature set of the dataset.
            labels (DataFrame): The labels of the dataset.

        Returns:
            list: Indices of the points in the reduced dataset.
        """
        features_array = np.array(features)
        labels_array = np.array(labels)

        # Initialize a list of indices for the dataset and shuffle them
        indices = list(range(len(features_array)))
        random.shuffle(indices)

        # Initialize the condensed set with the first random point
        first_index = indices.pop(0)
        condensed_indices = [first_index]

        # Loop through the shuffled dataset
        for i in indices:
            knn = self.originalKNN
            knn.fit(features_array[condensed_indices], labels_array[condensed_indices])

            # Predict the label of the current point using the reduced set
            prediction = knn.predict([features_array[i]])

            # If the prediction is incorrect, add the point to the condensed set
            if prediction != labels_array[i]:
                condensed_indices.append(i)

        return condensed_indices

    def evaluate(self, test_data: DataFrame):
        """
        Evaluates the performance of both the original and reduced KNN classifiers.

        Args:
            test_data (DataFrame): The test dataset to evaluate on.

        Returns:
            dict: A dictionary containing accuracy of the original and reduced KNNs.
        """
        test_features = test_data.iloc[:, :-1]
        test_labels = test_data.iloc[:, -1]

        # Original KNN
        original_accuracy = self.originalKNN.score(test_features, test_labels)

        # Reduced KNN
        reduced_accuracy = self.reducedKNN.score(test_features, test_labels) if self.reducedKNN else None

        return {
            "original_accuracy": original_accuracy,
            "reduced_accuracy": reduced_accuracy
        }


