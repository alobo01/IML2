import numpy as np
import random
from collections import Counter

from sklearn.neighbors import NearestNeighbors

import pandas as pd

from classes.KNN import KNNAlgorithm
from pandas import DataFrame

def distance_matrix(X, Y):
    return np.sqrt(((X[:, np.newaxis] - Y) ** 2).sum(axis=2))

class ReductionKNN:
    """
    Class for applying different reduction techniques to a dataset before 
    performing K-Nearest Neighbor (KNN) classification.

    Summary of types:

    A first category of techniques try to eliminate from the TS prototypes erroneously labeled, commonly
    outliers, and at the same time, to “clean” the possible overlapping between regions of
    different classes. These techniques are referred in the literature to as Editing, and the
    resulting classification rule is known as Edited NN rule

    A second group of PS techniques are aimed at selecting a certain subgroup of prototypes that behaves,
    employing the 1-NN rule, in a similar way to the one obtained by using the totality of the TS.
    This group of techniques are the so called Condensing algorithms and its corresponding Condensed NN rule




    Generalized Condensed Nearest Neighbor (GCNN) Implementation:
    The GCNN algorithm reduces the dataset using a generalized criterion to absorb
    samples into the prototype set, based on a threshold defined by the nearest homogeneous
    and heterogeneous samples.

    Citation:
    https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=d5cf73f3d4d165f3aa51f1492caefc1d0a52a743

    Repeated Edited Nearest Neighbor (RENN) Implementation:
    The RENN algorithm applies the ENN algorithm iteratively until all instances remaining
    have a majority of their neighbors with the same class, smoothing decision boundaries.

    Citation:
    https://link.springer.com/content/pdf/10.1023/A:1007626913721.pdf: Wilson (1972) developed the Edited Nearest Neighbor (ENN) algorithm in which S starts out the same as T, and then each instance in S is removed if it does not agree with the majority of its k nearest neighbors (with k = 3, typically).

    Instance-Based Learning 2 (IB2) Implementation:
    The IB2 algorithm retains border points while eliminating internal points that are surrounded by members of the same class, based on an incremental absorption criterion.

    Citation:
    https://link.springer.com/content/pdf/10.1023/A:1007626913721.pdf

    Attributes:
        originalKNN: The original KNN classifier.
        reducedKNN: The reduced KNN classifier after applying a reduction method.
    """

    def __init__(self, original: KNNAlgorithm, reduced: KNNAlgorithm):
        """
        Initializes the ReductionKNN class with the original KNN classifier.

        Args:
            original (KNNAlgorithm): The KNN classifier object.
            reduced (KNNAlgorithm): The KNN classifier object to reduct.
        """
        self.originalKNN = original
        self.reducedKNN = reduced

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
        elif reductionMethod == "EENTH":
            reduced_indices = self.editing_algorithm_estimating_class_probabilities_and_threshold(features, labels)
        elif reductionMethod == "IB2":
            reduced_indices = self.ib2(features, labels)
        elif reductionMethod == "DROP3":
            reduced_indices = self.drop3(features, labels)
        elif reductionMethod == "FENN":
            reduced_indices = self.fast_enn(features, labels)
        else:
            raise ValueError(f"Reduction method {reductionMethod} not recognized.")

        reduced_data = data.loc[reduced_indices]

        self.reducedKNN.fit(reduced_data.iloc[:, :-1], reduced_data.iloc[:, -1])

        return reduced_data

    def generalized_condensed_nearest_neighbor(self, features: DataFrame, labels: DataFrame, rho=0.5):
        """
        The Generalized Condensed Nearest Neighbor (GCNN) algorithm reduces the dataset by absorbing samples
        based on a generalized absorption criterion. CNN is when rho=0

        Citation:
        Nikolaidis, K., Rodriguez, J. J., & Goulermas, J. Y. (2011). Generalized Condensed Nearest 
        Neighbor for Prototype Reduction. In 2011 IEEE International Conference on Systems, Man, 
        and Cybernetics (pp. 2885-2890). IEEE.

        Args:
            features (DataFrame): The feature set of the dataset.
            labels (DataFrame): The labels of the dataset.
            rho (float): The absorption threshold, ρ ∈ [0, 1).

        Returns:
            list: Indices of the points in the reduced dataset.
        """
        # Initialize prototypes for each class
        prototypes = []
        prototype_labels = []
        prototype_indices = []

        unique_labels = labels.unique()
        for lbl in unique_labels:
            class_indices = labels[labels == lbl].index
            rand_index = np.random.choice(class_indices)
            prototypes.append(features.loc[rand_index])
            prototype_labels.append(lbl)
            prototype_indices.append(rand_index)

        prototypes = pd.DataFrame(prototypes)
        prototype_labels = pd.Series(prototype_labels)

        # Compute δn: minimum distance between points of different classes
        dist_matrix = pd.DataFrame(np.linalg.norm(features.values[:, np.newaxis] - features.values, axis=2))
        delta_n = dist_matrix[labels.values[:, np.newaxis] != labels.values].min().min()

        # Iteratively absorb points
        absorbed = pd.Series(False, index=features.index)
        absorbed[prototype_indices] = True

        while not absorbed.all():
            for i in features.index:
                if not absorbed[i]:
                    p_idx = (prototype_labels == labels[i]).idxmin()  # Get index of closest prototype in same class
                    q_idx = (prototype_labels != labels[i]).idxmin()  # Get index of closest prototype in different class

                    p = prototypes.iloc[p_idx]
                    q = prototypes.iloc[q_idx]

                    # Absorption criteria
                    if np.linalg.norm(features.loc[i] - q) - np.linalg.norm(features.loc[i] - p) > rho * delta_n:
                        prototypes = pd.concat([prototypes,features.loc[i]], ignore_index=True)
                        prototype_labels = pd.concat([prototype_labels,pd.Series([labels[i]])], ignore_index=True)
                        prototype_indices.append(i)
                    absorbed[i] = True

        return prototype_indices

    def DROP1(self, features: pd.DataFrame, labels: pd.DataFrame, k=3):
        points = features.copy()
        labels_copy = labels.copy()
        points.sort_index(inplace=True)
        labels_copy.sort_index(inplace=True)
        p = points.values.copy()
        l = labels_copy.values.copy()
        old_indices = points.index.to_list()
        knn = NearestNeighbors(n_neighbors=k+1)
        knn = KNNAlgorithm()
        knn.fit(points, labels)

        discarded_points = set()

        def get_no_of_correct_classes(list_of_neighbors):
            correctly_classified_points = 0
            for point in list_of_neighbors:
                if point in discarded_points:
                    continue
                list_excluding_itself = [x for x in list_of_neighbors if x != point or x in discarded_points]
                predicted = Counter(l[n] for n in list_excluding_itself).most_common(1)[0][0]
                if predicted == l[point]:
                    correctly_classified_points += 1
            return correctly_classified_points

        def get_neighbors(index):
            n_features, _ = knn.get_neighbors(points.iloc[[index]])
            return set(n_features.index.to_list())

        all_neighbors = [get_neighbors(row) for row in range(len(p))]

        point_deleted = True
        while point_deleted:
            point_deleted = False
            for i, row in enumerate(points.values):
                if i in discarded_points:
                    continue
                neighbors = all_neighbors[i]
                if neighbors.intersection(discarded_points): # if neighbors contain a discarded point
                    updated_old_indices = [index for index in old_indices if index]
                    knn.fit(points.loc[updated_old_indices], labels.loc[updated_old_indices])
                    all_neighbors[i] = get_neighbors(i) # re-compute the neighbors
                    neighbors = all_neighbors[i]
                neighbors_including_itself = neighbors.copy()
                neighbors_including_itself.add(i)
                no_with = get_no_of_correct_classes(neighbors_including_itself)
                no_without = get_no_of_correct_classes(neighbors)
                if no_without >= no_with:
                    discarded_points.add(i)
                    old_indices[i] = None
                    point_deleted = True

        return [index for index in old_indices if index]

    def drop3(self, features: pd.DataFrame, labels: pd.Series, n_neighbors: int = 3) -> list:
        """
        DROP3 algorithm for selecting points to keep based on majority voting of their neighbors' neighbors.

        Parameters:
        - features: pd.DataFrame, the feature vectors of the dataset.
        - labels: pd.Series, the corresponding labels for the dataset.
        - n_neighbors: int, the number of neighbors to consider (default is 3).

        Returns:
        - list: The indices of the points in the original dataset that should be kept.
        """

        # First step, apply a single ENN pass to remove noisy points
        enn_removed_indices = self.fast_enn(features, labels)
        features.drop(index=enn_removed_indices)
        labels.drop(index=enn_removed_indices)

        # Reset indices for features and labels to create a continuous index
        original_index = features.index
        features.reset_index(drop=True, inplace=True)
        labels.reset_index(drop=True, inplace=True)

        # Convert to numpy arrays for faster processing
        X = features.values
        y = labels.values.ravel() if labels.values.ndim > 1 else labels.values

        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # Calculate average distance to enemy points
        enemy_distances = []
        for i in range(len(X)):
            current_class = y[i]
            enemy_indices = [idx for idx in indices[i][1:] if y[idx] != current_class]

            if enemy_indices:
                avg_enemy_dist = np.mean([distances[i][indices[i].tolist().index(idx)] for idx in enemy_indices])
            else:
                avg_enemy_dist = float('inf')

            enemy_distances.append(avg_enemy_dist)

        sort_indices = np.argsort(enemy_distances)
        sorted_features = pd.DataFrame(X[sort_indices], columns=features.columns)
        sorted_labels = pd.Series(y[sort_indices])

        # Create a dictionary to map each row index to its nearest neighbors
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 to include the point itself
        knn.fit(sorted_features)

        # Dictionary of neighbors: {index of the point: indices of its neighbors}
        neighbors = dict()
        for row_index, row_value in zip(sorted_features.index.values, sorted_features.values):
            neighbors[row_index] = knn.kneighbors(row_value.reshape(1, -1))[1][0]  # Neighbors for the current point

        removed_indices = []
        remaining_indices = sorted_features.index.to_list()

        while True:
            to_remove = []

            # Iterate through remaining points
            for point_idx in remaining_indices:
                point_label = sorted_labels.loc[point_idx]
                point_neighbors = neighbors[point_idx][1:]  # Exclude the point itself

                count_with = 0
                count_without = 0
                # Check each neighbor's neighbors (the neighbors of the current point)
                for neighbor_idx in point_neighbors:
                    neighbor_neighbors = neighbors[neighbor_idx][1:]  # Exclude itself

                    # Get the labels of the neighbor's neighbors
                    neighbors_neighbor_labels = sorted_labels.loc[neighbor_neighbors].values

                    # Check the majority label among the neighbor's neighbors, without including the original point
                    without_point = np.argmax(np.bincount(neighbors_neighbor_labels))

                    # Check the majority label when including the original point's label
                    with_point = np.argmax(np.bincount(np.append(neighbors_neighbor_labels, point_label)))

                    # If the class of the neighbor is classified correctly without the point
                    if without_point == sorted_labels.loc[neighbor_idx]:
                        count_without += 1
                    # If the class of the neighbor is classified correctly with the point
                    if with_point == sorted_labels.loc[neighbor_idx]:
                        count_with += 1

                if count_with <= count_without:
                    to_remove.append(point_idx)

            if not to_remove:
                break

            removed_indices.extend(to_remove)
            remaining_indices = [idx for idx in remaining_indices if idx not in to_remove]

        # Return the original indices of the points to keep
        return original_index[remaining_indices].tolist()

    def fast_enn(self, features: DataFrame, labels: DataFrame, k=3):
        absorbed = pd.Series(True, index=features.index)

        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(features[absorbed], labels[absorbed])

        for i in features.index[absorbed]:
            _, neighbors = knn.kneighbors(features.loc[i].values.reshape(1, -1))
            neighbor_labels = labels.values[neighbors[0][1:]]  # excluding the point itself
            # If majority of k neighbors disagree, remove the point
            if np.argmax(np.bincount(neighbor_labels)) != labels[i]:
                absorbed[i] = False

        return absorbed[absorbed].index.tolist()

    def repeated_edited_nearest_neighbor(self, features: DataFrame, labels: DataFrame, k=3):
        """
        Repeated Edited Nearest Neighbor (RENN) applies the ENN algorithm iteratively until all instances 
        remaining have a majority of their k nearest neighbors with the same class.
        ENN, typically with k=3 we will remove a point x_0 if all neighbours x_i in NN(x_0,k) don't have consensus
        on the elected class

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
        absorbed = pd.Series(True, index=features.index)
        changed = True

        while changed:
            changed = False
            knn = self.reducedKNN
            knn.fit(features[absorbed], labels[absorbed])

            for i in features.index[absorbed]:
                neighbors, neighbor_labels= knn.get_neighbors(features.loc[i])

                # If majority of k neighbors disagree, remove the point
                if neighbor_labels.value_counts().get(labels[i], 0) <= k // 2:
                    absorbed[i] = False
                    changed = True

        return absorbed[absorbed].index.tolist()

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
        # Initialize a list of indices for the dataset and shuffle them
        indices = features.index.tolist()
        random.shuffle(indices)

        # Initialize the condensed set with the first random point
        first_index = indices.pop(0)
        condensed_indices = [first_index]

        # Loop through the shuffled dataset
        for i in indices:
            knn = self.reducedKNN
            knn.fit(features.loc[condensed_indices], labels.loc[condensed_indices])

            # Predict the label of the current point using the reduced set
            prediction = knn.predict(features.loc[[i]])

            # If the prediction is incorrect, add the point to the condensed set
            if prediction != labels[i]:
                condensed_indices.append(i)

        return condensed_indices

    def editing_algorithm_estimating_class_probabilities_and_threshold(self, features, labels, k=3, mu=0.1):
        """
        Reference: https://campusvirtual.ub.edu/pluginfile.php/8517391/mod_resource/content/1/EENTh_A_Stochastic_Approach_to_Wilsons_Editing_Algorith.pdf

        EENTH Algorithm: A Threshold-based Editing Algorithm for Classification with Nearest Neighbors.

        This algorithm implements a threshold version of Wilson's editing algorithm (EENTH).
        It starts with S = X (the entire dataset), and for each instance, it checks if the instance
        satisfies the following conditions:

        1. A distance threshold (theta) which is the maximum distance among the k nearest neighbors (δ_k-prob(x)).
        2. A class probability threshold (mu), where the maximum class probability p_j of the object x
           must exceed the threshold for it to be retained.

        For each instance in the dataset, if the predicted class probability is too low or the
        distance exceeds the threshold, the point is removed from the set S. This is a one-pass
        algorithm, meaning each point is evaluated once and removed if it doesn't meet the criteria.

        Args:
            features (numpy array): The feature vectors of the dataset.
            labels (numpy array): The corresponding labels for the dataset.
            k (int): The number of nearest neighbors to consider. Default is 3.
            theta (float): The distance threshold for δ_k-prob(x). Default is 0.1.
            mu (float): The minimum acceptable class probability p_j. Default is 0.6.

        Returns:
            Filtered features and labels after removing instances that do not meet the classification criteria.
        """

        indices = []
        for idx in features.index:
            (neighbours, distances), neighbours_labels = self.originalKNN.get_neighbors(features.loc[idx],
                                                                                        custom_k=k + 1,
                                                                                        return_distances=True)
            neighbours, distances, neighbours_labels = neighbours[1:], distances[1:], neighbours_labels[1:]
            weighting_vector = 1 / (np.ones(len(distances)) + distances)

            class_probabilities = {}
            for index, l in enumerate(neighbours_labels):
                class_probabilities[l] = class_probabilities.get(l, 0) + weighting_vector[index]

            # Estimate class probabilities for point x
            predicted_class = max(class_probabilities, key=class_probabilities.get)  # Class with highest probability
            p_j = class_probabilities[predicted_class] / sum(class_probabilities.values())  # Normalize probability
            # If the predicted class does not match the actual class or the thresholds are violated, remove the point
            if not (predicted_class != labels[idx] or p_j <= mu):
                indices.append(idx)

        return indices

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


