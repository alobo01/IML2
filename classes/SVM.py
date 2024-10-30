from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pandas as pd
import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize


class SVM():
    """
    SVM Classifier Class for two-class problems.

    Attributes:
        - train_data: DataFrame of training features.
        - train_labels: Series of training labels.
        - kernel: Kernel type to be used in the SVM ('linear', 'rbf', 'poly' or 'sigmoid')
        - C: Regularization parameter for the SVM.
        - gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        - degree: degree when the polynomial kernel is used
    """

    def __init__(self, train_data: pd.DataFrame, train_labels: pd.Series, kernel: str = 'rbf', C: float = 1.0,
                 gamma: str = 'scale', degree: int = 2):
        self.train_data = train_data
        self.train_labels = train_labels
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None
        self.degree= degree

    def train(self):
        """
        Train the SVM model using the One-vs-Rest/One-vs-One strategy for multiclass classification.
        """
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree)
        self.model.fit(self.train_data, self.train_labels)

    def predict(self, data: pd.DataFrame):
        """
        Make predictions using the trained SVM model.
        Args:
        - data: DataFrame containing the test features.
        """
        if self.model is None:
            raise ValueError("The model needs to be trained before making predictions.")
        return self.model.predict(data)


    def evaluate(self, test_data: pd.DataFrame, test_labels: pd.Series):
        """
        Evaluate the trained SVM model on the provided test set.

        Args:
        - test_data: DataFrame of testing features.
        - test_labels: Series of true labels for the test set.

        Returns classification report and test accuracy.
        """
        if self.model is None:
            raise ValueError("The model needs to be trained before evaluation.")

        start_time = time.time()
        y_pred = self.model.predict(test_data)
        elapsed_time = time.time() - start_time
        performance = elapsed_time / len(test_labels)
        #model_metrics = [accuracy_score(test_labels, y_pred),performance,roc_auc_score(test_labels, y_pred),
        #                 recall_score(test_labels, y_pred), precision_score(test_labels, y_pred),
        #                 f1_score(test_labels,y_pred), confusion_matrix(test_labels, y_pred)]

        model_metrics_reduced =[accuracy_score(test_labels, y_pred),performance,f1_score(test_labels,y_pred)]

        return model_metrics_reduced


    def set_params(self, kernel=None, C=None, gamma=None):
        """
        Update the model parameters.
        """
        if kernel:
            self.kernel = kernel
        if C:
            self.C = C
        if gamma:
            self.gamma = gamma

