from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd


class SVM():
    """
    SVM Classifier Class with One-vs-Rest for multiclass problems.

    Attributes:
        - train_data: DataFrame of training features.
        - train_labels: Series of training labels.
        - kernel: Kernel type to be used in the SVM ('linear', 'rbf', etc.)
        - C: Regularization parameter for the SVM.
        - gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    """

    def __init__(self, train_data: pd.DataFrame, train_labels: pd.Series, kernel: str = 'rbf', C: float = 1.0,
                 gamma: str = 'scale'):
        self.train_data = train_data
        self.train_labels = train_labels
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None

    def train(self):
        """
        Train the SVM model using the One-vs-Rest strategy for multiclass classification.
        """
        base_svc = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self.model = OneVsRestClassifier(base_svc)
        self.model.fit(self.train_data, self.train_labels)
        print(f"Model trained with kernel='{self.kernel}', C={self.C}, gamma='{self.gamma}'")

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

        predictions = self.predict(test_data)
        test_accuracy = accuracy_score(test_labels, predictions)
        # Automatically generate class labels
        unique_labels = sorted(self.train_labels.unique())
        class_names = [f'Class {i}' for i in unique_labels]
        report = classification_report(test_labels, predictions, target_names=class_names)

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Classification Report:\n{report}")

        return test_accuracy, report

    def cross_validate(self, cv: int = 5):
        """
        Perform cross-validation on the training data using One-vs-Rest strategy.
        Args:
        - cv: Number of folds for cross-validation.
        Returns cross-validation accuracy.
        """
        if self.model is None:
            raise ValueError("The model needs to be trained before cross-validation.")

        cv_scores = cross_val_score(self.model, self.train_data, self.train_labels, cv=cv)
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return cv_scores

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
        print(f"Model parameters updated: kernel='{self.kernel}', C={self.C}, gamma='{self.gamma}'")
