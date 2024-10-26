from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
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
                 gamma: str = 'scale', multiclass: str ='ovo'):
        self.train_data = train_data
        self.train_labels = train_labels
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None
        self.multiclass = multiclass

    def train(self):
        """
        Train the SVM model using the One-vs-Rest strategy for multiclass classification.
        """
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, decision_function_shape=self.multiclass)
        self.model.fit(self.train_data, self.train_labels)
        if self.kernel == 'linear':
            print(f"Model trained with kernel='{self.kernel}', C={self.C}")
        else:
            print(f"Model trained with kernel='{self.kernel}', C={self.C}, gamma='{self.gamma}'")


    @property
    def support_vectors_(self):
        """
        Get the support vectors from the trained model.
        Returns:
            numpy.ndarray: Array of support vectors
        """
        if self.model is None:
            raise ValueError("Model needs to be trained before accessing support vectors")
        return self.model.support_vectors_

    @property
    def estimators_(self):

        if self.model is None:
            raise ValueError("Model needs to be trained before accessing estimators_")
        return self.model.estimators_

    @property
    def n_support_(self):
        """
        Get the number of support vectors for each class.
        Returns:
            numpy.ndarray: Array with number of support vectors for each class
        """
        if self.model is None:
            raise ValueError("Model needs to be trained before accessing n_support")
        return self.model.n_support_

    @property
    def dual_coef_(self):
        """
        Get the dual coefficients.
        Returns:
            numpy.ndarray: Array of dual coefficients
        """
        if self.model is None:
            raise ValueError("Model needs to be trained before accessing dual coefficients")
        return self.model.dual_coef_

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
        print(f"Number of support vectors per class: {self.n_support_}")
        print(f"Total number of support vectors: {len(self.support_vectors_)}")

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
