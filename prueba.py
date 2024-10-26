# Import necessary libraries
import pandas as pd
from classes.SVM import SVM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data.csv')

# Split the data into features (X) and labels (y)
X = df.iloc[:, :-1]  # All columns except the last one are features
y = df.iloc[:, -1]  # The last column is the label (the class)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the SVM class with the training data
def_kernel='linear'
svm_classifier = SVM(train_data=X_train, train_labels=y_train, kernel=def_kernel, C=1.0,multiclass='ovr')

# Train the SVM model using One-vs-Rest (OvR)
svm_classifier.train()

# Evaluate the model on the test set
svm_classifier.evaluate(test_data=X_test, test_labels=y_test)

# Optional: Perform cross-validation
svm_classifier.cross_validate(cv=5)

# Make predictions on new data (use X_test for demonstration)
predictions = svm_classifier.predict(X_test)

print(predictions)

# Now plot the K groups of points and the K hyperplanes

# Create a scatter plot of the training data, coloring by label
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='coolwarm', s=30, edgecolors='k', label='Train')
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap='coolwarm', marker='x', s=30, label='Test')

# Plot the K separating hyperplanes, one for each class (since it's OvR, we have K hyperplanes)
for i, classifier in enumerate(svm_classifier.model.estimators_):
    w = classifier.coef_[0]  # Get the coefficients for the i-th hyperplane
    a = -w[0] / w[1]

    # Create the decision boundary line
    xx = np.linspace(min(X.iloc[:, 0]), max(X.iloc[:, 0]))
    yy = a * xx - (classifier.intercept_[0]) / w[1]

    # Plot the separating hyperplane
    plt.plot(xx, yy, label=f'Class {i} vs Rest Hyperplane')

    # Margins
    margin = 1 / np.sqrt(np.sum(w ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # Plot the margins
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

# Labels, legends, and title
plt.xlabel('X1')
plt.ylabel('X2')
plt.title(f'SVM {def_kernel} kernel hyperplanes for multiple classification')
plt.legend()
plt.show()