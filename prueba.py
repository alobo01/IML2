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


# Create a visualization of the data and decision boundaries
def plot_decision_boundary_and_data(X, y, model, title="SVM Decision Boundary"):
    # Create a mesh grid to plot decision boundary
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min-10, x_max+10, 0.1),
                         np.arange(y_min-10, y_max+10, 0.1))

    # Make predictions for each point in the mesh
    Z = model.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['x', 'y']))
    Z = np.array(Z).reshape(xx.shape)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')

    # Plot data points
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='coolwarm',
                          edgecolors='black', linewidth=1, alpha=0.8)
    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(),
                         title="Classes", loc='lower left', bbox_to_anchor=(0.0, 0.05))

    # Set the facecolor of the legend to white
    legend1.get_frame().set_facecolor('white')

    # Optionally, set the edge color of the legend to black (or any color you want)
    legend1.get_frame().set_edgecolor('black')

    # Add the legend to the axes
    plt.gca().add_artist(legend1)

    plt.title(title)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')

    # Add support vectors if available
    if hasattr(model, 'support_vectors_'):
        plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                    s=100, linewidth=1, facecolors='none', edgecolors='k',
                    label='Support Vectors')
        legend2=plt.legend(loc='lower left', bbox_to_anchor=(0.0, 0.01))
        # Add legend

        # Set the facecolor of the legend to white
        legend2.get_frame().set_facecolor('white')

        # Optionally, set the edge color of the legend to black (or any color you want)
        legend2.get_frame().set_edgecolor('black')

        # Add the legend to the axes
        plt.gca().add_artist(legend2)

    plt.tight_layout()
    return plt


# Plot the data and decision boundaries
plt = plot_decision_boundary_and_data(X, y, svm_classifier,
                                      title="SVM Decision Boundaries (Linear Kernel)")
plt.show()

