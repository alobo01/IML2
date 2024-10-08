# Import necessary libraries
import pandas as pd
from SVM import SVM
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')

# Split the data into features (X) and labels (y)
X = df.iloc[:,:-1]  # All columns except the last one are features
y = df.iloc[:,-1]   # The last column is the label

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the SVM class with the training data
svm_classifier = SVM(train_data=X_train, train_labels=y_train, kernel='rbf', C=1.0, gamma='scale')

# Train the SVM model
svm_classifier.train()

# Evaluate the model on the test set
svm_classifier.evaluate(test_data=X_test, test_labels=y_test)

# Optional: Perform cross-validation
svm_classifier.cross_validate(cv=5)

# Make predictions on new data (use X_test for demonstration)
predictions = svm_classifier.predict(X_test)

print(predictions)
