#this will most likely fail - but the goal of this is to show that it
#will not work as well as the neural network will
# we want to show that the neural network will perform better for a case with
# 10+ plus constraints and a nonlinear solution

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

#read in dataset
data = pd.read_csv('mas_dataset_NN.csv')

# Create labels
X = data[['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10']].values
y = data['Y'].values

# Split up data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train logistic regression model using all features
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model on test set
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of the logistic regression model: {accuracy:.4f}")

# Visualize classification results
# Create a boolean array indicating whether the classification is correct
correct_classification = predictions == y_test

# Plot the data points, coloring them based on whether they were classified correctly
plt.figure(figsize=(8, 8))
plt.scatter(
    X_test[correct_classification, 0], 
    X_test[correct_classification, 1], 
    c='blue', 
    label='Correctly Classified', 
    alpha=0.6
)
plt.scatter(
    X_test[~correct_classification, 0], 
    X_test[~correct_classification, 1], 
    c='orange', 
    label='Incorrectly Classified', 
    alpha=0.6
)


# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Add labels, legend, and grid
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.title('Classification Results Using All Features')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# # Reduce data to 2 dimensions
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# # Plot the reduced data
# plt.figure(figsize=(8, 8))
# plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='blue', label='Class 1', alpha=0.6)
# plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='orange', label='Class 0', alpha=0.6)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA Visualization of the Dataset')
# plt.legend()
# plt.grid(True)
# plt.show()