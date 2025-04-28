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
#read in dataset
data = pd.read_csv('mas_dataset_NN.csv')

#create labels
X = data[['X_1','X_2','X_3','X_4','X_5','X_6','X_7','X_8','X_9','X_10']]
y = data['Y'].values

#split up data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Train logistic regression models for each feature pair
feature_pairs = [(i,j) for i in range(10) for j in range(i+1, 10)]
print(feature_pairs)
models = {}
for i, (f1, f2) in enumerate(feature_pairs):
    X_train_pair = X_train.iloc[:, [f1, f2]].values # select two feautres
    model = LogisticRegression()
    model.fit(X_train_pair, y_train)
    models[(f1,f2)] = model
    #print(f"Model trained for features X_{f1 + 1} and X_{f2 + 1}: Coefficients = {model.coef_}, Intercept = {model.intercept_}")

#Evaluate model on test set
for (f1, f2), model in models.items():
    X_test_pair = X_test.iloc[:, [f1, f2]].values
    predictions = model.predict(X_test_pair)
    accuracy = accuracy_score(y_test, predictions)
    #print(f"Accuracy for features X_{f1 + 1} and X_{f2 + 1}: {accuracy:.4f}")

#visualize decision boundaries for a selected feature pair
selected_pair = (0, 1)  # Visualize for the first two features (X_1 and X_2)
X_pair = X.iloc[:, list(selected_pair)].values
model = models[selected_pair]

# Create a grid for visualization
x1_range = np.linspace(X_pair[:, 0].min(), X_pair[:, 0].max(), 400)
x2_range = np.linspace(X_pair[:, 1].min(), X_pair[:, 1].max(), 400)
X1, X2 = np.meshgrid(x1_range, x2_range)
grid_points = np.c_[X1.ravel(), X2.ravel()]

# Predict probabilities for the grid points
probs = model.predict_proba(grid_points)[:, 1]
Z = probs.reshape(X1.shape)

# Plot the decision boundary
plt.figure(figsize=(8, 8))
plt.contour(X1, X2, Z, levels=[0.5], linewidths=2, linestyles='--', colors='blue')

# Scatter plot of the data
inside = y == 1
outside = y == 0
plt.scatter(X_pair[inside, 0], X_pair[inside, 1], c='green', label='Inside MAS', alpha=0.4)
plt.scatter(X_pair[outside, 0], X_pair[outside, 1], c='red', label='Outside MAS', alpha=0.4)

# Add labels, legend, and grid
plt.xlabel(f'X_{selected_pair[0] + 1}')
plt.ylabel(f'X_{selected_pair[1] + 1}')
plt.title(f'Logistic Regression Decision Boundary for X_{selected_pair[0] + 1} and X_{selected_pair[1] + 1}')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()