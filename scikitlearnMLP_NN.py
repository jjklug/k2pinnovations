import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

data = pd.read_csv('mas_data_ex_3.5.csv')

X = data[['x1', 'x2']].values #input states

y = data['inside_MAS'].values  # whether inside or outside of MAS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #70/30 train test split

#defaults uses relu which can handle nonlinearity
#MPLClassifier is used for binary
clf = MLPClassifier(hidden_layer_sizes=(20,10), max_iter=300, random_state=42)
#clf = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=300, random_state=42)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test) #predictions

print("Accuracy:", accuracy_score(y_test, predictions, normalize=True)) #show accuracy

#graph
all_prediction = clf.predict(X)
x_values = X[:, 0]
y_values = X[:, 1]
plt.scatter(x_values, y_values, c=all_prediction, cmap='viridis', s=100) 
plt.show()

print("starting K folds")
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#K-folds
k = 5 #number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(20,10), max_iter=300, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")
print(f"Standard deviation: {scores.std()}")