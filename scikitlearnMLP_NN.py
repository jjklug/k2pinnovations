from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

data = pd.read_csv('mas_data_ex_3.5.csv')

X = data[['x1', 'x2']].values

y = data['inside_MAS'].values

#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)


print("Accuracy:", accuracy_score(y_test, predictions, normalize=True))