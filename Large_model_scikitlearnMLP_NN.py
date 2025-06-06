'''
This file trains a singular Neural Network and runs K-folds validation
Using an Ensemble of NN's is slightly more accurate but this can be used if speed is critical
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#K-Folds imports
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

print("--------------------------------------------------------------")

#import data
data = pd.read_csv('mas_dataset_NN.csv')

X = data[['X_1','X_2','X_3','X_4','X_5','X_6','X_7','X_8','X_9','X_10']].values #input states

y = data['Y'].values  # whether inside or outside of MAS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #70/30 train test split

#defaults uses relu which can handle nonlinearity
#MPLClassifier is used for binary
#clf = MLPClassifier(hidden_layer_sizes=(20,10), max_iter=350, random_state=42) only needs 300 iterations
clf = MLPClassifier(hidden_layer_sizes=(5,10), max_iter=1000, random_state=42)
#clf = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=300, random_state=42)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test) #predictions

print("Accuracy:", accuracy_score(y_test, predictions, normalize=True)) #show accuracy

#K-folds
print("starting K folds")
k = 8 #number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

#model = MLPClassifier(hidden_layer_sizes=(20,10), max_iter=350, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(5,10), max_iter=1000, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")
print(f"Standard deviation: {scores.std()}")

print("---------------------------------------------------------")

'''
#Here we tested different hidden layers.  (5, 10) had the highest accuracy of 98.5%.
#It is commented out to because it takes a while to run
print("starting K folds test")
accuracies = []
hidden_layers = ((20,20), (20,10), (50,10), (10,10), (20, 15,10), (10,5),(5,10), (5,5), (10),
                  (10, 10,10), (5,10, 5), (3,10), (5,10, 15), (6,12), (4,8), (8,16), (5, 10, 2), (5,10))
for i in range(0, len(hidden_layers)):
    k = 8 #number of folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=hidden_layers[i], max_iter=600, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    accuracies.append(scores.mean())
print(accuracies)
'''
