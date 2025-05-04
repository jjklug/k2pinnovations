import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

#K-Folds imports
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

print("-------------------------------------------------------------")

#import data
data = pd.read_csv('mas_dataset_NN.csv')

X = data[['X_1','X_2','X_3','X_4','X_5','X_6','X_7','X_8','X_9','X_10']].values #input states

y = data['Y'].values  # whether inside or outside of MAS


#seed
seed = 42
rng = np.random.default_rng(seed)

kf = KFold(n_splits=8, shuffle=True, random_state=seed)

NUM_ENSAMBLES = 120
ensemble_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    models = []
    for i in range(NUM_ENSAMBLES):
        sample_idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        X_bootstrap = X_train[sample_idx]
        y_bootstrap = y_train[sample_idx]

        clf = MLPClassifier(hidden_layer_sizes=(5, 10), max_iter=1000, random_state=i)
        clf.fit(X_bootstrap, y_bootstrap)
        models.append((f'nn_{i}', clf))

    ensemble = VotingClassifier(estimators=models, voting='soft')
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    ensemble_scores.append(accuracy)
    print(f"Fold {fold_idx + 1} accuracy: {accuracy:.3f}")

print(f"Mean accuracy: {np.mean(ensemble_scores):.3f}")

'''
Fold 1 accuracy: 0.992
Fold 2 accuracy: 0.984
Fold 3 accuracy: 0.986
Fold 4 accuracy: 0.987
Fold 5 accuracy: 0.981
Fold 6 accuracy: 0.989
Fold 7 accuracy: 0.976
Fold 8 accuracy: 0.987
Mean accuracy: 0.985
'''