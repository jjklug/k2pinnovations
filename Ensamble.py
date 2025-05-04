import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import VotingClassifier

print("--------------------------------------------------------------")

#set seed for repeatability
seed = 42
rng = np.random.default_rng(seed)

#import data
data = pd.read_csv('mas_dataset_NN.csv')

X = data[['X_1','X_2','X_3','X_4','X_5','X_6','X_7','X_8','X_9','X_10']].values #input states

y = data['Y'].values  # whether inside or outside of MAS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed) #70/30 train test split


# Train several neural networks on different bootstrap samples
n_ensemble = 120
models = []
for i in range(n_ensemble):
    idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_bootstrap = X_train[idx]
    y_bootstrap = y_train[idx]
    clf = MLPClassifier(hidden_layer_sizes=(5,10), max_iter=1000, random_state=i)
    clf.fit(X_bootstrap, y_bootstrap)
    models.append((f'nn_{i}', clf))

# Create a voting ensemble
ensemble = VotingClassifier(estimators=models, voting='soft')
ensemble.fit(X_train, y_train)

# Evaluate performance
individual_accuracies = [accuracy_score(y_test, clf.predict(X_test)) for _, clf in models]
ensemble_accuracy = accuracy_score(y_test, ensemble.predict(X_test))

print("NN Accuracies: ", individual_accuracies)
print("Ensemble Accuracy:" , ensemble_accuracy)

predictions = ensemble.predict(X_test)

fig, ax = plt.subplots()

def add_point(x, y, color):
    ax.scatter(x, y, color=color, s=50)
    plt.draw()

for i in range(0, len(y_test)):
    color = 'black'
    predicted_value = predictions[i]
    actual_value = y_test[i]
    x_on_plot = X_test[i][0]
    y_on_plot = X_test[i][1]
    if predicted_value == actual_value:
        if predicted_value == 1:
            color = 'light_green'
        else:
            color = 'pink'
    else:
        if predicted_value == 1:
            color = 'green'
        else:
            color = 'red'
    add_point(x_on_plot, y_on_plot, color)
    
    plt.show()


'''This part of the code isn't working properly so I used the above method instead
#shuffles the data and then rotates through creating as many unique subsets as possible
#with the given subset size and step
#For example with 80% and 20% it will generate subsets containing:
#0-80%, 20-100%, 40-100% + 0-20%, 60-100% + 0-40%, and 80-100% + 0-60%.
def rotating_subsets(X_train, y_train, percent_to_train_on, percent_to_step):
    shuffled_indices = rng.permutation(len(X_train))
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    n = len(X_train)
    subset_size = int(percent_to_train_on * n)
    step = int(percent_to_step * n)
    number_of_sets = int(100 // (percent_to_step*100))
    subsets = []
    for i in range(0, number_of_sets):
        start = i * step
        end = start + subset_size
        if end <= n:
            indices = list(range(start, end))
        else:
            # Wrap around
            end = end % n
            indices = list(range(start, n)) + list(range(0, end))
        X_subset = X_train[indices]
        y_subset = y_train[indices]
        subsets.append((X_subset, y_subset))
    return subsets

percent_to_train_on = .1
percent_to_step = .005
subsets = rotating_subsets(X_train, y_train, percent_to_train_on, percent_to_step)
models =[]
print(len(subsets))
for i in range(0,len(subsets)):
    clf = MLPClassifier(hidden_layer_sizes=(5,10), max_iter=1600, random_state=42)
    clf.fit(subsets[i][0], subsets[i][1])
    models.append(clf)

predictions = []
for model in models:
    prediction = model.predict(X_test)
    predictions.append(prediction)
combined_predictions = list(zip(*predictions))

ensabmble_predictions = []
for pred_list in combined_predictions:
    ensabmble_predictions.append(int(sum(pred_list)/len(pred_list)))

print("Accuracy:", accuracy_score(y_test, ensabmble_predictions, normalize=True))'''

print("---------------------------------------------------------")