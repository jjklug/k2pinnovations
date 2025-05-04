import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


data = pd.read_csv('mas_data_ex_3.5.csv')
X = data[['x1', 'x2']].values

label_names = ['label1', 'label2', 'label3', 'label4']
y_labels = {}

for name in label_names:
    y = (data[name] <= 0).astype(int)  
    y_labels[name] = y

models = {}
for i, name in enumerate(label_names):
    model = LogisticRegression()
    model.fit(X, y_labels[name])
    models[name] = model
    print(f"{name}: Coefficients = {model.coef_}, Intercept = {model.intercept_}")

x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 400)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 400)
X1, X2 = np.meshgrid(x1_range, x2_range)
grid_points = np.c_[X1.ravel(), X2.ravel()]

Z_all = []
for model in models.values():
    probs = model.predict_proba(grid_points)[:, 1]  
    Z = probs.reshape(X1.shape)
    Z_all.append(Z)

plt.figure(figsize=(8, 8))

inside = data['inside_MAS'] == 1
outside = data['inside_MAS'] == 0
plt.scatter(X[inside, 0], X[inside, 1], c='green', label='Inside MAS', alpha=0.4)
plt.scatter(X[outside, 0], X[outside, 1], c='red', label='Outside MAS', alpha=0.4)

for i, Z in enumerate(Z_all):
    plt.contour(X1, X2, Z, levels=[0.5], linewidths=2, linestyles='--', label=f'Boundary {i+1}')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Four Logistic Regression Boundaries Approximating MAS')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# Compute and plot confusion matrix based on the inside_MAS column
y_true = data['inside_MAS'].values  # True labels from the inside_MAS column

# Combine predictions from all models to determine the final classification
# Use the maximum probability across all models as the decision
predicted_probs = np.zeros(X.shape[0])
for model in models.values():
    predicted_probs += model.predict_proba(X)[:, 1]  # Sum probabilities from all models

# Final prediction: classify as 1 (inside MAS) if the combined probability is >= 0.5
y_pred = (predicted_probs >= 0.5).astype(int)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
disp.plot(cmap='Blues', values_format='d')
plt.title('Logistic Regression(Test Dataset) - Inside MAS Classification')
plt.show()