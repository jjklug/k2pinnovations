import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

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
