import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('mas_data_ex_3.5.csv')

X = data[['x1', 'x2']].values

y_labels = {
    'ineq1': data['label1'].values,
    'ineq2': data['label2'].values,
    'ineq3': data['label3'].values,
    'ineq4': data['label4'].values
}

models = {}
for key in y_labels:
    model = LinearRegression()
    model.fit(X, y_labels[key])
    models[key] = model
    print(f"Model trained explicitly for {key}: coefficients = {model.coef_}, intercept = {model.intercept_}")

plt.figure(figsize=(8,8))

inside_MAS = data['inside_MAS'] == 1
plt.scatter(X[inside_MAS,0], X[inside_MAS,1], color='green', label='Inside MAS', alpha=0.5)
plt.scatter(X[~inside_MAS,0], X[~inside_MAS,1], color='red', label='Outside MAS', alpha=0.5)

x1_plot = np.linspace(-3,3,200)
for key, model in models.items():
    a, b = model.coef_
    c = model.intercept_
    if b != 0:
        x2_plot = -(a*x1_plot + c)/b
        plt.plot(x1_plot, x2_plot, label=f'Boundary {key}')
    else:
        x_vert = -c/a
        plt.axvline(x=x_vert, label=f'Boundary {key}')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Multiple Linear Regression Approximation of MAS')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
