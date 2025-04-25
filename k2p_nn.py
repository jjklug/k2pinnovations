#This file will hopefully be where we build our neural network
#That will create a model with hundreds of contraints to solve the MAS problem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


#load data
data = pd.read_csv('mas_data_ex_3.5.csv')
#input states
X = data[['x1', 'x2']].values
#scaler = StandardScaler()
#X = scaler.fit_transform(X)
y = data[['label1', 'label2', 'label3', 'label4']].values  # Labels


#convert to PyTorch sensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
y_tensor = (y_tensor - y_tensor.min()) / (y_tensor.max() - y_tensor.min())
print(y_tensor)
print(f"Target value range: min={y_tensor.min()}, max={y_tensor.max()}")

#define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(2, 32)  # first Hidden layer with 32 neurons
        self.hidden2 = nn.Linear(32, 16)  # second hidden layer with 16 neurons
        self.output = nn.Linear(16, 4)  # Output layer with 4 neurons (one per label)
        self.activation = nn.ReLU()  # Activation function for hidden layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid for output layer

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.sigmoid(self.output(x))
        return x

# Initialize the model, loss function, and optimizer
model = NeuralNetwork()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

#train the neural network
epochs = 1000
for epoch in range(epochs):
    #forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    #backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Visualize the decision boundaries
x1_plot = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
x2_plot = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
x1_grid, x2_grid = np.meshgrid(x1_plot, x2_plot)
grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
grid_tensor = torch.tensor(grid, dtype=torch.float32)

# Predict for the grid points
with torch.no_grad():
    predictions = model(grid_tensor).numpy()
    print("Sample predictions:", predictions)
    print("Actual labels:", y_tensor[:5])

# Plot the decision boundaries for each label
plt.figure(figsize=(8, 8))
for i in range(4):  # For each label
    plt.contour(x1_grid, x2_grid, predictions[:, i].reshape(x1_grid.shape), levels=[0.5], colors=['blue'], linewidths=1.5, linestyles='--')

# Scatter plot of the data
inside_MAS = data['inside_MAS'] == 1
plt.scatter(X[inside_MAS, 0], X[inside_MAS, 1], color='green', label='Inside MAS', alpha=0.5)
plt.scatter(X[~inside_MAS, 0], X[~inside_MAS, 1], color='red', label='Outside MAS', alpha=0.5)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Neural Network Approximation of MAS')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()