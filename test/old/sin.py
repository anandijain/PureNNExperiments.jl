import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Create a simple feedforward network with one hidden layer
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the network
net = Net()

# Use mean squared error loss
criterion = nn.MSELoss()

# Use Adam optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Generate training data
x = np.linspace(0, 440*np.pi, 10000)
y = np.sin(x)
x = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

# Train the network
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Display the function and the approximation
plt.plot(x.detach().numpy(), y.detach().numpy(), label='sin(x)')
plt.plot(x.detach().numpy(), net(x).detach().numpy(), label='Net output')
plt.legend()
plt.show()
