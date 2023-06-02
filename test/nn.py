import torch
from torch import nn
from torch import optim
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Load the image
img = Image.open('data/10057.png').convert('L')
img = np.array(img)

# Normalize and prepare the data
img = img / 255.0
H, W = img.shape
data = []
for i in range(H):
    for j in range(W):
        data.append(([i/H, j/W], img[i, j]))

data = [(torch.FloatTensor(d[0]), torch.FloatTensor([d[1]])) for d in data]

# Define the model architecture


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)


# Instantiate the model
model = Model()

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters())

# Training loop
plt.ion() # turn on interactive mode
losses = []
for epoch in range(1000):
    for x, y in data:
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(x)
        # Calculate loss
        loss = criterion(output, y)
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()

    if epoch % 10 == 0:
        total_loss = sum(criterion(model(d[0]), d[1]) for d in data)
        losses.append(total_loss)
        print(f"Epoch: {epoch}, Loss: {total_loss:.3f}")

    # Display output
    output_img = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            output = model(torch.FloatTensor([i/H, j/W]))
            output_img[i,j] = output.detach().numpy()

    # Update the plot and pause for a bit
    plt.imshow(output_img, cmap='gray')
    plt.draw()
    # plt.pause(0.01)  # pause for a bit
