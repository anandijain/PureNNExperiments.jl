from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
from PIL import Image
import numpy as np

class TinyImgNet:
    def __init__(self):
        self.l1 = Tensor.uniform(2, 20)
        self.l2 = Tensor.uniform(20, 8)
        self.l3 = Tensor.uniform(8, 1)

    def forward(self, x):
        return x.dot(self.l1).sigmoid().dot(self.l2).sigmoid().dot(self.l3)

    def parameters(self):
        return [self.l1, self.l2, self.l3]

model = TinyImgNet()
optim = optim.Adam(model.parameters(), lr=1)

# Load the image
img = Image.open('data/10057.png').convert('L')
img = np.array(img)

# Normalize and prepare the data
img = img / 255.0
H, W = img.shape
data = []
for i in range(H):
    for j in range(W):
        data.append((Tensor([i/H, j/W]), Tensor([img[i, j]])))

# Training loop
for epoch in range(1000):
    total_loss = 0
    for x, y in data:
        # Forward pass
        output = model.forward(x)
        # Calculate loss
        loss = (output - y).pow(2).mean()
        total_loss += loss.numpy()
        # Backward pass and optimize
        optim.zero_grad()
        loss.backward()
        optim.step()

    # if epoch % 10 == 0:
    print(f"Epoch: {epoch}, Loss: {total_loss / len(data)}")

    # Display output
    output_img = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            output = model.forward(Tensor([i/H, j/W]))
            output_img[i,j] = output.numpy()

    Image.fromarray((output_img * 255).astype(np.uint8), 'L').show()
