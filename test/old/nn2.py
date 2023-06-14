import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, img_path):
        img = Image.open(img_path).convert('L')
        img = np.array(img) / 255.0
        self.H, self.W = img.shape
        self.data = [(torch.FloatTensor([i/self.H, j/self.W]), torch.FloatTensor([img[i, j]])) for i in range(self.H) for j in range(self.W)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return self.fc4(x)

def train_model(model, dataloader, criterion, optimizer, num_epochs=1000):
    plt.ion()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            total_loss = sum(criterion(model(d[0]), d[1]) for d in dataloader.dataset)
            print(f"Epoch: {epoch}, Loss: {total_loss:.3f}")

            output_img = np.zeros((dataloader.dataset.H, dataloader.dataset.W))
            for i in range(dataloader.dataset.H):
                for j in range(dataloader.dataset.W):
                    output = model(torch.FloatTensor([i/dataloader.dataset.H, j/dataloader.dataset.W]))
                    output_img[i,j] = output.detach().numpy()

            plt.imshow(output_img, cmap='gray')
            plt.draw()
            plt.pause(0.01)

img_path = 'data/10057.png'
dataset = ImageDataset(img_path)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

train_model(model, dataloader, criterion, optimizer)
