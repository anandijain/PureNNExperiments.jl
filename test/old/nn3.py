import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img = Image.open(img_path).convert('L')
        self.img = np.array(self.img) / 255.0
        self.H, self.W = self.img.shape
        self.transform = transform

    def __len__(self):
        return self.H * self.W

    def __getitem__(self, idx):
        i = idx // self.W
        j = idx % self.W
        sample = {'image': torch.FloatTensor([i/self.H, j/self.W]), 
                  'label': torch.FloatTensor([self.img[i, j]])}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

def train_model(model, dataloader, criterion, optimizer, num_epochs=1000):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch['image']
            labels = batch['label']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {total_loss/len(dataloader):.3f}")

model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
dataset = ImageDataset("data/10057.png")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
train_model(model, dataloader, criterion, optimizer)
