import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Config
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 3
DATA_DIR = "data/wikiart_style5"
MODEL_DIR = "models"

# Crée le dossier models/ s'il n'existe pas
os.makedirs(MODEL_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
num_classes = len(train_dataset.classes)

class BaselineMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 128 * 128, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = BaselineMLP(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model_m1_baseline.pth"))
print("Modèle m1 sauvegardé dans models/model_m1_baseline.pth")