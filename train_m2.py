import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 15  # Un peu plus d'époques pour laisser le temps à un réseau profond d'apprendre
DATA_DIR = "data/wikiart_style5"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> Périphérique utilisé : {device}")

# ==========================================
# 2. DATASET ET TRANSFORMS
# ==========================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
num_classes = len(full_dataset.classes)

# ==========================================
# 3. ARCHITECTURE : MLP PROFOND (5 COUCHES CACHÉES + SIGMOID)
# ==========================================
class DeepSigmoidMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # 5 couches cachées successives avec réduction progressive des dimensions
        self.fc1 = nn.Linear(3 * 128 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        
        # Couche de sortie
        self.fc_out = nn.Linear(32, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1) # Équivalent d'un Softmax pour la loss
        
    def forward(self, x):
        x = self.flatten(x)
        
        # Passage à travers les 5 couches cachées
        x = self.dropout(self.sigmoid(self.fc1(x)))
        x = self.dropout(self.sigmoid(self.fc2(x)))
        x = self.dropout(self.sigmoid(self.fc3(x)))
        x = self.dropout(self.sigmoid(self.fc4(x)))
        x = self.dropout(self.sigmoid(self.fc5(x)))
        
        # Sortie
        x = self.fc_out(x)
        x = self.log_softmax(x)
        return x

model = DeepSigmoidMLP(num_classes).to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Listes pour stocker les métriques
train_losses = []
val_losses = []
val_accuracies = []

# ==========================================
# 4. BOUCLE D'ENTRAÎNEMENT
# ==========================================
print("--> Début de l'entraînement du MLP à 5 couches...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    train_loss = running_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f"Époque {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

# ==========================================
# 5. SAUVEGARDE ET TRACÉ DES COURBES
# ==========================================
model_path = os.path.join(MODEL_DIR, "model_m2_deep_sigmoid.pth")
torch.save(model.state_dict(), model_path)
print(f"--> Modèle sauvegardé dans : {model_path}")

# Tracé des courbes de perte (Loss)
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss', marker='o', color='blue')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss', marker='s', color='orange')
plt.xlabel('Époques')
plt.ylabel('Perte (Loss)')
plt.title('Évolution de la fonction de coût (Deep MLP - 5 couches + Sigmoid)')
plt.legend()
plt.grid(True)

graph_path = os.path.join(MODEL_DIR, "training_curves_deep_mlp.png")
plt.savefig(graph_path)
print(f"--> Graphique sauvegardé dans : {graph_path}")
plt.show()