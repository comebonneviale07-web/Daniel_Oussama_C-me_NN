import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ==========================================
# 1. CONFIGURATION ET GESTION DU MATÉRIEL
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # Taux d'apprentissage plus bas pour le transfer learning
EPOCHS = 10
DATA_DIR = "data/wikiart_style5"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Détection automatique (GPU si dispo, sinon CPU pour coller aux bornes du musée)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> Périphérique utilisé : {device}")

# ==========================================
# 2. TRANSFORMATIONS SPÉCIFIQUES POUR RESNET
# ==========================================
# ResNet attend des images de 224x224 et une normalisation spécifique (ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Chargement du dataset
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# Découpage strict 80% Entraînement / 20% Validation
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(full_dataset.classes)
print(f"--> Dataset : {train_size} train, {val_size} validation.")
print(f"--> Classes : {full_dataset.classes}")

# ==========================================
# 3. ARCHITECTURE DU MODÈLE (ResNet-18 Pré-entraîné)
# ==========================================
# Chargement du ResNet-18 avec ses poids pré-entraînés par défaut
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modification de la dernière couche (fully connected) pour correspondre à nos 5 classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# Transfert du modèle sur le périphérique (CPU/GPU)
model = model.to(device)

# Critère de perte et optimisateur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. BOUCLE D'ENTRAÎNEMENT ET DE VALIDATION
# ==========================================
for epoch in range(EPOCHS):
    # --- Phase d'entraînement ---
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
    
    # --- Phase de validation ---
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
    
    print(f"Époque {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

# ==========================================
# 5. SAUVEGARDE DU MODÈLE CNN
# ==========================================
model_path = os.path.join(MODEL_DIR, "model_m3_resnet.pth")
torch.save(model.state_dict(), model_path)
print(f"--> Modèle m3 (ResNet) sauvegardé avec succès dans : {model_path}")