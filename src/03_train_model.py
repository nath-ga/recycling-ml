# 03_train_model.py

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# === Pfade anpassen ===
BASE_DIR = Path("C:/Users/Nathalie/neue projekte/recycling-ml/data")
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
MODEL_DIR = Path("C:/Users/Nathalie/neue projekte/recycling-ml/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# === Bild-Vorverarbeitung ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Grob genügt
])

# === Datensätze laden ===
train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_data  = datasets.ImageFolder(TEST_DIR, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=32)

# === ResNet18 laden und anpassen ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))  # Klassenanzahl z. B. 10

# === Trainingsparameter ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# === Training starten ===
EPOCHS = 5
print(f"📢 Starte Training auf {device} – {len(train_data.classes)} Klassen")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"🔁 Epoche {epoch+1}/{EPOCHS}, Verlust: {running_loss:.4f}")

# === Modell speichern ===
model_path = MODEL_DIR / "model.pt"
torch.save(model.state_dict(), model_path)
print(f"\n✅ Modell gespeichert unter: {model_path}")
