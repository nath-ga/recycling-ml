# 03_train_model_padded.py

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import ImageOps

# === Pfade ===
BASE_DIR = Path("C:/Users/Nathalie/neue projekte/recycling-ml/data")
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
MODEL_DIR = Path("C:/Users/Nathalie/neue projekte/recycling-ml/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# === Padding-Funktion ===
def pad_to_square(image, fill_color=(255, 255, 255)):
    width, height = image.size
    max_side = max(width, height)
    delta_w = max_side - width
    delta_h = max_side - height
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(image, padding, fill=fill_color)

# === Bild-Vorverarbeitung mit Padding ===
transform = transforms.Compose([
    transforms.Lambda(lambda img: pad_to_square(img, fill_color=(255, 255, 255))),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Datensätze laden ===
train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_data  = datasets.ImageFolder(TEST_DIR, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=32)

# === Modell vorbereiten ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))  # Anzahl Klassen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# === Training ===
EPOCHS = 5
print(f"📢 Starte Training mit Padding auf {device} {len(train_data.classes)} Klassen")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"🔁 Epoche {epoch+1}/{EPOCHS}, Verlust: {running_loss:.4f}")

# === Modell speichern ===
model_path = MODEL_DIR / "model_padded.pt"
torch.save(model.state_dict(), model_path)
print(f"\n✅ Modell gespeichert unter: {model_path}")
