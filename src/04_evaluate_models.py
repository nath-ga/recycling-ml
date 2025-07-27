# 04_evaluate_models.py

import os
from pathlib import Path
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# === Pfade ===
BASE_DIR = Path("C:/Users/Nathalie/neue projekte/recycling-ml/data")
MODEL_DIR = Path("C:/Users/Nathalie/neue projekte/recycling-ml/models")
TEST_DIR = BASE_DIR / "test"

# === Gemeinsame Transform (ohne Padding!) für Evaluation ===
# Bilder wurden beim Training unterschiedlich vorbereitet, aber hier normalisieren wir sie einheitlich.
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # wird später bei Padded auch wirken, da schon quadratisch
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Testdaten laden ===
test_data = datasets.ImageFolder(TEST_DIR, transform=eval_transform)
test_loader = DataLoader(test_data, batch_size=32)

# === Modelle definieren ===
def load_model(path, num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# === Auswertung ===
def evaluate_model(model, label="Modell"):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100
    print(f"\n📊 {label} Accuracy: {acc:.2f}%")

    print("\n🧾 Klassifikationsbericht:")
    print(classification_report(y_true, y_pred, target_names=test_data.classes))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_data.classes)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title(f"{label} Confusion Matrix")
    plt.tight_layout()
    plt.show()

# === Modelle laden und auswerten ===
plain_model_path = MODEL_DIR / "model.pt"
padded_model_path = MODEL_DIR / "model_padded.pt"

model_plain = load_model(plain_model_path, len(test_data.classes))
model_padded = load_model(padded_model_path, len(test_data.classes))

evaluate_model(model_plain, "Modell ohne Padding")
evaluate_model(model_padded, "Modell mit Padding")
