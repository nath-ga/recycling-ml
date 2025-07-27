# evaluate_and_log_predictions.py

import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm

# 🔁 Parameter
USE_PADDING = True  # ← Ändere zu False für Modell ohne Padding
MODEL_PATH = "models/model_padded.pt" if USE_PADDING else "models/model.pt"
CSV_PATH = "predictions_padded.csv" if USE_PADDING else "predictions_plain.csv"
TEST_DIR = "data/test"

# 🔧 Transforms
if USE_PADDING:
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

# 📂 Dataset & Dataloader
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
class_names = test_dataset.classes
paths = [s[0] for s in test_dataset.samples]

# 🧠 Modell laden
device = torch.device("cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 🧾 Auswertung + Logging
records = []

with torch.no_grad():
    for i, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predicted = class_names[preds[0].item()]
        true = class_names[labels[0].item()]
        filename = os.path.basename(paths[i])

        records.append({
            "filename": filename,
            "true_label": true,
            "predicted_label": predicted
        })

# 💾 Speichern
df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False)
print(f"✅ Vorhersagen gespeichert in {CSV_PATH}")
