
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Modellpfad
MODEL_PATH = "models/model_padded.pt"

# Klassen
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Bildtransformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Modell laden
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Streamlit App
def predict(image):
    image = transform(image).unsqueeze(0)  # Batch dimension
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs[0], dim=0).numpy()
        pred_idx = probs.argmax()
    return pred_idx, probs

# App-Layout
st.title("♻️ Müllklassifikator Demo")

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    pred_idx, probs = predict(image)
    st.subheader(f"🚀 Vorhersage: **{class_names[pred_idx]}**")

    # Balkendiagramm
    fig, ax = plt.subplots()
    ax.barh(class_names, probs)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Wahrscheinlichkeit")
    ax.set_title("Klassenvorhersage")
    st.pyplot(fig)
