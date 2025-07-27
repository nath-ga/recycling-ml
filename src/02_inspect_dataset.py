# 02_inspect_dataset.py

import os
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt

# Pfade
BASE_DIR = Path("C:/Users/Nathalie/neue projekte/recycling-ml/data")
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

def count_images(directory):
    counts = defaultdict(int)
    for class_name in os.listdir(directory):
        class_path = directory / class_name
        image_files = list(class_path.glob("*.*"))
        counts[class_name] = len(image_files)
    return counts

def plot_sample_images(directory, title="Train"):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i, class_name in enumerate(sorted(os.listdir(directory))[:5]):
        class_path = directory / class_name
        sample_image = list(class_path.glob("*.*"))[0]
        img = Image.open(sample_image)
        axs[i].imshow(img)
        axs[i].set_title(class_name)
        axs[i].axis('off')
    plt.suptitle(f"{title} Beispielbilder aus 5 Klassen")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("📊 Bildanzahl pro Klasse (Train):")
    for k, v in count_images(TRAIN_DIR).items():
        print(f"  {k}: {v} Bilder")

    print("\n📊 Bildanzahl pro Klasse (Test):")
    for k, v in count_images(TEST_DIR).items():
        print(f"  {k}: {v} Bilder")

    print("\n📷 Zeige Beispielbilder...")
    plot_sample_images(TRAIN_DIR)
