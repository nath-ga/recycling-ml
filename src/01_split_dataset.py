# 01_split_dataset.py

import os
import shutil
import random
from pathlib import Path

# Pfade anpassen
BASE_DIR = Path("C:/Users/Nathalie/neue projekte/recycling-ml/data")
RAW_DIR = BASE_DIR / "raw_data"
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

SPLIT_RATIO = 0.8  # 80 % train, 20 % test

def create_dirs_if_needed():
    for class_name in os.listdir(RAW_DIR):
        (TRAIN_DIR / class_name).mkdir(parents=True, exist_ok=True)
        (TEST_DIR / class_name).mkdir(parents=True, exist_ok=True)

def split_and_copy():
    for class_name in os.listdir(RAW_DIR):
        class_path = RAW_DIR / class_name
        images = list(class_path.glob("*.*"))  # alle Bilder
        random.shuffle(images)

        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        for img in train_imgs:
            shutil.copy(img, TRAIN_DIR / class_name / img.name)
        for img in test_imgs:
            shutil.copy(img, TEST_DIR / class_name / img.name)

    print("Aufteilung abgeschlossen!")

if __name__ == "__main__":
    create_dirs_if_needed()
    split_and_copy()
