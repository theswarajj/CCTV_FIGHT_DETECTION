import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

from train import FightDataset, C3D  # reuse dataset + model

def evaluate_model(data_dir="processed_dataset", model_path="checkpoints/best_c3d.pth", batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = FightDataset(data_dir)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    _, _, test_ds = random_split(dataset, [train_size, val_size, test_size])
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Load model
    model = C3D(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for clips, labels in test_dl:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Print classification report
    print("✅ Evaluation Results:")
    print(classification_report(all_labels, all_preds, target_names=["NonFight", "Fight"]))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    evaluate_model()
