import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ----------------------------
# Dataset Loader
# ----------------------------
class FightDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        classes = ["NonFight", "Fight"]
        for idx, cls in enumerate(classes):
            folder = os.path.join(root_dir, cls)
            for file in os.listdir(folder):
                if file.endswith(".npy"):
                    self.samples.append(os.path.join(folder, file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip = np.load(self.samples[idx])  # (C,T,H,W)
        clip = clip.astype(np.float32) / 255.0  # normalize
        label = self.labels[idx]
        return torch.tensor(clip), torch.tensor(label, dtype=torch.long)


# ----------------------------
# 3D CNN Model (C3D-like)
# ----------------------------
class C3D(nn.Module):
    def __init__(self, num_classes=2):
        super(C3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7 * 1, 4096),  # Adjust depending on input size
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------------
# Training Function
# ----------------------------
def train_model(data_dir, epochs=20, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FightDataset(data_dir)

    # Split dataset (70% train, 15% val, 15% test)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset sizes: Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    model = C3D(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses, train_preds, train_labels = [], [], []
        for clips, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for clips, labels in tqdm(val_dl, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                clips, labels = clips.to(device), labels.to(device)
                outputs = model(clips)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss={np.mean(train_losses):.4f} | "
              f"Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_c3d.pth")
            print("✅ Best model saved!")

    print("Training complete.")


if __name__ == "__main__":
    train_model("processed_dataset", epochs=20, batch_size=8, lr=1e-4)
