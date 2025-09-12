# train.py
import os
import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split

from model import build_resnet50   
from custom_dataset import CustomImageDataset

base_dir = Path(__file__).resolve().parent.parent
img_dir = os.path.join(base_dir,"training_images")
print("img_dir is:",img_dir)

transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CustomImageDataset(img_dir=img_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
model = build_resnet50(len(dataset.classes),freeze_backbone=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(50):
    # -------------------
    # Training phase
    # -------------------
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs = batch["tensor"].to(device)
        labels = batch["label"].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["tensor"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / len(val_dataset)

    print(f"Epoch {epoch+1}: "
          f"Train Loss = {avg_train_loss:.4f}, "
          f"Val Loss = {avg_val_loss:.4f}, "
          f"Val Acc = {val_acc:.4f}")

