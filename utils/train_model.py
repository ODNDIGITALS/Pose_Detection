# train.py
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim

from dataload import CustomImageDataset
from model import build_resnet50   

base_dir = Path(__file__).resolve().parent.parent
img_dir = os.path.join(base_dir,"training_images")
print("img_dir is:",img_dir)

transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CustomImageDataset(img_dir=img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = build_resnet50(len(dataset.classes),freeze_backbone=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(50):
    for batch in dataloader:
        inputs = batch["tensor"].to(device)
        labels = batch["label"].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
