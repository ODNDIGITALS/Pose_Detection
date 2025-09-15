# train.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter  # Tensorboard
from model import build_resnet50   
from custom_dataset import CustomImageDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau # Learning Rate Scheduler
from early_stopping import EarlyStopping
import json

BASE_DIR = Path(__file__).resolve().parent.parent
json_file = BASE_DIR/"Configs"/"training.json"

with open(json_file, "r") as f:
    training_data = json.load(f)

writer = SummaryWriter(log_dir="runs/exp1") 

transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CustomImageDataset(json_file = json_file, transform=transform)
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

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3) 
early_stopping = EarlyStopping(patience=6, min_delta=0.001)

os.makedirs("checkpoints", exist_ok=True)
global_step = 0
for epoch in range(50):
    model.train()
    running_loss = 0.0
    train_correct, total = 0, 0
    for batch in train_loader:
        inputs = batch["tensor"].to(device)
        labels = batch["label"].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() # loss for this batch (before weight update)

        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        total += labels.shape[0]

        writer.add_scalar("Loss/Train_Batch", loss.item(), global_step) # monitors batch losses for each epoch
        global_step+=1
        for fname in batch["file_name"]:
           if fname in training_data:
               training_data[fname]["status"] = "trained"

        with open(json_file, "w") as f:
              json.dump(training_data, f, indent=4)

    avg_train_loss = running_loss / len(train_loader)
    train_acc = train_correct / total

    writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch) # monitors epoch losses
    writer.add_scalar("Accuracy/Train", train_acc, epoch)

    model.eval()
    val_loss, correct = 0.0, 0
    sample_images, sample_preds, sample_labels = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs = batch["tensor"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            # Save only first batch for visualization
            if batch_idx == 0:
                sample_images = inputs.cpu()
                sample_preds = preds.cpu()
                sample_labels = labels.cpu()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / len(val_dataset)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))  # 16 images
    for i, ax in enumerate(axes.flat):
        if i < len(sample_images):
            img = sample_images[i].numpy().transpose((1, 2, 0))
            img = np.clip(img, 0, 1)
            gt = dataset.classes[sample_labels[i]]
            pred = dataset.classes[sample_preds[i]]
            ax.imshow(img)
            ax.set_title(f"GT:{gt}\nPred:{pred}", fontsize=7)
            ax.axis("off")
        else:
            ax.axis("off")

    early_stopping(avg_val_loss)     # early stopping
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/Validation", val_acc, epoch)

    scheduler.step(avg_val_loss) # updating learning rate on basis of vaidation loss
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    print(f"Epoch {epoch+1}: "
          f"Train Loss = {avg_train_loss:.4f}, "
          f"Val Loss = {avg_val_loss:.4f}, "
          f"Val Acc = {val_acc:.4f}")
    torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

torch.save(model.state_dict(), "checkpoints/model_final.pth")

