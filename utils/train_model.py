import os
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.model import build_resnet50   
from utils.custom_dataset import CustomImageDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.early_stopping import EarlyStopping
import json
import cv2
def train(epochs = 50,batch_size = 16):
    BASE_DIR = Path(__file__).resolve().parent.parent
    json_file = BASE_DIR / "Configs" / "training.json"

    with open(json_file, "r") as f:
        training_data = json.load(f)

    writer = SummaryWriter(log_dir="runs/exp1")

    # transform = transforms.Compose([
    #     transforms.RandomRotation(degrees=30),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    transform = transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomRotation((0,0)),
            transforms.RandomRotation((90, 90)),   
            transforms.RandomRotation((180, 180)), 
            transforms.RandomRotation((270, 270))  
        ]),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = CustomImageDataset(json_file=json_file, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = build_resnet50(len(dataset.classes), freeze_backbone=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    early_stopping = EarlyStopping(patience=6, min_delta=0.001)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
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

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total += labels.shape[0]

            for fname in batch["file_name"]:
                if fname in training_data:
                    training_data[fname]["status"] = "trained"

            with open(json_file, "w") as f:
                json.dump(training_data, f, indent=4)

        avg_train_loss = running_loss / len(train_loader)
        train_acc = train_correct / total
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                inputs = batch["tensor"].to(device)
                labels = batch["label"].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

                img = inputs[0].cpu()
                label_idx = labels[0].item()
                pred_idx = preds[0].item()
                fname = batch["file_name"][0]  

                img_np = img.numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_np, f"GT: {dataset.classes[label_idx]}", (5, 20),
                            font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img_np, f"Pred: {dataset.classes[pred_idx]}", (5, 40),
                            font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                img_tensor = torch.tensor(img_np.transpose(2, 0, 1)) / 255.0
                writer.add_image(f"Validation/{fname}", img_tensor, global_step=epoch)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_dataset)

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        scheduler.step(avg_val_loss)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")

        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "checkpoints/model_final.pth")


