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
import random

def train(epochs = 50,batch_size = 16):
    BASE_DIR = Path(__file__).resolve().parent.parent
    json_file = BASE_DIR / "Configs" / "training.json"
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_dir = BASE_DIR / "checkpoints"
    all_files = list(checkpoint_dir.glob("*.pth"))
    start_epoch = 0

    if all_files:
        all_files.sort(key=lambda f: os.path.getmtime(f))
        latest_checkpoint = checkpoint_dir / all_files[-1]
        import re
        match = re.search(r'model_epoch_(\d+).pth', str(latest_checkpoint))
        if match:
            start_epoch = int(match.group(1)) 
    else:
        latest_checkpoint = None

    with open(json_file, "r") as f:
        training_data = json.load(f)

    writer = SummaryWriter(log_dir="runs/exp1")

    # transform = transforms.Compose([
    #     transforms.RandomRotation(degrees=30),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    # transform = transforms.Compose([
    #     transforms.RandomChoice([
    #         transforms.RandomRotation((0,0)),
    #         transforms.RandomRotation((90, 90)),   
    #         transforms.RandomRotation((180, 180)), 
    #         transforms.RandomRotation((270, 270))  
    #     ]),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    class RandomRotate20:
        def __init__(self, angles=(90, 180, 270), p=0.2):
            self.angles = angles
            self.p = p

        def __call__(self, img):
            if random.random() < self.p:
                angle = random.choice(self.angles)
                return transforms.functional.rotate(img, angle)
            return img

    transform = transforms.Compose([
        RandomRotate20(p=0.2),
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

    checkpoint_exists = latest_checkpoint is not None and os.path.exists(latest_checkpoint)
    if checkpoint_exists:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        print("Loaded model weights from checkpoint.")
    else:
        print("No checkpoint found. Starting from scratch.")

    if checkpoint_exists:
        lr = 1e-5   
    else:
        lr = 1e-3   
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    early_stopping = EarlyStopping(patience=6, min_delta=0.001)

    for epoch in range(start_epoch,start_epoch+epochs):
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

                img = inputs[0].cpu().clone()
                gt = dataset.classes[labels[0].item()]
                pred = dataset.classes[preds[0].item()]
                fname = batch["file_name"][0]

                img_np = img.numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)

                font_scale = 0.5
                thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX

                gt_text = f"GT: {gt}"
                gt_size = cv2.getTextSize(gt_text, font, font_scale, thickness)[0]
                cv2.rectangle(img_np, (5, 5), (5 + gt_size[0], 5 + gt_size[1] + 5), (255, 255, 255), -1)
                cv2.putText(img_np, gt_text, (5, gt_size[1] + 5), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

                pred_text = f"Pred: {pred}"
                pred_size = cv2.getTextSize(pred_text, font, font_scale, thickness)[0]
                x_pos = img_np.shape[1] - pred_size[0] - 5
                cv2.rectangle(img_np, (x_pos, 5), (x_pos + pred_size[0], 5 + pred_size[1] + 5), (255, 255, 255), -1)
                cv2.putText(img_np, pred_text, (x_pos, pred_size[1] + 5), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

                img_np = cv2.resize(img_np, (img_np.shape[1] * 2, img_np.shape[0] * 2), interpolation=cv2.INTER_LINEAR)

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