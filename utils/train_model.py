import os
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from model import build_resnet50   
from custom_dataset import CustomImageDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from early_stopping import EarlyStopping
import json
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
json_file = BASE_DIR / "Configs" / "training.json"

with open(json_file, "r") as f:
    training_data = json.load(f)

writer = SummaryWriter(log_dir="runs/exp1")

transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CustomImageDataset(json_file=json_file, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
model = build_resnet50(len(dataset.classes), freeze_backbone=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
early_stopping = EarlyStopping(patience=6, min_delta=0.001)

os.makedirs("checkpoints", exist_ok=True)

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

            if batch_idx == 0:
                imgs_to_log = inputs.clone().cpu()
                for i in range(len(imgs_to_log)):
                    gt = dataset.classes[labels[i].item()]
                    pred = dataset.classes[preds[i].item()]

                    img_np = imgs_to_log[i].numpy().transpose(1, 2, 0)
                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                    img_np = np.ascontiguousarray(img_np)

                    font_scale = 0.5  # smaller text
                    thickness = 1
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # GT text
                    gt_text = f"GT: {gt}"
                    gt_size = cv2.getTextSize(gt_text, font, font_scale, thickness)[0]
                    cv2.rectangle(img_np, (5, 5), (5 + gt_size[0], 5 + gt_size[1] + 5), (255, 255, 255), -1)
                    cv2.putText(img_np, gt_text, (5, gt_size[1] + 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

                    # Pred text
                    pred_text = f"Pred: {pred}"
                    pred_size = cv2.getTextSize(pred_text, font, font_scale, thickness)[0]
                    x_pos = img_np.shape[1] - pred_size[0] - 5
                    cv2.rectangle(img_np, (x_pos, 5), (x_pos + pred_size[0], 5 + pred_size[1] + 5), (255, 255, 255), -1)
                    cv2.putText(img_np, pred_text, (x_pos, pred_size[1] + 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

                    img_np = cv2.resize(img_np, (img_np.shape[1]*2, img_np.shape[0]*2), interpolation=cv2.INTER_LINEAR)
                    img_tensor = torch.tensor(img_np.transpose(2, 0, 1)) / 255.0
                    writer.add_image("Validation/Batch0_AllImages", img_tensor, epoch)

            elif batch_idx % 5 == 0:
                i = np.random.randint(0, len(inputs))
                img = inputs[i].cpu().clone()
                gt = dataset.classes[labels[i].item()]
                pred = dataset.classes[preds[i].item()]

                img_np = img.numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)

                font_scale = 0.5
                thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX

                # GT text
                gt_text = f"GT: {gt}"
                gt_size = cv2.getTextSize(gt_text, font, font_scale, thickness)[0]
                cv2.rectangle(img_np, (5, 5), (5 + gt_size[0], 5 + gt_size[1] + 5), (255, 255, 255), -1)
                cv2.putText(img_np, gt_text, (5, gt_size[1] + 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

                # Pred text
                pred_text = f"Pred: {pred}"
                pred_size = cv2.getTextSize(pred_text, font, font_scale, thickness)[0]
                x_pos = img_np.shape[1] - pred_size[0] - 5
                cv2.rectangle(img_np, (x_pos, 5), (x_pos + pred_size[0], 5 + pred_size[1] + 5), (255, 255, 255), -1)
                cv2.putText(img_np, pred_text, (x_pos, pred_size[1] + 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

                img_np = cv2.resize(img_np, (img_np.shape[1]*2, img_np.shape[0]*2), interpolation=cv2.INTER_LINEAR)
                img_tensor = torch.tensor(img_np.transpose(2, 0, 1)) / 255.0
                writer.add_image(f"Validation/Sample_Batch{batch_idx}", img_tensor, epoch)


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


