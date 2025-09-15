from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from model import build_resnet50
from custom_dataset import CustomImageDataset

# Define BASE_DIR and checkpoint path
BASE_DIR = Path(__file__).resolve().parent
checkpoint_path = BASE_DIR / "checkpoints" / "model_final.pth"

def load_model(checkpoint_path, num_classes, device):
    model = build_resnet50(num_classes=num_classes, freeze_backbone=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def infer(img_dir, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inference transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = CustomImageDataset(img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = load_model(checkpoint_path, len(dataset.classes), device)
    criterion = nn.CrossEntropyLoss()

    total, correct, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["tensor"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    acc = correct / total if total > 0 else 0.0

    print(f"Inference Results → Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    return avg_loss, acc
