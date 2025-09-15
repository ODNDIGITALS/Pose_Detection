# infer.py
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from model import build_resnet50

# Load model from checkpoint
BASE_DIR = Path(__file__).resolve().parent
checkpoint_path = BASE_DIR / "checkpoints" / "model_final.pth"

# Update this with your actual classes (same order as training)
class_names = ["back","detail_top_front","front","left","right"]

def load_model(checkpoint_path, num_classes, device):
    model = build_resnet50(num_classes=num_classes, freeze_backbone=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Minimal transform (no augmentation, just resize + normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path, model, device):
    image = Image.open(image_path)  # already RGB
    tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        outputs = model(tensor)
        pred = torch.argmax(outputs, dim=1).item()

    return class_names[pred]

