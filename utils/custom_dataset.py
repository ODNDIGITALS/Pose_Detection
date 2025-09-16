import json
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms
import random

BASE_DIR = Path(__file__).resolve().parent.parent
json_file = BASE_DIR/"Configs"/"training.json"

class CustomImageDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, "r") as f:
            data = json.load(f)

        # Group images by pose
        pose_to_files = {}
        for file_name, v in data.items():
            if v.get("status") == "downloaded":  # only keep downloaded
                pose = v["pose"]
                pose_to_files.setdefault(pose, []).append({
                    "path": v["path"],
                    "pose": pose,
                    "file_name": file_name
                })

        
        min_count = min(len(files) for files in pose_to_files.values())
        self.image_info = []
        for pose, files in pose_to_files.items():
            chosen = random.sample(files, min_count)  # pick equal samples
            self.image_info.extend(chosen)

        # Sorted list of unique poses
        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        entry = self.image_info[idx]
        img_path = entry["path"]
        pose = entry["pose"]
        file_name = entry["file_name"]
        label_idx = self.class_to_idx[pose]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        return {
            "tensor": image,
            "label": label_idx,
            "file_name": file_name
        }

train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
    
dataset = CustomImageDataset(json_file=json_file,transform=train_transform)
# all_samples = []

# attribute = dataset[0]
# print(attribute)
        
# print(len(dataset))
# import os
# print(os.path.exists("C:\\Users\\Dell\\Desktop\\Pose_Detection\\training_images\\back\\100243224_100243224_3.jpg"))