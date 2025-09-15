import json
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent
json_file = BASE_DIR/"Configs"/"training.json"

class CustomImageDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, "r") as f:
            data = json.load(f)

        self.image_info = []
        self.classes = set()

        # Build list of image entries
        for file_name, v in data.items():
            if v.get("status") == "downloaded":  # only keep downloaded
                pose = v["pose"]
                self.image_info.append({
                    "path": v["path"],
                    "pose": pose,
                    "file_name": file_name
                })
                self.classes.add(pose)

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