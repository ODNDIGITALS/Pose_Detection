import os 
import torch 
from PIL import Image
from pathlib import Path
from torchvision import transforms 
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
 
class CustomImageDataset(Dataset):

    def __init__(self,img_dir,transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.classes = sorted(os.listdir(img_dir))
        self.data = []
        for label,class_name in enumerate(self.classes):
            class_folder = os.path.join(img_dir,class_name)
            for fname in os.listdir(class_folder):
                img_path  = os.path.join(class_folder,fname)
                self.data.append((img_path,label,fname))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        img_path, label, img_name = self.data[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        c,h,w = image.shape
        
        attribute = {
            "tensor": image,
            "label": label,
            "image name": img_name
        }
        return attribute
    
# dataset = CustomImageDataset(img_dir=img_dir,transform=transform)
# all_samples = []

# attribute = dataset[0]
# print(attribute)
        
# print(dataset.shape)
        