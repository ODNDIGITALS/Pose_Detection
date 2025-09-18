import torch
weights = torch.load("checkpoints/model_final.pth", map_location="cpu")
print(type(weights))
