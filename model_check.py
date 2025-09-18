import torch
weights = torch.load("checkpoints/model_final.pth", map_location="gpu")
print(type(weights))
