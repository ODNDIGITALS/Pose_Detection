import torch
weights = torch.load("checkpoints/model_final.pth", map_location="cpu")
print(type(weights))
for name, param in weights.items():
    print(name, param.shape)