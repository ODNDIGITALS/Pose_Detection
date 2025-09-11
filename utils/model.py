import torch
import torch.nn as nn
import torchvision.models as models

def build_resnet50(num_classes=5, freeze_backbone=True):

    resnet = models.resnet50(pretrained=True)
    if freeze_backbone:
        for param in resnet.parameters():
            param.requires_grad = False

    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    # Ensure FC is trainable
    for param in resnet.fc.parameters():
        param.requires_grad = True

    return resnet
