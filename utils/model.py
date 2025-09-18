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
        nn.Linear(num_ftrs, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.45),
        nn.Linear(1024, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.45),
        nn.Linear(256, num_classes)
    )

    # Ensure FC is trainable
    for param in resnet.fc.parameters():
        param.requires_grad = True

    return resnet
