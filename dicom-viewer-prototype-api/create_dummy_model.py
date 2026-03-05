import torch
import torch.nn as nn
from torchvision import models

class KneeAnglePredictor(nn.Module):
    def __init__(self):
        super(KneeAnglePredictor, self).__init__()
        self.backbone = models.resnet50(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.backbone(x)

model = KneeAnglePredictor()
torch.save(model.state_dict(), "/Users/kohei/Documents/dicom-viewer-prototype-api/knee_angle_predictor_best.pth")
print("Dummy model created.")
