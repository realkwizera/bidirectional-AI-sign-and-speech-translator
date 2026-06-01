import torch
import torch.nn as nn

class CNNBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        # Add BatchNorm2d after each Conv2d for better gradient flow and feature variance
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),  # FIX BUG #1: Added batch norm
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),  # FIX BUG #1: Added batch norm
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),  # FIX BUG #1: Added batch norm
            nn.ReLU()
        )

        self.proj = nn.Linear(128 * 8 * 8, 256)
        
        # Initialize projection with proper scaling (FIX BUG #1)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.proj(x)