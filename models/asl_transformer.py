import torch
import torch.nn as nn

from models.cnn_backbone import CNNBackbone
from models.transformer_encoder import TransformerEncoder

class ASLTransformer(nn.Module):

    def __init__(self, num_classes=26):
        super().__init__()

        self.cnn = CNNBackbone()
        self.transformer = TransformerEncoder()

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_attn=False):

        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        x = self.cnn(x)

        x = x.view(B, T, -1)

        if return_attn:
            x, attn = self.transformer(x, return_attn=True)
        else:
            x = self.transformer(x)
            attn = None

        x = x.mean(dim=1)
        logits = self.classifier(x)

        if return_attn:
            return logits, attn

        return logits