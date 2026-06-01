import torch
import torch.nn as nn

from models.cnn_backbone import CNNBackbone
from models.ctc_head import CTCHead
from models.temporal_conv import TemporalConvModule
from utils.hand_landmarks import HAND_LANDMARK_DIM


class ASLCTCModel(nn.Module):
    """CNN + hand-landmark fusion model for letter CTC/classification."""

    def __init__(self, num_classes, landmark_dim=HAND_LANDMARK_DIM, use_landmarks=True, dropout=0.25):
        super().__init__()
        self.use_landmarks = use_landmarks
        self.landmark_dim = landmark_dim
        self.cnn = CNNBackbone()

        if use_landmarks:
            self.landmark_encoder = nn.Sequential(
                nn.Linear(landmark_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(256 + 128, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.landmark_encoder = None
            self.fusion = None

        self.temporal = TemporalConvModule(hidden_dim=256, num_layers=3, kernel_size=3, dropout=dropout)
        self.ctc_head = CTCHead(256, num_classes, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, landmarks=None, return_aux=False):
        # x: (B, T, 3, H, W)
        bsz, steps, channels, height, width = x.shape
        visual = self.cnn(x.reshape(bsz * steps, channels, height, width))

        if self.use_landmarks:
            if landmarks is None:
                landmarks = torch.zeros(
                    bsz,
                    steps,
                    self.landmark_dim,
                    dtype=visual.dtype,
                    device=visual.device,
                )
            landmarks = landmarks.to(device=visual.device, dtype=visual.dtype)
            landmark_features = self.landmark_encoder(landmarks.reshape(bsz * steps, self.landmark_dim))
            visual = self.fusion(torch.cat([visual, landmark_features], dim=-1))

        sequence = visual.reshape(bsz, steps, 256)
        sequence = self.temporal(sequence)
        logits = self.ctc_head(sequence)

        if not return_aux:
            return logits

        return {
            "ctc_logits": logits,
            "class_logits": self.classifier(sequence.mean(dim=1)),
        }
