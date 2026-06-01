import torch
import torch.nn as nn


class TemporalConvModule(nn.Module):
    """Lightweight temporal model that preserves one output per input frame."""

    def __init__(self, hidden_dim=256, num_layers=3, kernel_size=3, dropout=0.25):
        super().__init__()
        layers = []
        padding = kernel_size // 2
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, C)
        y = x.transpose(1, 2)
        for layer in self.layers:
            y = y + layer(y)
        y = y.transpose(1, 2)
        return self.norm(y)
