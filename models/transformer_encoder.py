import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):

    def __init__(self, dim=256, heads=8, layers=3):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=512,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

    def forward(self, x, return_attn=False):
        # x: (B, T, D)

        out = self.encoder(x)

        # NOTE: PyTorch transformer does not expose attention by default
        # so we approximate attention using variance across time tokens

        if return_attn:
            attn = torch.softmax(torch.var(out, dim=-1), dim=1)
            return out, attn

        return out