import torch
import torch.nn as nn

class CTCHead(nn.Module):
    """
    CTC prediction head: outputs logits for each character class at each timestep
    Maps transformer embeddings (B, T, dim) → (B, T, num_classes)
    """
    
    def __init__(self, input_dim=256, num_classes=28, dropout=0.1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim) from transformer encoder
        
        Returns:
            logits: (B, T, num_classes)
        """
        # x shape: (B, T, input_dim)
        B, T, D = x.shape
        
        # Reshape to (B*T, D) for linear layers
        x = x.reshape(B * T, D)
        
        # Apply MLP
        logits = self.layers(x)  # (B*T, num_classes)
        
        # Reshape back to (B, T, num_classes)
        logits = logits.reshape(B, T, -1)
        
        return logits
