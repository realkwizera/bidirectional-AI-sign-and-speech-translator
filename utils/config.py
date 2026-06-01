"""
Configuration and utilities for ASL CTC System
"""

import torch
from pathlib import Path

# Character mapping for CTC
BLANK_ID = 0
CHAR_TO_ID = {' ': 1}  # space is ID 1
for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    CHAR_TO_ID[char] = i + 2  # A-Z are IDs 2-27

ID_TO_CHAR = {v: k for k, v in CHAR_TO_ID.items()}
ID_TO_CHAR[BLANK_ID] = ''

# Model paths
MODEL_DIR = Path("checkpoints")
BEST_MODEL = MODEL_DIR / "best_model.pt"

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Character to ID mapping: {CHAR_TO_ID}")
print(f"Total classes: {len(CHAR_TO_ID) + 1} (including blank)")
print(f"Using device: {DEVICE}")
