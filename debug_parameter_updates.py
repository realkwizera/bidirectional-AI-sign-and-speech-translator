"""
DEBUG: Check if model parameters are actually updating
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.asl_ctc_transformer import ASLCTCModel

class SmallDataset(Dataset):
    def __init__(self, num_samples=2):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        frames = torch.randn(30, 3, 64, 64)
        labels = torch.tensor([2, 3, 4], dtype=torch.long)
        return {
            'frames': frames,
            'labels': labels,
            'frame_len': torch.tensor(30),
            'label_len': torch.tensor(3)
        }

def collate_fn_small(batch):
    frames = torch.stack([b['frames'] for b in batch])
    labels = torch.cat([b['labels'] for b in batch])
    frame_lens = torch.stack([b['frame_len'] for b in batch])
    label_lens = torch.stack([b['label_len'] for b in batch])
    return {
        'frames': frames,
        'labels': labels,
        'frame_lens': frame_lens,
        'label_lens': label_lens
    }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Create model
model = ASLCTCModel(num_classes=28).to(device)
model.train()

# Save initial weights
initial_weights = {}
for name, param in model.named_parameters():
    if 'proj' in name or 'ctc_head' in name:  # Check last layers
        initial_weights[name] = param.data.clone()

print("Saved initial weights of key layers\n")

# Dataset and optimizer
dataset = SmallDataset(num_samples=2)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_small)

criterion = nn.CTCLoss(blank=0, reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("="*70)
print("TRAINING ITERATIONS")
print("="*70)

# Just 5 training iterations
for iteration in range(5):
    for batch in dataloader:
        frames = batch['frames'].to(device)
        labels = batch['labels'].to(device)
        frame_lens = batch['frame_lens'].to(device)
        label_lens = batch['label_lens'].to(device)
        
        # Forward
        logits = model(frames)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs_ctc = log_probs.permute(1, 0, 2)
        
        loss = criterion(log_probs_ctc, labels, frame_lens, label_lens)
        
        print(f"\nIteration {iteration+1}:")
        print(f"  Loss: {loss.item():.4f}")
        
        # Check parameter updates
        for name in initial_weights.keys():
            param = model.state_dict()[name]
            initial = initial_weights[name]
            diff = (param - initial).abs().mean().item()
            print(f"  Param change in {name}: {diff:.6f}")
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"  Total gradient norm: {total_grad_norm:.6f}")
        
        optimizer.step()

print("\n" + "="*70)
print("FINDINGS:")
print("="*70)
print("Check if:")
print("1. Parameter changes > 0? (learning/not learning)")
print("2. Gradient norms > 0? (backprop working)")
print("3. Loss decreasing? (convergence)")
print("="*70)
