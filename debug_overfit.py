"""
DEBUG: Overfit Test with Detailed Logging
Find what's wrong with the CTC loss computation or decoding
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from models.asl_ctc_transformer import ASLCTCModel
from nlp.ctc_beam_search import CTCBeamSearch
from utils.config import ID_TO_CHAR

class SmallDataset(Dataset):
    def __init__(self, num_samples=5):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        frames = torch.randn(30, 3, 64, 64)
        labels = torch.tensor([2, 3, 4], dtype=torch.long)  # A, B, C
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

def debug_overfit():
    print("\n" + "="*70)
    print("DEBUG: OVERFIT TEST WITH DETAILED LOGGING")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Dataset
    dataset = SmallDataset(num_samples=5)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn_small
    )
    
    # Model
    model = ASLCTCModel(num_classes=28).to(device)
    model.train()
    print(f"✓ Model initialized\n")
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    beam_search = CTCBeamSearch(beam_width=5, blank=0)
    
    # Training
    print("TRAINING DEBUG LOG:")
    print("-" * 70)
    
    for epoch in range(10):  # Just 10 epochs for debugging
        for batch_idx, batch in enumerate(dataloader):
            frames = batch['frames'].to(device)
            labels = batch['labels'].to(device)
            frame_lens = batch['frame_lens'].to(device)
            label_lens = batch['label_lens'].to(device)
            
            # Forward
            logits = model(frames)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Debug: Log shapes and values
            print(f"\nEpoch {epoch+1}, Batch {batch_idx+1}:")
            print(f"  Frames shape: {frames.shape}")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Log probs shape: {log_probs.shape}")
            print(f"  Labels: {labels}")
            print(f"  Frame lens: {frame_lens}")
            print(f"  Label lens: {label_lens}")
            
            # Debug: Check logits values
            print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            print(f"  Logits mean: {logits.mean():.4f}")
            
            # Debug: Check probabilities
            probs = torch.softmax(logits, dim=-1)
            print(f"  Probs shape: {probs.shape}")
            print(f"  Probs range: [{probs.min():.4f}, {probs.max():.4f}]")
            
            # CTC loss
            log_probs_ctc = log_probs.permute(1, 0, 2)  # T, B, C
            loss = criterion(log_probs_ctc, labels, frame_lens, label_lens)
            
            print(f"  CTC Loss: {loss.item():.4f}")
            
            # Check top predictions
            batch_probs = probs[0].cpu().detach()  # First sample
            top_ids = torch.argmax(batch_probs, dim=1)  # Predicted ID at each timestep
            top_ids_list = top_ids.numpy().tolist()
            top_ids_text = ''.join([ID_TO_CHAR.get(i, '?') for i in top_ids_list[:10]])
            print(f"  Top pred IDs (first 10 timesteps): {top_ids_list[:10]}")
            print(f"  Top pred text (first 10): {top_ids_text}")
            
            # Decode with beam search
            decoded = beam_search.decode(batch_probs.numpy(), ID_TO_CHAR)
            print(f"  Beam search decode: '{decoded}'")
            print(f"  Expected: 'ABC' (IDs: [2, 3, 4])")
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if batch_idx >= 2:  # Only log first 2 batches per epoch
                break
        
        print("-" * 70)
        if epoch == 1:  # Only show 2 epochs
            break
    
    print("\n" + "="*70)
    print("DEBUG FINDINGS:")
    print("="*70)
    print("Check above for:")
    print("1. CTC Loss > 0? (yes/no)")
    print("2. Top predicted IDs changing? (varied/constant)")
    print("3. Beam search decoding? (correct/stuck)")
    print("="*70)

if __name__ == "__main__":
    try:
        debug_overfit()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
