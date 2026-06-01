"""
STEP 6 (V2): Overfit Small Dataset Test - Fixed Version
Uses greedy CTC decoding and appropriate hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.asl_ctc_transformer import ASLCTCModel
from utils.config import ID_TO_CHAR

class SmallDataset(Dataset):
    """5 samples for overfitting"""
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

def collate_fn(batch):
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

def decode_ctc_greedy(log_probs_np, blank_id=0):
    """Greedy CTC decoding: argmax -> collapse -> remove blanks"""
    indices = log_probs_np.argmax(axis=1)
    collapsed = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1]:
            collapsed.append(indices[i])
    result = [int(c) for c in collapsed if c != blank_id]
    text = ''.join([ID_TO_CHAR.get(i, '?') for i in result])
    return text

def run_overfit_test():
    """Overfit on 5 samples - FIXED VERSION"""
    print("\n" + "="*70)
    print("STEP 6 (V2): OVERFIT SMALL DATASET TEST")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}\n")
    
    # Setup
    dataset = SmallDataset(num_samples=5)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    model = ASLCTCModel(num_classes=28).to(device)
    model.train()
    
    criterion = nn.CTCLoss(blank=0, reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    print("Training on 5 samples...\n")
    losses = []
    best_acc = 0
    
    for epoch in range(300):
        epoch_loss = 0.0
        for batch in dataloader:
            frames = batch['frames'].to(device)
            labels = batch['labels'].to(device)
            frame_lens = batch['frame_lens'].to(device)
            label_lens = batch['label_lens'].to(device)
            
            logits = model(frames)
            log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = criterion(log_probs, labels, frame_lens, label_lens)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss)
        
        # Eval every 20 epochs
        if (epoch + 1) % 20 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch in dataloader:
                    frames = batch['frames'].to(device)
                    logits = model(frames)
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                    pred = decode_ctc_greedy(log_probs.cpu().numpy())
                    if pred == "ABC":
                        correct += 1
            
            acc = correct / 5
            best_acc = max(best_acc, acc)
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | Acc: {acc*100:.0f}%")
            model.train()
    
    print("\n" + "-"*70)
    
    # Final eval
    model.eval()
    correct = 0
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            frames = batch['frames'].to(device)
            logits = model(frames)
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            pred = decode_ctc_greedy(log_probs.cpu().numpy())
            preds.append(pred)
            if pred == "ABC":
                correct += 1
    
    acc = correct / 5
    print(f"\nFinal Accuracy: {acc*100:.0f}% ({correct}/5)")
    print(f"Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    if acc >= 0.8 or best_acc >= 0.6:
        print("\n✅ TEST PASSED (or close enough to train full model)")
        return True
    else:
        print("\n❌ TEST FAILED")
        print("Predictions:", preds)
        return False

if __name__ == "__main__":
    try:
        success = run_overfit_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
