"""
STEP 6 (REVISED): Overfit Small Dataset - Simplified Decoding
Use argmax decoding instead of beam search
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
    """Tiny dataset of 5 samples for overfitting test"""
    def __init__(self, num_samples=5):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        frames = torch.randn(30, 3, 64, 64)
        # Simple labels: each sample has same label for debugging
        labels = torch.tensor([2, 3, 4], dtype=torch.long)  # A, B, C
        return {
            'frames': frames,
            'labels': labels,
            'frame_len': torch.tensor(30),
            'label_len': torch.tensor(3)
        }

def collate_fn_small(batch):
    """Collate for small dataset"""
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

def decode_ctc_simple(log_probs_np, blank_id=0):
    """Simple CTC decoding: greedy + collapse + remove blanks"""
    # log_probs: (T, C)
    indices = log_probs_np.argmax(axis=1)  # T,
    
    # Collapse consecutive duplicates
    collapsed = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1]:
            collapsed.append(indices[i])
    
    # Remove blanks
    result = [c for c in collapsed if c != blank_id]
    
    # Convert to text
    text = ''.join([ID_TO_CHAR.get(i, '?') for i in result])
    return text

def overfit_test():
    """Test overfitting on 5 samples"""
    print("\n" + "="*70)
    print("STEP 6 (REVISED): OVERFIT SMALL DATASET TEST")
    print("Using simplified CTC decoding (greedy + collapse)")
    print("="*70)
    print("Goal: Model should reach ~100% accuracy on 5 samples")
    print("      If not, check architecture/labels/loss")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Create tiny dataset
    dataset = SmallDataset(num_samples=5)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one at a time for clarity
        shuffle=False,
        collate_fn=collate_fn_small
    )
    
    # Initialize model
    model = ASLCTCModel(num_classes=28).to(device)
    model.train()
    print(f"✓ Model initialized")
    
    # Loss and optimizer - use very conservative settings
    criterion = nn.CTCLoss(blank=0, reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    
    print("\nTraining on 5 samples (batch_size=1)...")
    print("-" * 70)
    
    losses = []
    best_accuracy = 0.0
    
    for epoch in range(200):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            frames = batch['frames'].to(device)
            labels = batch['labels'].to(device)
            frame_lens = batch['frame_lens'].to(device)
            label_lens = batch['label_lens'].to(device)
            
            # Forward
            logits = model(frames)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs_ctc = log_probs.permute(1, 0, 2)
            
            # Loss
            loss = criterion(log_probs_ctc, labels, frame_lens, label_lens)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    frames = batch['frames'].to(device)
                    logits = model(frames)
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                    
                    # Decode
                    pred_text = decode_ctc_simple(log_probs.cpu().numpy())
                    expected = "ABC"
                    
                    if pred_text == expected:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            best_accuracy = max(best_accuracy, accuracy)
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy*100:.1f}%")
            
            model.train()
    
    print("-" * 70)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    model.eval()
    
    with torch.no_grad():
        correct = 0
        predictions = []
        
        for batch in dataloader:
            frames = batch['frames'].to(device)
            logits = model(frames)
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            
            pred_text = decode_ctc_simple(log_probs.cpu().numpy())
            expected = "ABC"
            predictions.append((pred_text, expected))
            
            if pred_text == expected:
                correct += 1
    
    accuracy = correct / 5
    
    print(f"Final Accuracy: {accuracy*100:.1f}% ({correct}/5)")
    print(f"Best Accuracy: {best_accuracy*100:.1f}%")
    print("\nPredictions:")
    for i, (pred, exp) in enumerate(predictions):
        status = "✓" if pred == exp else "✗"
        print(f"  {status} Sample {i}: predicted='{pred}', expected='{exp}'")
    
    # Check loss convergence
    print(f"\nLoss convergence: {losses[0]:.4f} → {losses[-1]:.4f}")
    if losses[-1] < 0.5:
        print("✓ Loss converged well")
    else:
        print("⚠ Loss didn't converge (may still be learning)")
    
    if accuracy >= 0.8:
        print("\n✅ OVERFIT TEST PASSED!")
        print("="*70)
        return True
    else:
        print("\n❌ OVERFIT TEST FAILED!")
        print("Note: If loss is decreasing and predictions changing,")
        print("      continue training longer or adjust learning rate.")
        print("="*70)
        return False

if __name__ == "__main__":
    try:
        success = overfit_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ OVERFIT TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
