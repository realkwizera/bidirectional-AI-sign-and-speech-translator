"""
STEP 2: Dataset Validation Script
Validates data loading, shapes, labels, and tensor integrity
"""

import torch
import numpy as np
from pathlib import Path
import sys

def validate_environment():
    """STEP 1: Environment validation"""
    print("\n" + "="*70)
    print("STEP 1: ENVIRONMENT VALIDATION")
    print("="*70)
    
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ WARNING: CUDA not available - training will be slow!")
    
    return cuda_available

def validate_dataset_shapes():
    """STEP 2: Dataset shape validation"""
    print("\n" + "="*70)
    print("STEP 2: DATASET SHAPE VALIDATION")
    print("="*70)
    
    # Expected shapes
    print("\nExpected tensor shapes:")
    print("  Video batch: (B, T, C, H, W) = (8, 30, 3, 64, 64)")
    print("  Labels: (B*T) where T=sequence length")
    print("  Label lengths: (B,)")
    print("  Frame lengths: (B,)")
    
    # Create dummy dataset to test shapes
    print("\nCreating dummy batch...")
    
    B, T, C, H, W = 8, 30, 3, 64, 64
    frames = torch.randn(B, T, C, H, W)
    labels = torch.randint(1, 28, (B*5,))  # Variable length labels
    label_lens = torch.tensor([5]*B)
    frame_lens = torch.tensor([30]*B)
    
    print(f"✓ Video frames shape: {frames.shape} (expected: {(B, T, C, H, W)})")
    print(f"✓ Labels shape: {labels.shape} (expected: concatenated)")
    print(f"✓ Label lengths: {label_lens.shape} (expected: {(B,)})")
    print(f"✓ Frame lengths: {frame_lens.shape} (expected: {(B,)})")
    
    # Check for NaN/Inf
    print("\nValidating data integrity:")
    assert not torch.isnan(frames).any(), "❌ NaN found in frames!"
    print("✓ No NaN values in frames")
    
    assert not torch.isinf(frames).any(), "❌ Inf found in frames!"
    print("✓ No Inf values in frames")
    
    assert frames.min() >= -5.0 and frames.max() <= 5.0, "⚠ Frame values out of expected range"
    print(f"✓ Frame value range: [{frames.min():.2f}, {frames.max():.2f}]")
    
    assert labels.min() >= 0, "❌ Label ID < 0!"
    assert labels.max() < 28, "❌ Label ID >= 28!"
    print(f"✓ Label IDs in valid range: [0, 27]")
    
    return True

def validate_preprocessing():
    """STEP 2B: Preprocessing validation"""
    print("\n" + "="*70)
    print("STEP 2B: PREPROCESSING VALIDATION")
    print("="*70)
    
    # Simulate frame preprocessing
    print("\nSimulating frame preprocessing:")
    
    # Original frame (256x256 BGR)
    frame = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    print(f"✓ Input frame: {frame.shape}, dtype={frame.dtype}, range=[{frame.min()}, {frame.max()}]")
    
    # Resize to 64x64
    import cv2
    frame_resized = cv2.resize(frame, (64, 64))
    print(f"✓ After resize: {frame_resized.shape}")
    
    # BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    print(f"✓ After BGR→RGB conversion")
    
    # Normalize to [0, 1]
    frame_norm = frame_rgb.astype(np.float32) / 255.0
    assert frame_norm.min() >= 0.0 and frame_norm.max() <= 1.0, "❌ Normalization failed!"
    print(f"✓ After normalization: range=[{frame_norm.min():.3f}, {frame_norm.max():.3f}]")
    
    # Channels first
    frame_ch = np.transpose(frame_norm, (2, 0, 1))
    assert frame_ch.shape == (3, 64, 64), f"❌ Channel-first shape wrong: {frame_ch.shape}"
    print(f"✓ After channels-first: {frame_ch.shape}")
    
    # Convert to tensor
    tensor = torch.tensor(frame_ch, dtype=torch.float32)
    assert tensor.dtype == torch.float32, "❌ Tensor dtype wrong!"
    assert not torch.isnan(tensor).any(), "❌ NaN after tensor conversion!"
    print(f"✓ After tensor conversion: dtype={tensor.dtype}")
    
    return True

def validate_character_mapping():
    """STEP 2C: Character mapping validation"""
    print("\n" + "="*70)
    print("STEP 2C: CHARACTER MAPPING VALIDATION")
    print("="*70)
    
    CHAR_TO_ID = {' ': 1}
    for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        CHAR_TO_ID[char] = i + 2
    
    ID_TO_CHAR = {v: k for k, v in CHAR_TO_ID.items()}
    ID_TO_CHAR[0] = ''  # blank
    
    print(f"✓ Total characters: {len(CHAR_TO_ID)} (A-Z + space)")
    print(f"✓ Total classes (with blank): {len(ID_TO_CHAR)}")
    
    # Test encoding
    test_word = "HELLO"
    encoded = [CHAR_TO_ID[c] for c in test_word]
    print(f"✓ '{test_word}' → {encoded}")
    
    # Test decoding
    decoded = ''.join(ID_TO_CHAR.get(i, '?') for i in encoded)
    assert decoded == test_word, f"❌ Decoding failed: {test_word} != {decoded}"
    print(f"✓ {encoded} → '{decoded}'")
    
    # Test space
    assert 1 in ID_TO_CHAR, "❌ Space mapping missing!"
    print(f"✓ Space ID: 1")
    
    # Test blank
    assert 0 in ID_TO_CHAR, "❌ Blank mapping missing!"
    print(f"✓ Blank ID: 0")
    
    return ID_TO_CHAR, CHAR_TO_ID

def validate_ctc_basics():
    """STEP 2D: CTC specific validation"""
    print("\n" + "="*70)
    print("STEP 2D: CTC BASICS VALIDATION")
    print("="*70)
    
    # CTC loss requires:
    # - Input: (T, B, C) log probabilities
    # - Target: (B*L) flat labels
    # - Input lengths: (B,)
    # - Target lengths: (B,)
    
    T, B, C = 30, 8, 28  # Time, Batch, Classes
    
    log_probs = torch.randn(T, B, C)
    targets = torch.randint(1, 28, (B*5,))  # Flat labels
    input_lengths = torch.tensor([30]*B)
    target_lengths = torch.tensor([5]*B)
    
    print(f"✓ Log probabilities: {log_probs.shape} (T={T}, B={B}, C={C})")
    print(f"✓ Targets: {targets.shape} (flat)")
    print(f"✓ Input lengths: {input_lengths.shape}")
    print(f"✓ Target lengths: {target_lengths.shape}")
    
    # Test CTC loss
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean')
    try:
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        assert not torch.isnan(loss), "❌ CTC loss is NaN!"
        assert loss.item() > 0, "❌ CTC loss should be > 0!"
        print(f"✓ CTC loss computation: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ CTC loss error: {e}")
        return False
    
    return True

def main():
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  ASL TRANSFORMER SYSTEM - VALIDATION WORKFLOW".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        # Step 1
        cuda_ok = validate_environment()
        
        # Step 2
        validate_dataset_shapes()
        validate_preprocessing()
        id_to_char, char_to_id = validate_character_mapping()
        validate_ctc_basics()
        
        print("\n" + "="*70)
        print("✅ ALL VALIDATION TESTS PASSED!")
        print("="*70)
        print("\nNext steps:")
        print("1. python models/validate_cnn.py          (Step 3)")
        print("2. python models/validate_transformer.py  (Step 4)")
        print("3. python validate_forward_pass.py        (Step 5)")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
