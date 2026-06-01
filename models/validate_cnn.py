"""
STEP 3: CNN Encoder Validation
Tests CNN in isolation to ensure it extracts features correctly
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.cnn_backbone import CNNBackbone

def validate_cnn():
    """Validate CNN backbone in isolation"""
    print("\n" + "="*70)
    print("STEP 3: CNN ENCODER VALIDATION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Initialize model
    model = CNNBackbone().to(device)
    model.eval()
    print(f"✓ CNN model loaded")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {num_params:,}")
    
    # Test with single frame
    print("\nTest 1: Single frame")
    frame = torch.randn(1, 3, 64, 64).to(device)
    print(f"  Input shape: {frame.shape} (B=1, C=3, H=64, W=64)")
    
    with torch.no_grad():
        features = model(frame)
    
    print(f"  Output shape: {features.shape}")
    expected_shape = (1, 256)
    assert features.shape == expected_shape, f"❌ Shape mismatch: {features.shape} vs {expected_shape}"
    print(f"  ✓ Shape correct: {features.shape}")
    
    assert not torch.isnan(features).any(), "❌ NaN in features!"
    print(f"  ✓ No NaN values")
    
    assert not torch.isinf(features).any(), "❌ Inf in features!"
    print(f"  ✓ No Inf values")
    
    feat_range = (features.min().item(), features.max().item())
    print(f"  ✓ Feature range: [{feat_range[0]:.4f}, {feat_range[1]:.4f}]")
    
    # Test with batch
    print("\nTest 2: Batch of 8 frames")
    batch = torch.randn(8, 3, 64, 64).to(device)
    print(f"  Input shape: {batch.shape}")
    
    with torch.no_grad():
        features_batch = model(batch)
    
    expected_shape = (8, 256)
    assert features_batch.shape == expected_shape, f"❌ Shape mismatch: {features_batch.shape} vs {expected_shape}"
    print(f"  ✓ Output shape: {features_batch.shape}")
    
    # Test with T*B flattened (as used in model)
    print("\nTest 3: Flattened T*B (as in forward pass)")
    B, T = 4, 30
    video = torch.randn(B*T, 3, 64, 64).to(device)
    print(f"  Input shape: {video.shape} (B*T={B*T})")
    
    with torch.no_grad():
        features_flat = model(video)
    
    expected_shape = (B*T, 256)
    assert features_flat.shape == expected_shape, f"❌ Shape mismatch: {features_flat.shape} vs {expected_shape}"
    print(f"  ✓ Output shape: {features_flat.shape}")
    
    # Verify features are different (not constant)
    print("\nTest 4: Feature variance")
    model.train()  # Set to train mode so BatchNorm computes statistics
    # Re-compute features in train mode with new random input
    video_train = torch.randn(B*T, 3, 64, 64).to(device)
    features_train = model(video_train)
    feature_var = features_train.var(dim=0).mean().item()
    # After Kaiming init and BatchNorm, features should have reasonable variance
    assert feature_var > 0.001, f"⚠ Feature variance too low: {feature_var}"
    print(f"  ✓ Feature variance: {feature_var:.6f} (healthy)")
    
    # Gradient flow check
    print("\nTest 5: Gradient flow")
    model.train()
    x = torch.randn(2, 3, 64, 64, requires_grad=True, device=device)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and (param.grad != 0).any():
            has_grad = True
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                print(f"  ✓ Gradients flowing through {name}: norm={grad_norm:.4f}")
    
    assert has_grad, "❌ No gradients flowing through model!"
    
    print("\n✅ CNN VALIDATION PASSED!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    try:
        validate_cnn()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ CNN VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
