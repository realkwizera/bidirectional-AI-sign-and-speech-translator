"""
STEP 4: Transformer Encoder Validation
Tests transformer in isolation for attention and output correctness
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.transformer_encoder import TransformerEncoder

def validate_transformer():
    """Validate transformer encoder in isolation"""
    print("\n" + "="*70)
    print("STEP 4: TRANSFORMER ENCODER VALIDATION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Initialize model
    model = TransformerEncoder(dim=256, heads=8, layers=3).to(device)
    model.eval()
    print(f"✓ Transformer model loaded")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {num_params:,}")
    
    # Test 1: Basic output shape
    print("\nTest 1: Output shape")
    B, T, D = 4, 30, 256
    x = torch.randn(B, T, D).to(device)
    print(f"  Input shape: {x.shape} (B={B}, T={T}, D={D})")
    
    with torch.no_grad():
        output = model(x)
    
    expected_shape = (B, T, D)
    assert output.shape == expected_shape, f"❌ Shape mismatch: {output.shape} vs {expected_shape}"
    print(f"  ✓ Output shape: {output.shape}")
    
    # Test 2: Attention weights
    print("\nTest 2: Attention weights")
    with torch.no_grad():
        output, attn = model(x, return_attn=True)
    
    assert attn is not None, "❌ Attention returned None!"
    print(f"  ✓ Attention shape: {attn.shape}")
    assert attn.shape[0] == B, f"❌ Batch mismatch in attention"
    assert attn.shape[1] == T, f"❌ Time mismatch in attention"
    print(f"  ✓ Attention dimensions correct")
    
    # Test 3: Attention values (should sum to 1 per timestep)
    attn_sum = attn.sum(dim=1)
    print(f"  Attention sum range: [{attn_sum.min():.4f}, {attn_sum.max():.4f}]")
    assert ((attn_sum >= 0.9) & (attn_sum <= 1.1)).all(), "⚠ Attention weights don't sum to 1"
    print(f"  ✓ Attention sums approximately to 1")
    
    # Test 4: Temporal context propagation
    print("\nTest 3: Temporal context propagation")
    # Two identical batches should have identical output
    x1 = torch.randn(2, 30, 256).to(device)
    x2 = x1.clone()
    
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    
    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-4, f"❌ Determinism issue: diff={diff}"
    print(f"  ✓ Deterministic output (diff={diff:.2e})")
    
    # Test 5: Gradient flow
    print("\nTest 4: Gradient flow")
    model.train()
    x = torch.randn(2, 30, 256, requires_grad=True, device=device)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and (param.grad != 0).any():
            has_grad = True
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                print(f"  ✓ Gradients in {name}: norm={grad_norm:.4f}")
    
    assert has_grad, "❌ No gradients flowing through transformer!"
    
    # Test 6: Varying sequence lengths
    print("\nTest 5: Variable sequence lengths")
    for T_test in [10, 20, 30, 50]:
        x_var = torch.randn(2, T_test, 256).to(device)
        with torch.no_grad():
            out_var = model(x_var)
        assert out_var.shape == (2, T_test, 256), f"❌ Failed for T={T_test}"
        print(f"  ✓ T={T_test}: {out_var.shape}")
    
    # Test 7: Output value ranges
    print("\nTest 6: Output value ranges")
    x = torch.randn(4, 30, 256).to(device)
    with torch.no_grad():
        output = model(x)
    
    output_range = (output.min().item(), output.max().item())
    print(f"  Output range: [{output_range[0]:.4f}, {output_range[1]:.4f}]")
    assert not torch.isnan(output).any(), "❌ NaN in output!"
    assert not torch.isinf(output).any(), "❌ Inf in output!"
    print(f"  ✓ No NaN/Inf values")
    
    # Test 8: Attention visualization
    print("\nTest 7: Attention heatmap generation")
    if attn is not None:
        for b in range(min(2, B)):
            attn_sample = attn[b]  # (T,)
            print(f"  Sample {b}: min={attn_sample.min():.3f}, max={attn_sample.max():.3f}, mean={attn_sample.mean():.3f}")
    
    print("\n✅ TRANSFORMER VALIDATION PASSED!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    try:
        validate_transformer()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TRANSFORMER VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
