"""
STEP 5: Full Forward Pass Validation
Tests entire pipeline from video frames to predictions
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.asl_ctc_transformer import ASLCTCModel
from nlp.ctc_beam_search import CTCBeamSearch
from utils.config import CHAR_TO_ID, ID_TO_CHAR

def validate_forward_pass():
    """Validate full model forward pass"""
    print("\n" + "="*70)
    print("STEP 5: FULL FORWARD PASS VALIDATION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Initialize model
    model = ASLCTCModel(num_classes=28).to(device)
    model.eval()
    print(f"✓ Model loaded")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {num_params:,}")
    
    # Test 1: Batch through entire pipeline
    print("\nTest 1: Forward pass - single sample")
    B, T, C, H, W = 1, 30, 3, 64, 64
    video = torch.randn(B, T, C, H, W).to(device)
    print(f"  Input: video {video.shape}")
    
    with torch.no_grad():
        logits = model(video)
    
    expected_shape = (B, T, 28)
    assert logits.shape == expected_shape, f"❌ Shape mismatch: {logits.shape} vs {expected_shape}"
    print(f"  ✓ Logits shape: {logits.shape}")
    
    assert not torch.isnan(logits).any(), "❌ NaN in logits!"
    assert not torch.isinf(logits).any(), "❌ Inf in logits!"
    print(f"  ✓ No NaN/Inf values")
    
    # Test 2: Batch processing
    print("\nTest 2: Batch processing (B=8)")
    B = 8
    video_batch = torch.randn(B, T, C, H, W).to(device)
    
    with torch.no_grad():
        logits_batch = model(video_batch)
    
    assert logits_batch.shape == (B, T, 28), f"❌ Batch shape wrong: {logits_batch.shape}"
    print(f"  ✓ Batch output shape: {logits_batch.shape}")
    
    # Test 3: Softmax and probability conversion
    print("\nTest 3: Softmax conversion")
    probs = torch.softmax(logits_batch, dim=-1)
    
    # Check that probabilities sum to 1
    prob_sum = probs.sum(dim=-1)
    assert (prob_sum > 0.99).all() and (prob_sum < 1.01).all(), "❌ Probabilities don't sum to 1!"
    print(f"  ✓ Probabilities sum to 1.0 (range: {prob_sum.min():.4f}-{prob_sum.max():.4f})")
    
    # Test 4: Log softmax for CTC
    print("\nTest 4: Log softmax (for CTC loss)")
    log_probs = torch.log_softmax(logits_batch, dim=-1)
    
    # Rearrange to (T, B, C) for CTC loss
    log_probs_ctc = log_probs.permute(1, 0, 2)
    assert log_probs_ctc.shape == (T, B, 28), f"❌ CTC shape wrong: {log_probs_ctc.shape}"
    print(f"  ✓ CTC-formatted log probs: {log_probs_ctc.shape} (T, B, C)")
    
    # Test 5: Beam search decoding
    print("\nTest 5: Beam search decoding")
    beam_search = CTCBeamSearch(beam_width=5, blank=0)
    
    # Decode single sample
    probs_single = probs[0].cpu().numpy()  # (T, 28)
    decoded_text = beam_search.decode(probs_single, ID_TO_CHAR)
    print(f"  ✓ Decoded text: '{decoded_text}'")
    
    # Validate decoded text only contains valid characters
    valid_chars = set(ID_TO_CHAR.values())
    for char in decoded_text:
        assert char in valid_chars, f"❌ Invalid character in output: {char}"
    print(f"  ✓ Decoded text contains only valid characters")
    
    # Test 6: Batch decoding
    print("\nTest 6: Batch decoding")
    decoded_batch = []
    for b in range(B):
        probs_b = probs[b].cpu().numpy()
        text_b = beam_search.decode(probs_b, ID_TO_CHAR)
        decoded_batch.append(text_b)
    
    print(f"  ✓ Decoded batch ({B} samples):")
    for i, text in enumerate(decoded_batch[:3]):
        print(f"    [{i}] '{text}'")
    if B > 3:
        print(f"    ... and {B-3} more")
    
    # Test 7: CTC Loss
    print("\nTest 7: CTC Loss computation")
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
    
    # Create fake labels
    target_lens = torch.tensor([5]*B)
    targets = torch.randint(1, 28, (B*5,))
    input_lens = torch.tensor([30]*B)
    
    try:
        loss = ctc_loss(log_probs_ctc, targets, input_lens, target_lens)
        assert not torch.isnan(loss), "❌ CTC loss is NaN!"
        assert loss.item() > 0, "❌ CTC loss should be > 0!"
        print(f"  ✓ CTC loss: {loss.item():.4f}")
    except Exception as e:
        print(f"  ❌ CTC loss error: {e}")
        return False
    
    # Test 8: Gradient computation
    print("\nTest 8: Gradient flow through full pipeline")
    model.train()
    
    video_grad = torch.randn(2, 30, 3, 64, 64, requires_grad=True, device=device)
    logits_grad = model(video_grad)
    loss_grad = logits_grad.sum()
    loss_grad.backward()
    
    has_grad = video_grad.grad is not None and (video_grad.grad != 0).any()
    assert has_grad, "❌ No gradient reaching input!"
    print(f"  ✓ Gradients flow through entire model")
    
    # Test 9: Different sequence lengths
    print("\nTest 9: Variable sequence lengths")
    for T_test in [15, 30, 45, 60]:
        video_var = torch.randn(2, T_test, 3, 64, 64).to(device)
        with torch.no_grad():
            logits_var = model(video_var)
        assert logits_var.shape == (2, T_test, 28), f"❌ Failed for T={T_test}"
        print(f"  ✓ T={T_test}: {logits_var.shape}")
    
    # Test 10: Inference mode (no_grad)
    print("\nTest 10: Inference efficiency")
    video_inf = torch.randn(4, 30, 3, 64, 64).to(device)
    
    model.eval()
    with torch.no_grad():
        logits_inf = model(video_inf)
    
    assert logits_inf.shape == (4, 30, 28), "❌ Inference shape wrong!"
    print(f"  ✓ Inference mode works: {logits_inf.shape}")
    
    print("\n✅ FULL FORWARD PASS VALIDATION PASSED!")
    print("="*70)
    print("\nNext step: Run overfit test on small dataset")
    print("Command: python validate_overfit_small.py")
    
    return True

if __name__ == "__main__":
    try:
        validate_forward_pass()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ FORWARD PASS VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
