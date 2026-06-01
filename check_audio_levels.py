import wave
import numpy as np
import sys
from pathlib import Path

audio_dir = Path("audio_captures")
files = sorted(audio_dir.glob("*.wav"), reverse=True)[:3]

print("=" * 60)
print("LATEST AUDIO CAPTURES - AUDIO LEVEL ANALYSIS")
print("=" * 60)

for filepath in files:
    print(f"\n📁 {filepath.name}")
    
    try:
        with wave.open(str(filepath)) as f:
            frames = f.readframes(f.getnframes())
            audio_int16 = np.frombuffer(frames, dtype=np.int16)
            
            rms = np.sqrt(np.mean(audio_int16.astype(float) ** 2))
            peak = np.max(np.abs(audio_int16))
            
            print(f"   RMS level: {rms:.0f}")
            print(f"   Peak level: {peak:.0f}")
            
            if rms > 1000:
                print("   ✓ AUDIO DETECTED - Good levels!")
            elif rms > 500:
                print("   ⚠ AUDIO QUIET - Speak louder")
            else:
                print("   ✗ AUDIO SILENT - No voice")
    except Exception as e:
        print(f"   Error: {e}")

print("\n" + "=" * 60)
