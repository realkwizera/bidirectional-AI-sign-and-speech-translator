"""
Raw Microphone Capture Testing
Tests if microphone is actually capturing audio at the PyAudio level
"""

import pyaudio
import numpy as np
import wave
import sys
from pathlib import Path


def test_microphone_raw_capture(mic_index=None, duration=3, output_file="test_mic_raw.wav"):
    """
    Test microphone capture at raw PyAudio level
    
    Args:
        mic_index: Device index to test (None = default)
        duration: How long to record in seconds
        output_file: Where to save the test WAV file
    """
    print(f"\n{'='*60}")
    print(f"RAW MICROPHONE CAPTURE TEST")
    print(f"{'='*60}")
    print(f"Device Index: {mic_index if mic_index is not None else 'DEFAULT'}")
    print(f"Duration: {duration} seconds")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Get device info
        if mic_index is not None:
            info = p.get_device_info_by_index(mic_index)
        else:
            info = p.get_device_info_by_index(p.get_default_input_device_index())
        
        print(f"Device: {info['name']}")
        print(f"Sample Rate: {int(info['defaultSampleRate'])} Hz")
        print(f"Channels: {int(info['maxInputChannels'])}")
        print(f"\nRECORDING... Speak clearly and loudly!")
        print("-" * 60)
        
        # Record audio
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = int(info['defaultSampleRate'])
        frames = []
        
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=mic_index,
                frames_per_buffer=CHUNK
            )
            
            # Record for duration seconds
            for i in range(int(RATE / CHUNK * duration)):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    
                    # Convert to numpy and check levels
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    rms = np.sqrt(np.mean(audio_data ** 2))
                    peak = np.max(np.abs(audio_data))
                    
                    print(f"  [{i+1:2d}] RMS: {rms:.6f} | Peak: {peak:.6f}", end="\r")
                except Exception as e:
                    print(f"\nError reading frame: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            print("\n" + "-" * 60)
            
        except Exception as e:
            print(f"\n✗ Stream error: {e}")
            p.terminate()
            return False
        
        p.terminate()
        
        # Save to WAV file
        print(f"\nSaving to: {output_file}")
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(4)  # 4 bytes for float32
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        
        # Analyze the file
        print(f"\n{'='*60}")
        print("ANALYSIS")
        print(f"{'='*60}")
        
        with wave.open(output_file, 'rb') as wf:
            total_frames = wf.getnframes()
            duration_actual = total_frames / wf.getframerate()
            print(f"✓ File saved successfully")
            print(f"  Total frames: {total_frames}")
            print(f"  Duration: {duration_actual:.2f} seconds")
            print(f"  File size: {Path(output_file).stat().st_size} bytes")
        
        # Check if file is actually silent
        with wave.open(output_file, 'rb') as wf:
            audio_frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(audio_frames, dtype=np.float32)
            
            rms_total = np.sqrt(np.mean(audio_data ** 2))
            peak_total = np.max(np.abs(audio_data))
            
            print(f"\n  Overall RMS: {rms_total:.6f}")
            print(f"  Overall Peak: {peak_total:.6f}")
            
            if rms_total < 0.01:
                print(f"\n✗ WARNING: Audio is VERY QUIET or SILENT!")
                print(f"  RMS < 0.01 suggests no voice was captured")
                print(f"  → Check microphone mute status")
                print(f"  → Check OS volume settings")
                print(f"  → Try different microphone index")
                return False
            elif rms_total < 0.05:
                print(f"\n⚠ WARNING: Audio is quite quiet")
                print(f"  → Speak louder or closer to microphone")
                return True
            else:
                print(f"\n✓ Audio levels look good!")
                return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_and_test_all_microphones():
    """Test all available microphone devices"""
    print(f"\n{'='*60}")
    print("TESTING ALL MICROPHONE DEVICES")
    print(f"{'='*60}\n")
    
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    
    print(f"Found {device_count} devices:\n")
    
    results = []
    for i in range(device_count):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:  # Input device
            print(f"Device {i:2d}: {info['name']}")
            print(f"          Channels: {info['maxInputChannels']}, Sample Rate: {int(info['defaultSampleRate'])} Hz")
            
            # Quick test
            success = test_microphone_raw_capture(
                mic_index=i,
                duration=2,
                output_file=f"test_mic_{i:02d}.wav"
            )
            results.append((i, info['name'], success))
            print()
    
    p.terminate()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for idx, name, success in results:
        status = "✓ WORKS" if success else "✗ SILENT"
        print(f"Device {idx:2d}: {name:<50} {status}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            mic_index = int(sys.argv[1])
            test_microphone_raw_capture(mic_index=mic_index, duration=5)
        except ValueError:
            print(f"Usage: python test_microphone_capture.py [device_index]")
            print(f"Example: python test_microphone_capture.py 7")
    else:
        # Test all devices
        list_and_test_all_microphones()
