#!/usr/bin/env python3
"""
Test if Python can access the microphone at all.
This checks Windows permissions and device availability.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_pyaudio_access():
    """Test if PyAudio can open the default input device"""
    print("\n" + "="*60)
    print("🎤 TEST 1: PyAudio Direct Access")
    print("="*60)
    
    try:
        import pyaudio
        
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        print(f"✓ PyAudio initialized")
        print(f"✓ Found {device_count} audio devices")
        
        # Get default input device
        default_input = p.get_default_input_device_info()
        print(f"\n✓ Default Input Device:")
        print(f"  Index: {default_input['index']}")
        print(f"  Name: {default_input['name']}")
        print(f"  Channels: {default_input['maxInputChannels']}")
        print(f"  Sample Rate: {int(default_input['defaultSampleRate'])} Hz")
        
        # Try to open the device
        try:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024,
                input_device_index=None,  # Use default
            )
            print(f"\n✓ Successfully opened default input device")
            
            # Try to read one frame
            try:
                data = stream.read(1024, exception_on_overflow=False)
                print(f"✓ Successfully read audio data ({len(data)} bytes)")
            except Exception as e:
                print(f"✗ Failed to read audio: {e}")
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"✗ Failed to open input device: {e}")
            print("\n⚠️  DIAGNOSIS:")
            print("  - Microphone may be disabled in Windows")
            print("  - Windows may not have granted permission")
            print("  - Physical microphone may not be connected")
            return False
        
        p.terminate()
        return True
        
    except ImportError:
        print("✗ PyAudio not installed")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_speech_recognition_access():
    """Test if SpeechRecognition library can access microphone"""
    print("\n" + "="*60)
    print("🎙️  TEST 2: SpeechRecognition Microphone Access")
    print("="*60)
    
    try:
        import speech_recognition as sr
        
        # Try to list microphones
        try:
            mics = sr.Microphone.list_microphone_names()
            print(f"✓ Found {len(mics)} microphone(s)")
            if len(mics) > 0:
                print(f"  Default: {mics[0]}")
        except Exception as e:
            print(f"✗ Could not list microphones: {e}")
            return False
        
        # Try to open the default microphone
        try:
            with sr.Microphone() as source:
                print(f"✓ Successfully opened default microphone")
                print(f"  Sample rate: {source.SAMPLE_RATE}")
                print(f"  Chunk size: {source.CHUNK}")
            return True
        except Exception as e:
            print(f"✗ Failed to open microphone: {e}")
            print("\n⚠️  DIAGNOSIS:")
            print("  - Microphone is not accessible to SpeechRecognition")
            print("  - May need Windows permission grant")
            print("  - Device may be disabled or disconnected")
            return False
            
    except ImportError:
        print("✗ SpeechRecognition not installed")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_windows_permissions():
    """Check Windows microphone permissions"""
    print("\n" + "="*60)
    print("🔐 TEST 3: Windows Microphone Permissions")
    print("="*60)
    
    print("""
To grant Windows microphone permission to Python:

1. Open Settings (Windows key + I)
2. Go to: Privacy & Security → Microphone
3. Under "Allow access to microphone on this device" → Toggle ON
4. Under "Allow desktop apps to access your microphone" → Toggle ON
5. Scroll down and make sure your Python installation is allowed

Alternative: Run as Administrator
  - Right-click PowerShell
  - Select "Run as administrator"
  - Run the app again

Check System Tray:
  - Look for microphone icon in system tray
  - Make sure it's not showing as blocked or muted
  - Check volume mixer for any muted applications
""")


def main():
    print("\n" + "█"*60)
    print("█ MICROPHONE DIAGNOSTIC TOOL")
    print("█"*60)
    
    results = {}
    
    # Run tests
    results['pyaudio'] = test_pyaudio_access()
    results['speech_recognition'] = test_speech_recognition_access()
    
    # Show permissions info
    test_windows_permissions()
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    if results['pyaudio'] and results['speech_recognition']:
        print("\n✅ All tests passed! Microphone should work.")
        print("\nTry running the app again:")
        print("  python realtime/inference_ctc.py --mic-energy-threshold 200")
        return 0
    else:
        print("\n❌ Some tests failed. Likely issues:")
        if not results['pyaudio']:
            print("  • PyAudio cannot access microphone")
            print("  • Check Windows permissions (Privacy & Security → Microphone)")
            print("  • Check if microphone is enabled in Device Manager")
        if not results['speech_recognition']:
            print("  • SpeechRecognition cannot access microphone")
            print("  • Grant microphone permission to Python/PowerShell")
            print("  • Try running as Administrator")
        
        print("\nNext steps:")
        print("  1. Check microphone in Windows Settings")
        print("  2. Run PowerShell as Administrator")
        print("  3. Try the app again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
