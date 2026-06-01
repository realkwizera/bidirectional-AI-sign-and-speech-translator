import random
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "data" / "images"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"


st.set_page_config(page_title="ASL Letter Translator", layout="wide")


def normalize_letters(text):
    return [char for char in text.upper() if "A" <= char <= "Z"]


def image_for_letter(letter):
    folder = IMAGE_ROOT / letter
    if not folder.exists():
        return None
    images = [
        path
        for path in folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if not images:
        return None
    return random.choice(images)


def speech_to_text():
    try:
        import speech_recognition as sr
    except ImportError:
        st.warning("Install SpeechRecognition and PyAudio to use microphone input.")
        return ""

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
        return recognizer.recognize_google(audio)
    except Exception as exc:
        st.warning(f"Speech recognition failed: {exc}")
        return ""


def start_live_process():
    if st.session_state.get("live_process") is not None:
        process = st.session_state.live_process
        if process.poll() is None:
            return

    command = [
        sys.executable,
        str(PROJECT_ROOT / "realtime" / "inference_ctc.py"),
        "--model",
        str(CHECKPOINT),
    ]
    st.session_state.live_process = subprocess.Popen(command, cwd=str(PROJECT_ROOT))


def stop_live_process():
    process = st.session_state.get("live_process")
    if process is not None and process.poll() is None:
        process.terminate()
    st.session_state.live_process = None


if "live_process" not in st.session_state:
    st.session_state.live_process = None
if "spoken_text" not in st.session_state:
    st.session_state.spoken_text = ""


st.title("ASL Letter Translator")

sign_tab, text_tab = st.tabs(["Sign to Speech", "Speech/Text to Sign"])

with sign_tab:
    st.subheader("Live sign-to-speech")
    st.caption("The live recognizer opens an OpenCV camera window. It speaks the completed letter set when your hand leaves the frame.")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Start camera", use_container_width=True):
            start_live_process()
    with col2:
        if st.button("Stop camera", use_container_width=True):
            stop_live_process()
    with col3:
        process = st.session_state.get("live_process")
        running = process is not None and process.poll() is None
        st.write("Status:", "running" if running else "stopped")

    st.code(
        f"{sys.executable} realtime/inference_ctc.py --model checkpoints/best_model.pt",
        language="powershell",
    )

with text_tab:
    st.subheader("Speech/text-to-sign")
    st.caption("Type text or capture speech, then the app displays a sequence of matching letter signs from data/images.")

    mic_col, clear_col = st.columns([1, 1])
    with mic_col:
        if st.button("Use microphone", use_container_width=True):
            spoken = speech_to_text()
            if spoken:
                st.session_state.spoken_text = spoken
    with clear_col:
        if st.button("Clear text", use_container_width=True):
            st.session_state.spoken_text = ""

    text = st.text_area(
        "Text",
        value=st.session_state.spoken_text,
        height=90,
        placeholder="Enter words here. Only A-Z letters are displayed as signs.",
    )
    st.session_state.spoken_text = text

    letters = normalize_letters(text)
    st.write("Letters:", " ".join(letters) if letters else "--")

    delay = st.slider("Playback delay", 0.1, 2.0, 0.7, 0.1)
    play = st.button("Play sign sequence", disabled=not letters)
    sequence_box = st.empty()

    if play:
        for letter in letters:
            image_path = image_for_letter(letter)
            if image_path is None:
                sequence_box.warning(f"No image found for {letter}")
            else:
                sequence_box.image(str(image_path), caption=letter, width=320)
            time.sleep(delay)

    if letters:
        columns = st.columns(6)
        for idx, letter in enumerate(letters):
            image_path = image_for_letter(letter)
            with columns[idx % len(columns)]:
                if image_path is None:
                    st.warning(letter)
                else:
                    st.image(str(image_path), caption=letter, use_container_width=True)
