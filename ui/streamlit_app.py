import streamlit as st
import cv2
import numpy as np

from ui.attention_viz import overlay_attention

st.title("🧠 ASL Transformer with Attention Visualization")

run = st.checkbox("Start Camera")

cap = cv2.VideoCapture(0)

frame_placeholder = st.image([])

while run:

    ret, frame = cap.read()
    if not ret:
        continue

    # fake attention for demo (replace with model output)
    attention = np.random.rand(30)

    frame = overlay_attention(frame, attention)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_placeholder.image(frame)