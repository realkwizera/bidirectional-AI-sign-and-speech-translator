import numpy as np
import cv2

def overlay_attention(frame, attention_score):

    """
    Draw attention heatmap over frame
    """

    h, w, _ = frame.shape

    heatmap = np.zeros((h, w), dtype=np.float32)

    # normalize attention
    attn = np.array(attention_score)
    attn = attn / (attn.max() + 1e-6)

    # spread attention across vertical time axis
    for i, score in enumerate(attn):

        y_start = int((i / len(attn)) * h)
        y_end = int(((i + 1) / len(attn)) * h)

        heatmap[y_start:y_end, :] = score

    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    return overlay