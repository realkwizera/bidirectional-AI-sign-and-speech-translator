import cv2
import torch
import numpy as np
from collections import deque

from models.asl_transformer import ASLTransformer

class RealtimePipeline:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = ASLTransformer().to(self.device)
        self.model.eval()

        self.cap = cv2.VideoCapture(0)

        self.buffer = deque(maxlen=30)

    def preprocess(self, frame):

        frame = cv2.resize(frame, (64, 64))
        frame = frame / 255.0

        frame = np.transpose(frame, (2, 0, 1))
        return torch.tensor(frame, dtype=torch.float32)

    def run(self):

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            tensor = self.preprocess(frame)
            self.buffer.append(tensor)

            if len(self.buffer) == 30:

                seq = torch.stack(list(self.buffer))
                seq = seq.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = self.model(seq)
                    pred = torch.argmax(out, dim=1).item()

                print("Prediction:", pred)

            cv2.imshow("ASL", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break