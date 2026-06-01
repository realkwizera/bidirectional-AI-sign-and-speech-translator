import torch
import numpy as np

from models.asl_transformer import ASLTransformer

class PredictorService:

    def __init__(self):
        self.model = ASLTransformer()
        self.model.eval()

    async def predict(self, file):

        # placeholder preprocessing
        tensor = torch.randn(1, 30, 3, 64, 64)

        with torch.no_grad():
            out = self.model(tensor)

        prob = torch.softmax(out, dim=1)

        return {
            "label": int(torch.argmax(prob)),
            "confidence": float(torch.max(prob))
        }