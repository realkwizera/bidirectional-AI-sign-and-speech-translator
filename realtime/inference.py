import numpy as np
import torch

from nlp.beam_search_decoder import BeamSearchDecoder

class InferenceEngine:

    def __init__(self, model, device, labels):

        self.model = model
        self.device = device
        self.labels = labels

        self.decoder = BeamSearchDecoder(beam_width=5)

    def predict(self, sequence):

        sequence = sequence.to(self.device)

        with torch.no_grad():
            logits = self.model(sequence)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # reshape to fake temporal distribution if single output
        probs_sequence = [probs[0]] * 10

        sentence = self.decoder.decode(probs_sequence, self.labels)

        return sentence