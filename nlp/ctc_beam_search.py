import math

import numpy as np


class CTCBeamSearch:
    def __init__(self, beam_width=5, blank=0):
        self.beam_width = beam_width
        self.blank = blank

    @staticmethod
    def _logaddexp(a, b):
        if a == -math.inf:
            return b
        if b == -math.inf:
            return a
        return float(np.logaddexp(a, b))

    def decode(self, probs, labels):
        """
        Decode CTC probabilities with log-domain beam search.

        Args:
            probs: (T, C) probabilities or log-probabilities.
            labels: mapping from class id to text.
        """
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2:
            raise ValueError(f"Expected probs with shape (T, C), got {probs.shape}")

        if np.nanmax(probs) <= 0.0:
            log_probs = probs
        else:
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))

        beams = {("", self.blank): 0.0}

        for timestep in log_probs:
            next_beams = {}

            for (prefix, last_token), score in beams.items():
                for token, token_score in enumerate(timestep):
                    if token == self.blank:
                        new_key = (prefix, self.blank)
                    else:
                        char = labels.get(token, "")
                        if token == last_token:
                            new_key = (prefix, token)
                        else:
                            new_key = (prefix + char, token)

                    new_score = score + float(token_score)
                    next_beams[new_key] = self._logaddexp(next_beams.get(new_key, -math.inf), new_score)

            beams = dict(
                sorted(next_beams.items(), key=lambda item: item[1], reverse=True)[: self.beam_width]
            )

        best_key = max(beams.items(), key=lambda item: item[1])[0]
        return best_key[0].strip()
