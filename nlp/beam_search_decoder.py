import numpy as np

class BeamSearchDecoder:

    def __init__(self, beam_width=5, blank_token="EMPTY"):
        self.beam_width = beam_width
        self.blank_token = blank_token

    def decode(self, probs_sequence, labels):

        """
        probs_sequence: (T, num_classes)
        labels: list of class labels (A-Z + EMPTY)
        """

        beams = [("", 1.0)]  # (sequence, probability)

        for t in range(len(probs_sequence)):

            new_beams = []

            probs = probs_sequence[t]

            # get top-k candidates at this timestep
            top_k_idx = np.argsort(probs)[-self.beam_width:]

            for seq, score in beams:

                for idx in top_k_idx:

                    label = labels[idx]
                    prob = probs[idx]

                    if label == self.blank_token:
                        new_seq = seq
                    else:
                        new_seq = seq + label

                    new_score = score * prob

                    new_beams.append((new_seq, new_score))

            # keep best beams only
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
            beams = new_beams[:self.beam_width]

        # return best sequence
        return beams[0][0] if beams else ""