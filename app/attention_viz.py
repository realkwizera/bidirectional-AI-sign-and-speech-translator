import numpy as np
import matplotlib.pyplot as plt


def plot_attention(attention_matrix):

    if attention_matrix is None:
        return None

    if len(attention_matrix.shape) == 4:
        attention_matrix = attention_matrix[0, 0]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(attention_matrix, cmap="viridis")

    ax.set_title("Transformer Attention")
    plt.colorbar(im, ax=ax)

    return fig