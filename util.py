import matplotlib.pyplot as plt
import numpy as np


def make_plot(loss: list, accuracy: list):
    """
    Make a side-by-side training plot of loss and accuracy.

    loss: List of the loss values
    accuracy: List of the accuracy values
    """
    fig, axes = plt.subplots(1, 2)
    fig.suptitle('Training plot across epochs')
    axes[0].set_xlabel('Epoch')
    axes[1].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_ylabel('Loss')
    x = np.arange(len(loss))
    axes[0].plot(x, accuracy)
    axes[1].plot(x, loss)
