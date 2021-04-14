import matplotlib.pyplot as plt
import numpy as np
import torch


def make_plot(loss_list: list, acc: list):
    """
    Make a side-by-side training plot of loss_list and accuracy.

    loss_list: List of the loss_list values
    accuracy: List of the accuracy values
    """
    fig, axes = plt.subplots(1, 2)
    fig.suptitle('Training plot across epochs')
    axes[0].set_xlabel('Epoch')
    axes[1].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_ylabel('Loss')
    x = np.arange(len(loss_list))
    axes[0].plot(x, acc, '--o')
    axes[1].plot(x, loss_list, '--o')
    plt.show()


def evaluate_model(model, loader, device):
    """
    Calculate and return the accuracy (average relative error) of the mode upon validation or test set.

    model: the model to evaluate.
    loader: the dataloader of test or validation set
    device: either CPU or CUDA
    """
    model.eval()
    accuracies = []
    with torch.no_grad:
        for batch, truth in loader:
            batch = batch.to(device)
            truth = truth.to(device)
            pred = model(batch)
            accuracies.append(torch.mean(torch.abs(pred - truth) / truth).item())
        acc = sum(accuracies) / len(accuracies)
        print("Evaluation accuracy: {}".format(acc))
    return acc


if __name__ == '__main__':
    loss = np.arange(10)
    accuracy = np.arange(10) + 1
    make_plot(loss, accuracy)
