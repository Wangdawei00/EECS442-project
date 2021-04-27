import matplotlib.pyplot as plt
import numpy as np
import torch
from criterion import DepthLoss
import os
import itertools


def config(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(config, "config"):
        with open("config.json") as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split("."):
        node = node[part]
    return node


def make_plot(tr_loss: list, tr_acc: list, va_loss, va_acc):
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
    x = np.arange(len(tr_loss))
    axes[0].plot(x, tr_acc, '--o')
    axes[0].plot(x, va_acc, '--o')
    axes[0].legend(['Train', 'Validation'])
    axes[1].plot(x, tr_loss, '--o')
    axes[1].plot(x, va_loss, '--o')
    axes[1].legend(['Train', 'Validation'])
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
    losses = []
    for i, batch in enumerate(loader):
        X = torch.Tensor(batch["image"]).to(device)
        y = torch.Tensor(batch["depth"]).to(device)
        #batch = batch.to(device)
        #truth = truth.to(device)
        outputs = model(X)
        accuracies.append(torch.mean(torch.abs(outputs - y) / y).item())
        loss = DepthLoss(0.1)
        losses.append(loss(outputs, y).item())
    acc = sum(accuracies) / len(accuracies)
    loss = sum(losses) / len(losses)
    print("Evaluation accuracy: {}".format(acc))
    return acc, loss


def train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`.

    Use `optimizer` to optimize the specified `criterion`
    """
    # for i, (X, y) in enumerate(data_loader):
    for i, batch in enumerate(data_loader):
        print("trainning... batch number", i)
        optimizer.zero_grad()
        X = torch.Tensor(batch["image"]).to(device)
        y = torch.Tensor(batch["depth"]).to(device)
        outputs = model(X)
        # calculate loss
        loss = criterion(outputs, y)
        # gradient_loss = gradient_criterion(outputs, y, device="cuda")
        loss.backward()
        optimizer.step()


def save_checkpoint(model, epoch, checkpoint_dir, stats):
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats,
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(state, filename)


def check_for_augmented_data(data_dir):
    """Ask to use augmented data if `augmented_dogs.csv` exists in the data directory."""
    if "augmented_dogs.csv" in os.listdir(data_dir):
        print("Augmented data found, would you like to use it? y/n")
        print(">> ", end="")
        rep = str(input())
        return rep == "y"
    return False


def restore_checkpoint(model, checkpoint_dir, cuda=False, force=False, pretrain=False):
    """Restore model from checkpoint if it exists.

    Returns the model and the current epoch.
    """
    try:
        cp_files = [
            file_
            for file_ in os.listdir(checkpoint_dir)
            if file_.startswith("epoch=") and file_.endswith(".checkpoint.pth.tar")
        ]
    except FileNotFoundError:
        cp_files = None
        os.makedirs(checkpoint_dir)
    if not cp_files:
        print("No saved model parameters found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print(
            "Which epoch to load from? Choose in range [0, {}].".format(epoch),
            "Enter 0 to train from scratch.",
        )
        print(">> ", end="")
        inp_epoch = int(input())
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        print("Which epoch to load from? Choose in range [1, {}].".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(
        checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(inp_epoch)
    )

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    try:
        start_epoch = checkpoint["epoch"]
        stats = checkpoint["stats"]
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats


def clear_checkpoint(checkpoint_dir):
    """Remove checkpoints in `checkpoint_dir`."""
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


if __name__ == '__main__':
    train_loss = np.arange(10)
    validation_loss = train_loss + 1
    train_accuracy = np.arange(10) + 1
    validation_accuracy = train_accuracy + 1
    make_plot(train_loss, train_accuracy, validation_loss, validation_accuracy)
