from utils import config
import numpy as np
import itertools
import os
import torch
from torch.nn.functional import softmax
from sklearn import metrics
import utils
from loss import ssim_loss as ssim_criterion
from loss import gradient_loss as gradient_criterion


def count_parameters(model):
    """Count number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def early_stopping(stats, curr_patience, prev_val_loss):
    """Calculate new patience and validation loss.

    Increment curr_patience by one if new loss is not less than prev_val_loss
    Otherwise, update prev_val_loss with the current val loss

    Returns: new values of curr_patience and prev_val_loss
    """
    # TODO implement early stopping
    curr_val_loss = stats[-1][1]
    if (curr_val_loss > prev_val_loss):
        curr_patience += 1
    else:
        prev_val_loss = curr_val_loss
        curr_patience = 0
    #
    return curr_patience, prev_val_loss


def evaluate_epoch(
    axes,
    tr_loader,
    val_loader,
    te_loader,
    model,
    criterion,  
    epoch,
    stats,
    include_test=False,
    update_plot=True,
    multiclass=False,
):
    """Evaluate the `model` on the train and validation set."""

    def _get_metrics(loader):
        y_true, y_pred = [], []
        #correct, total = 0, 0

        running_loss = []

        #for X, y in loader:
        for i, batch in enumerate(loader):
            with torch.no_grad():
                print("evaluating... batch number", i)
                #X = torch.Tensor(batch["image"]).to("cuda")
                #y = torch.Tensor(batch["depth"]).to(device="cuda")
                X = torch.Tensor(batch["image"])
                y = torch.Tensor(batch["depth"])

                output = model(X)
                predicted = predictions(output.data)
                y_true.append(y)
                y_pred.append(predicted)

                #total += y.size(0)

                #correct += (predicted == y).sum().item()

                # Calculate the net loss of this batch
                depth_loss = criterion(predicted, y)
                #gradient_loss = gradient_criterion(predicted, y, device="cuda")
                gradient_loss = gradient_criterion(predicted, y)
                ssim_loss = torch.clamp(
                    (1-ssim_criterion(predicted, y, 1000.0/10.0))*0.5, 
                    min=0, 
                    max=1
                )
                loss = (1.0 * ssim_loss) + (1.0 * torch.mean(gradient_loss)) + (0.1 * torch.mean(depth_loss))
                #running_loss.append(criterion(output, y).item())
                running_loss.append(loss)

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)

        loss = np.mean(running_loss)

        acc = 0.9
        auroc = 0.9

        return acc, loss, auroc

    print("evaluating training set...")
    train_acc, train_loss, train_auc = _get_metrics(tr_loader)
    print("evaluating validation set")
    val_acc, val_loss, val_auc = _get_metrics(val_loader)

    stats_at_epoch = [
        val_acc,
        val_loss,
        val_auc,
        train_acc,
        train_loss,
        train_auc,
    ]
    if include_test:
        print("evaluating testing set...")
        stats_at_epoch += list(_get_metrics(te_loader))

    stats.append(stats_at_epoch)
    utils.log_training(epoch, stats)
    if update_plot:
        utils.update_training_plot(axes, epoch, stats)



def train_epoch(data_loader, model, criterion, optimizer):
    """Train the `model` for one epoch of data from `data_loader`.

    Use `optimizer` to optimize the specified `criterion`
    """
    #for i, (X, y) in enumerate(data_loader):
    for i, batch in enumerate(data_loader):
        print("trainning... batch number", i)
        optimizer.zero_grad()
        X = torch.Tensor(batch["image"])
        y = torch.Tensor(batch["depth"])
        outputs = model(X)
        # calculate loss
        depth_loss = criterion(outputs, y)
        gradient_loss = gradient_criterion(outputs, y)
        #gradient_loss = gradient_criterion(outputs, y, device="cuda")
        ssim_loss = torch.clamp(
                (1-ssim_criterion(outputs, y, 1000.0/10.0))*0.5, 
                min=0, 
                max=1
            )
        loss = (1.0 * ssim_loss) + (1.0 * torch.mean(gradient_loss)) + (0.1 * torch.mean(depth_loss))
        loss.backward()
        optimizer.step()
    pass




def predictions(logits):
    """Determine predicted class index given logits.

    Returns:
        the predicted class output as a PyTorch Tensor
    """
    # TODO implement predictions
    return logits
