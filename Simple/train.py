import torch
import numpy as np
import random
from data import getTrainingValidationTestingData
from model import DenseDepth
from common import *
from utils import config
import utils

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():
    """Train CNN and show training plots."""
    # Data loaders
    """
    if check_for_augmented_data("./data"):
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target", batch_size=config("cnn.batch_size"), augment=True
        )
    else:
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target",
            batch_size=config("cnn.batch_size"),
        )
    """
    # pathname = "data/nyu_depth.zip"
    pathname = "data/nyu_small.zip"
    tr_loader, va_loader, te_loader = getTrainingValidationTestingData(pathname, batch_size=config("unet.batch_size"))

    # Model
    model = DenseDepth()

    # TODO: define loss function, and optimizer
    learning_rate = config("unet.learning_rate")
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #

    print("Number of float-valued parameters:", count_parameters(model))

    # Attempts to restore the latest checkpoint if exists
    print("Loading unet...")
    model, start_epoch, stats = restore_checkpoint(model, config("unet.checkpoint"))

    axes = utils.make_training_plot()

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats
    )

    # initial val loss for early stopping
    # prev_val_loss = stats[0][1]

    # TODO: define patience for early stopping
    # patience = 1
    # curr_patience = 0
    #

    # Loop over the entire dataset multiple times
    # for epoch in range(start_epoch, config('cnn.num_epochs')):
    epoch = start_epoch
    # while curr_patience < patience:
    while epoch < 10:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, config("unet.checkpoint"), stats)

        # update early stopping parameters
        """
        curr_patience, prev_val_loss = early_stopping(
            stats, curr_patience, prev_val_loss
        )
        """

        epoch += 1
    print("Finished Training")
    # Save figure and keep plot open
    utils.save_training_plot()
    # utils.hold_training_plot()


if __name__ == "__main__":
    main()
