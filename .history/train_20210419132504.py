import torch
import numpy as np
import random
from data import getTrainingValidationTestingData
from model import Net
# from common import *
from criterion import DepthLoss
from utils import config
import utils
import argparse as arg

# from util import evaluate_model, make_plot, config

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main(device=torch.device('cuda:0')):
    # CLI arguments  
    parser = arg.ArgumentParser(description='We all know what we are doing. Fighting!')
    parser.add_argument("--datasize", "-d", default="small", type=str, help="data size you want to use, small, medium, total")
    # Parsing
    args = parser.parse_args()
    # Data loaders
    datasize = args.datasize
    pathname = "data/nyu.zip"
    tr_loader, va_loader, te_loader = getTrainingValidationTestingData(datasize, pathname, batch_size=config("unet.batch_size"))

    # Model
    model = Net()
    model = model.to(device)

    # TODO: define loss function, and optimizer
    learning_rate = utils.config("unet.learning_rate")
    criterion = DepthLoss(0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    number_of_epoches = 10
    #

    # print("Number of float-valued parameters:", util.count_parameters(model))

    # Attempts to restore the latest checkpoint if exists
    print("Loading unet...")
    model, start_epoch, stats = utils.restore_checkpoint(model, utils.config("unet.checkpoint"))

    # axes = utils.make_training_plot()

    # Evaluate the randomly initialized model
    # evaluate_epoch(
    #     axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats
    # )
    # loss = criterion()

    # initial val loss for early stopping
    # prev_val_loss = stats[0][1]

    running_va_loss = []
    running_va_acc = []
    running_tr_loss = []
    running_tr_acc = []
    # TODO: define patience for early stopping
    # patience = 1
    # curr_patience = 0
    #
    tr_acc, tr_loss = utils.evaluate_model(model, tr_loader, device)
    acc, loss = utils.evaluate_model(model, va_loader, device)
    running_va_acc.append(acc)
    running_va_loss.append(loss)
    running_tr_acc.append(tr_acc)
    running_tr_loss.append(tr_loss)

    # Loop over the entire dataset multiple times
    # for epoch in range(start_epoch, config('cnn.num_epochs')):
    epoch = start_epoch
    # while curr_patience < patience:
    while epoch < number_of_epoches:
        # Train model
        utils.train_epoch(device, tr_loader, model, criterion, optimizer)
        tr_acc, tr_loss = utils.evaluate_model(model, tr_loader, device)
        va_acc, va_loss = utils.evaluate_model(model, va_loader, device)
        running_va_acc.append(va_acc)
        running_va_loss.append(va_loss)
        running_tr_acc.append(tr_acc)
        running_tr_loss.append(tr_loss)
        # Evaluate model
        # evaluate_epoch(
        #     axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats
        # )

        # Save model parameters
        utils.save_checkpoint(model, epoch + 1, utils.config("unet.checkpoint"), stats)

        # update early stopping parameters
        """
        curr_patience, prev_val_loss = early_stopping(
            stats, curr_patience, prev_val_loss
        )
        """

        epoch += 1
    print("Finished Training")
    # Save figure and keep plot open
    # utils.save_training_plot()
    # utils.hold_training_plot()
    utils.make_plot(running_tr_loss, running_tr_acc, running_va_loss, running_va_acc)


if __name__ == "__main__":
    main()
