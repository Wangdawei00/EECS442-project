import torch
import numpy as np
import random
import argparse as arg

from data import getTrainingValidationTestingData
from criterion import DepthLoss
from utils import config
import utils

from res50 import Res50
from dense169 import Dense169
from dense121 import Dense121
from mob_v2 import Mob_v2
from mob_v1 import Net
from squeeze import Squeeze


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

modelSelection = input('Please input the type of model to be used(res50,dense121,dense169,mob_v2,mob,squeeze):')
datasize = input('Please input the size you want to use(small/medium/total): ')
normal = input('Please input whether you want to do normalization(1 for yes/0 for no): ')
normalization = (int(normal) == 1)
grey = input('Please input whether you want to do greyscale as an augmentation(1 for yes/0 for no): ')
greyscale = (int(grey) == 1)
filename = "data/nyu_new.zip"
pathname = f"drive/MyDrive/Dense-Depth/{filename}"
csv = "data/nyu_csv.zip"
tr_loader, va_loader, te_loader = getTrainingValidationTestingData(datasize, csv, filename,
                                                                   batch_size=config(modelSelection + ".batch_size"), normalization = normalization, greyscale = greyscale)


def main(device, tr_loader, va_loader, te_loader, modelSelection):
    """Train CNN and show training plots."""
    # Model
    if modelSelection.lower() == 'res50':
        model = Res50()
    elif modelSelection.lower() == 'dense121':
        model = Dense121()
    elif modelSelection.lower() == 'mobv2':
        model = Mob_v2()
    elif modelSelection.lower() == 'dense169':
        model = Dense169()
    elif modelSelection.lower() == 'mob':
        model = Net()
    elif modelSelection.lower() == 'squeeze':
        model = Squeeze()
    else:
        assert False, 'Wrong type of model selection string!'
    model = model.to(device)

    # TODO: define loss function, and optimizer
    learning_rate = utils.config(modelSelection + ".learning_rate")
    criterion = DepthLoss(0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    number_of_epoches = 10
    #

    # Attempts to restore the latest checkpoint if exists
    print("Loading unet...")
    model, start_epoch, stats = utils.restore_checkpoint(model, utils.config(modelSelection + ".checkpoint"))

    running_va_loss = [] if 'va_loss' not in stats else stats['va_loss']
    running_va_acc = [] if 'va_err' not in stats else stats['va_err']
    running_tr_loss = [] if 'tr_loss' not in stats else stats['tr_loss']
    running_tr_acc = [] if 'tr_err' not in stats else stats['tr_err']
    tr_acc, tr_loss = utils.evaluate_model(model, tr_loader, device)
    acc, loss = utils.evaluate_model(model, va_loader, device)
    running_va_acc.append(acc)
    running_va_loss.append(loss)
    running_tr_acc.append(tr_acc)
    running_tr_loss.append(tr_loss)
    stats = {
        'va_err': running_va_acc,
        'va_loss': running_va_loss,
        'tr_err': running_tr_acc,
        'tr_loss': running_tr_loss,
    }
    # Loop over the entire dataset multiple times
    # for epoch in range(start_epoch, config('cnn.num_epochs')):
    epoch = start_epoch
    # while curr_patience < patience:
    while epoch < number_of_epoches:
        # Train model
        utils.train_epoch(device, tr_loader, model, criterion, optimizer)
        # Save checkpoint
        utils.save_checkpoint(model, epoch + 1, utils.config(modelSelection + ".checkpoint"), stats)
        # Evaluate model
        tr_acc, tr_loss = utils.evaluate_model(model, tr_loader, device)
        va_acc, va_loss = utils.evaluate_model(model, va_loader, device)
        running_va_acc.append(va_acc)
        running_va_loss.append(va_loss)
        running_tr_acc.append(tr_acc)
        running_tr_loss.append(tr_loss)
        epoch += 1
    print("Finished Training")
    utils.make_plot(running_tr_loss, running_tr_acc, running_va_loss, running_va_acc)

if __name__ == "__main__":
    main(torch.device('cuda:0'), tr_loader, va_loader, te_loader, modelSelection)