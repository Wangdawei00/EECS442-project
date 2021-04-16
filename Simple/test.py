import torch
import numpy as np
import random
from data import getTrainingValidationTestingData
from model import DenseDepth
from common import *
from utils import config
import utils
from sklearn import metrics
from torch.nn.functional import softmax

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():
    """Print performance metrics for model at specified epoch."""
    # Data loaders
    pathname = "data/nyu_small.zip"
    tr_loader, va_loader, te_loader = getTrainingValidationTestingData(pathname, batch_size=config("unet.batch_size"))

    # Model
    model = DenseDepth()

    # define loss function
    criterion = torch.nn.L1Loss()

    # Attempts to restore the latest checkpoint if exists
    print("Loading unet...")
    model, start_epoch, stats = restore_checkpoint(model, config("unet.checkpoint"))

    axes = utils.make_training_plot()

    # Evaluate the model
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
        update_plot=False,
    )


if __name__ == "__main__":
    main()
