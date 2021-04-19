import torch
import numpy as np
import random
from data import getTrainingValidationTestingData
from model import Net
# from common import *
from utils import config
import utils
from sklearn import metrics
from torch.nn.functional import softmax
import argparse as arg

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

    # define loss function
    # criterion = torch.nn.L1Loss()

    # Attempts to restore the latest checkpoint if exists
    print("Loading unet...")
    model, start_epoch, stats = utils.restore_checkpoint(model, utils.config("unet.checkpoint"))
    acc, loss = utils.evaluate_model(model, te_loader, device)
    # axes = util.make_training_plot()
    print(f'Test Error:{acc}')
    print(f'Test Loss:{loss}')

    # Evaluate the model
    # evaluate_epoch(
    #     axes,
    #     tr_loader,
    #     va_loader,
    #     te_loader,
    #     model,
    #     criterion,
    #     start_epoch,
    #     stats,
    #     include_test=True,
    #     update_plot=False,
    # )


if __name__ == "__main__":
    main()
