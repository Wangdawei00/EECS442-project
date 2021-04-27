import torch
import numpy as np
import random
from sklearn import metrics
from torch.nn.functional import softmax
import argparse as arg

from data import getTestingData
from utils import config
import utils

from res50 import Res50
from dense169 import Dense169
from dense121 import Dense121
from mob_v2 import Mob_v2
from model import Net
from squeeze import Squeeze


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main(device=torch.device('cuda:0')):
    # CLI arguments  
    parser = arg.ArgumentParser(description='We all know what we are doing. Fighting!')
    parser.add_argument("--datasize", "-d", default="small", type=str,
                        help="data size you want to use, small, medium, total")
    # Parsing
    args = parser.parse_args()
    # Data loaders
    
    # TODO:
    ####### Enter the model selection here! #####
    modelSelection = input('Please input the type of model to be used(res50,dense121,dense169,mob_v2,mob):')
    
    datasize = args.datasize
    filename = "nyu_new.zip"
    pathname = f"data/{filename}"
    csv = "data/nyu_csv.zip"
    te_loader = getTestingData(datasize, csv, pathname, batch_size=config(modelSelection+".batch_size"))

    # Model
    if modelSelection.lower() == 'res50':
        model = Res50()
    elif modelSelection.lower() == 'dense121':
        model = Dense121()
    elif modelSelection.lower() == 'mob_v2':
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

    # define loss function
    # criterion = torch.nn.L1Loss()

    # Attempts to restore the latest checkpoint if exists
    print(f"Loading {mdoelSelection}...")
    model, start_epoch, stats = utils.restore_checkpoint(model, utils.config(modelSelection+".checkpoint"))
    acc, loss = utils.evaluate_model(model, te_loader, device, test=True)
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
