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
from dense161 import Dense161
from mob_v2 import Mob_v2
from mob_v1 import Net
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
    
    ####### Enter the model selection here! #####
    modelSelection = input('Please input the type of model to be used(res50,dense121,dense169,dense161,mob_v2,mob):')
    
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
    elif modelSelection.lower() == 'dense161':
        model = Dense161()
    elif modelSelection.lower() == 'dense169':
        model = Dense169()
    elif modelSelection.lower() == 'mob_v2':
        model = Mob_v2()
    elif modelSelection.lower() == 'mob':
        model = Net()
    elif modelSelection.lower() == 'squeeze':
        model = Squeeze()
    else:
        assert False, 'Wrong type of model selection string!'
    model = model.to(device)


    # Attempts to restore the latest checkpoint if exists
    print(f"Loading {modelSelection}...")
    model, start_epoch, stats = utils.restore_checkpoint(model, utils.config(modelSelection+".checkpoint"))
    # Evaluate metrics
    acc, loss = utils.evaluate_model(model, te_loader, device, test=True)
    # Another implementation to valuate metrics
    """
    rel, rms, log10, theta1, theta2, theta3, loss = utils.evaluate_final_model(model, te_loader, device)
    print(f'Test rel Error:{rel}')
    print(f'Test rms Error:{rms}')
    print(f'Test log10 Error:{log10}')
    print(f'Test theta1 Error:{theta1}')
    print(f'Test theta2 Error:{theta2}')
    print(f'Test theta3 Error:{theta3}')
    print(f'Test Loss:{loss}')
    """



if __name__ == "__main__":
    main()
