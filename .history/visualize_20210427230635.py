import torch
import numpy as np
import random
from data import getTrainingValidationTestingData
from utils import config
import utils
from sklearn import metrics
from torch.nn.functional import softmax
from glob import glob
from PIL import Image
import matplotlib.cm as cm
import cv2
import os

from res50 import Res50
from dense121 import Dense121
from dense169 import Dense169
from dense161 import Dense161
from mob_v2 import Mob_v2
from mob_v1 import Net
from squeeze import Squeeze

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def colorize(value, vmin=10, vmax=1000, cmap="viridis"):

    value = value.cpu().numpy()[0, :, :]

    # normalize 
    vmin = value.min() if vmin is None else vmin 
    vmax = value.max() if vmax is None else vmax   
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0  
    
    cmapper =  cm.get_cmap(cmap)
    value = cmapper(value, bytes=True) 

    img = value[:,:,:3]

    return img.transpose((2, 0, 1))

def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(np.asarray(Image.open( file ).resize((640, 480)), dtype=float) / 255, 0, 1).transpose(2, 0, 1)
        
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)


def main(device=torch.device('cuda:0')):
    # Model
    modelSelection = input('Please input the type of model to be used(res50,dense121,dense169,mob_v2,mob):')
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

    # Attempts to restore the latest checkpoint if exists
    print("Loading unet...")
    model, start_epoch, stats = utils.restore_checkpoint(model, utils.config(modelSelection+".checkpoint"))

    # Get Test Images  
    img_list = glob("examples/"+"*.png")
    
    # Set model to eval mode 
    model.eval()
    model = model.to(device)

    # Begin testing loop 
    print("Begin Test Loop ...")
    
    for idx, img_name in enumerate(img_list):

        img = load_images([img_name])     
        img = torch.Tensor(img).float().to(device)   
        print("Processing {}, Tensor Shape: {}".format(img_name, img.shape))

        with torch.no_grad():
            preds = model(img).squeeze(0)           

        output = colorize(preds.data)
        output = output.transpose((1, 2, 0))
        cv2.imwrite(img_name.split(".")[0]+"_"+modelSelection+"_result.png", output)

        print("Processing {} done.".format(img_name))



if __name__ == "__main__":
    main()
