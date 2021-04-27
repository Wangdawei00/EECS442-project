from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch

import numpy as np  
import pandas as pd
from PIL import Image
from io import BytesIO
import random
from itertools import permutations

_check_pil = lambda x: isinstance(x, Image.Image)

_check_np_img = lambda x: isinstance(x, np.ndarray)


class RandomHorizontalFlip(object):

    def __call__(self, sample):

        img, depth = sample["image"], sample["depth"]

        if not _check_pil(img):
            raise TypeError("Expected PIL type. Got {}".format(type(img)))
        if not _check_pil(depth):
            raise TypeError("Expected PIL type. Got {}".format(type(depth)))

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {
            "image": img, 
            "depth": depth
        }


class RandomChannelSwap(object):

    def __init__(self, probability):
        
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):

        image, depth = sample["image"], sample["depth"]
        
        if not _check_pil(image):
            raise TypeError("Expected PIL type. Got {}".format(type(image)))
        if not _check_pil(depth):
            raise TypeError("Expected PIL type. Got {}".format(type(depth)))
        
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        
        return {"image": image, "depth": depth}

def loadZipToMem(zip_file, size):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    
    if size == "total":
        nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
        nyu2_valid = list((row.split(',') for row in (data['data/nyu2_valid.csv']).decode("utf-8").split('\n') if len(row) > 0))
        nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))
    if size == "medium":
        nyu2_train = list((row.split(',') for row in (data['data/nyu2_train_m.csv']).decode("utf-8").split('\n') if len(row) > 0))
        nyu2_valid = list((row.split(',') for row in (data['data/nyu2_valid_m.csv']).decode("utf-8").split('\n') if len(row) > 0))
        nyu2_test = list((row.split(',') for row in (data['data/nyu2_test_m.csv']).decode("utf-8").split('\n') if len(row) > 0))
    if size == "small":
        nyu2_train = list((row.split(',') for row in (data['data/nyu2_train_s.csv']).decode("utf-8").split('\n') if len(row) > 0))
        nyu2_valid = list((row.split(',') for row in (data['data/nyu2_valid_s.csv']).decode("utf-8").split('\n') if len(row) > 0))
        nyu2_test = list((row.split(',') for row in (data['data/nyu2_test_s.csv']).decode("utf-8").split('\n') if len(row) > 0))
    
    from sklearn.utils import shuffle
    nyu2_train = shuffle(nyu2_train, random_state=0)

    #if True: nyu2_train = nyu2_train[:40]

    print('Loaded train ({0}).'.format(len(nyu2_train)))
    print('Loaded valid ({0}).'.format(len(nyu2_valid)))
    print('Loaded test ({0}).'.format(len(nyu2_test)))
    return data, nyu2_train, nyu2_valid, nyu2_test

class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_set, transform=None):
        self.data, self.nyu_dataset = data, nyu2_set
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open( BytesIO(self.data[sample[0]]) )
        depth = Image.open( BytesIO(self.data[sample[1]]) )
        sample = {'image': image, 'depth': depth}
        if self.transform: sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        image = self.to_tensor(image)

        depth = depth.resize((320, 240))
        depth = self.to_tensor(depth).float() * 1000
        
        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not(_check_pil(pic) or _check_np_img(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def getNoTransform():
    return transforms.Compose([
        ToTensor()
    ])

def getDefaultTrainTransform():
    # Normalization
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor(),
        normalize
    ])
    """
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])

def getTrainingValidationTestingData(data_size, path, batch_size):
    data, nyu2_train, nyu2_valid, nyu2_test = loadZipToMem(path, data_size)

    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_validation = depthDatasetMemory(data, nyu2_valid, transform=getNoTransform())
    transformed_testing = depthDatasetMemory(data, nyu2_test, transform=getNoTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_validation, batch_size, shuffle=False), DataLoader(transformed_testing, batch_size, shuffle=False)
   
def getTestingData(data_size, path, batch_size):
    data, _, _, nyu2_test = loadZipToMem(path, data_size)
    transformed_testing = depthDatasetMemory(data, nyu2_test, transform=getNoTransform())

    return DataLoader(transformed_testing, batch_size, shuffle=False)

