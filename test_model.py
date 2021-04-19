import torch
import numpy as np
import random

from sklearn import metrics
import torch.nn.functional as F

import torch.nn as nn
import torch.optim

from model import Net
from dense169 import Dense169
from dense121 import Dense121
from res50 import Res50
from mob_v2 import Mob_v2

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():
    input = torch.randn(5, 3, 480, 640)
    print(input.size())
    # model = Net()
    # model = Dense169()
    # model = Dense121()
    # model = Res50()
    model = Mob_v2()
    output = model(input)
    print(output.size())


if __name__ == "__main__":
    main()
