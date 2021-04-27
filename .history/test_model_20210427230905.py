import torch
import numpy as np
import random

from sklearn import metrics
import torch.nn.functional as F

import torch.nn as nn
import torch.optim

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


def main():
    input = torch.randn(5, 3, 480, 640)
    print(input.size())
    # model = Res50()
    # model = Dense169()
    # model = Dense121()
    # model = Dense161()
    # model = Net()
    # model = Mob_v2()
    model = Squeeze()
    output = model(input)
    print(output.size())


if __name__ == "__main__":
    main()
