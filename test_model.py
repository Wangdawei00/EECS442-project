import torch
import numpy as np
import random

from model import Net
from sklearn import metrics
import torch.nn.functional as F

import torch.nn as nn
import torch.optim

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():
    input = torch.randn(5, 3, 480, 640)
    print(input.size())
    model = Net()
    output = model(input)
    print(output.size())


if __name__ == "__main__":
    main()
