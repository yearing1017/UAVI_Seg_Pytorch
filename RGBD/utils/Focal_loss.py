import numpy as np
from torch import nn
import torch

import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.loss = nn.NLLLoss(weight=torch.from_numpy(np.array(weight)).float(),
                                 size_average=self.size_average, reduce=False)

    def forward(self, input, target):
        return self.loss((1 - F.softmax(input, 1))**2 * F.log_softmax(input, 1), target)
