from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn

class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass

class TL(Regularizer):
    def __init__(self, weight: float):
        super(TL, self).__init__()
        self.weight = weight

    def forward(self, factor):
        dif = torch.mm(factor[1:], factor[:-1].transpose(0,1)).trace()
        dif = torch.norm(dif, p=2)
        return self.weight * dif / (factor.shape[0] - 1)