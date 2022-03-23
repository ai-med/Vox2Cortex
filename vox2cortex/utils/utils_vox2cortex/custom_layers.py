
""" Custom layers for networks """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

import torch.nn as nn

class IdLayer(nn.Module):
    """ Identity layer """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
