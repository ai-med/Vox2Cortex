
""" Modes """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

from enum import IntEnum

class ExecModes(IntEnum):
    """ Modes for execution """
    TRAIN = 1
    TEST = 2
    TRAIN_TEST = 3
    TUNE = 4

class DataModes(IntEnum):
    """ Modes for data """
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


