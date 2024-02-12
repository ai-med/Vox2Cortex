
""" Combination of raining and subsequent testing. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from utils.train import training_routine
from utils.test import test_routine

def train_test_routine(hps: dict, resume=False):
    """ Run a training and subsequent test routine in one run. """

    # Train
    training_routine(hps, resume)

    # Test
    test_routine(hps, resume=False)
