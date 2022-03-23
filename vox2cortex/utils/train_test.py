
""" Combination of raining and subsequent testing. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

from utils.train import training_routine
from utils.test import test_routine

def train_test_routine(hps: dict, experiment_name=None, loglevel='INFO',
                       resume=False):
    """ Run a training and subsequent test routine in one run. """

    # Train
    experiment_name = training_routine(hps, experiment_name, loglevel, resume)

    # Test
    test_routine(hps, experiment_name, loglevel)




