
""" Convenient dataset splitting. Add new split functions here.  """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

from data.supported_datasets import SupportedDatasets
from data.cortex import Cortex

# Mapping supported datasets to split functions
dataset_split_handler = {
    SupportedDatasets.DATASET_NAME.name: Cortex.split,
}

