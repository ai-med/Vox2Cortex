
""" Convenient dataset splitting. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from data.cortex import CortexDataset
from data.abdomen import AbdomenCTDataset, AbdomenMRIDataset
from data.supported_datasets import (
    CortexDatasets,
    AbdomenCTDatasets,
    AbdomenMRIDatasets,
)

# Mapping supported datasets to split functions
dataset_split_handler = {
    **{x.name: CortexDataset.split for x in CortexDatasets},
    **{x.name: AbdomenCTDataset.split for x in AbdomenCTDatasets},
    **{x.name: AbdomenMRIDataset.split for x in AbdomenMRIDatasets},
}

