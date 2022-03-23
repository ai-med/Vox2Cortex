
""" Add supported datasets and their paths here. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

import os
import sys
from enum import IntEnum

class SupportedDatasets(IntEnum):
    """ List supported datasets """
    DATASET_NAME = 1

class CortexDatasets(IntEnum):
    """ List cortex datasets """
    DATASET_NAME = SupportedDatasets.DATASET_NAME.value

dataset_paths = {
    SupportedDatasets.DATASET_NAME.name: {
        # UPDATE HERE
        'RAW_DATA_DIR': "/path/to/raw/data",
        'PREPROCESSED_DATA_DIR': "/path/to/preprocessed/data",
        'N_REF_POINTS_PER_STRUCTURE': "<min. number of verts in training set>",
        'FIXED_SPLIT': [
            "train_ids.txt",
            "val_ids.txt",
            "test_ids.txt"
        ] # Read from files
    },
}

def valid_ids_DATASET_NAME(candidates: list):
    """ Sort out non-valid ids of 'candidates' of samples.
    """
    valid = [c for c in candidates]
    return valid


def valid_ids(raw_data_dir: str):
    """ Get valid ids for supported datasets."""

    # All files in directory are ID candidates
    all_files = os.listdir(raw_data_dir)

    # Intersection of dataset dir and dataset name
    dataset = set(
        x for y in SupportedDatasets.__members__.keys()
        for x in raw_data_dir.split("/")
        if (x in y and x != "")
    ).pop()
    this_module = sys.modules[__name__]
    return getattr(this_module, "valid_ids_" + dataset)(all_files)
