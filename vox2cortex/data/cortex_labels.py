
""" Mapping cortex label names and ids. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

from enum import IntEnum

import numpy as np
import torch

class CortexLabels(IntEnum):
    right_white_matter = 41
    left_white_matter = 2
    left_cerebral_cortex = 3
    right_cerebral_cortex = 42

def combine_labels(labels, names, value=1):
    """ Only consider labels in 'names' and set all those labels equally to
    'value'.
    """
    ids = [CortexLabels[n].value for n in names]
    combined_labels = np.isin(labels, ids).astype(int) * value

    if isinstance(labels, torch.Tensor):
        combined_labels = torch.from_numpy(combined_labels)

    return combined_labels
