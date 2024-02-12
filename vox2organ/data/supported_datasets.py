
""" Put module information here """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import re
import os
import sys
from enum import IntEnum

from pandas import read_csv


# Discover whether application is running in docker
_run_docker = os.path.isdir("/mnt/code")


class SupportedDatasets(IntEnum):
    """ List supported datasets """
    ADNI_CSR_large = 4
    AbdomenCT_1K = 22
    AbdomenMRI = 23

class AbdomenCTDatasets(IntEnum):
    """ List abdominal datasets """
    AbdomenCT_1K = SupportedDatasets.AbdomenCT_1K.value

class AbdomenMRIDatasets(IntEnum):
    """ Abdomen MRI data """
    AbdomenMRI = SupportedDatasets.AbdomenMRI.value

class CortexDatasets(IntEnum):
    """ List cortex datasets """
    ADNI_CSR_large = SupportedDatasets.ADNI_CSR_large.value


dataset_paths = {
    SupportedDatasets.AbdomenMRI.name: {
        'RAW_DATA_DIR': "/mnt/data/AbdomenMRI/" if (
            _run_docker
        ) else "/mnt/nas/Data_WholeBody/AbdomenMRI/",
        'FIXED_SPLIT': ["abdomenmri_train_w_o_missing.txt",
                        "abdomenmri_val_w_o_missing.txt",
                        "abdomenmri_test_w_o_missing.txt"] # Read from files
    },
    SupportedDatasets.AbdomenCT_1K.name: {
        'RAW_DATA_DIR': "/mnt/data/AbdomenCT-1K/new_data" if (
            _run_docker
        ) else "/mnt/nas/Data_WholeBody/AbdomenCT-1K/new_data/",
        'FIXED_SPLIT': ["fold_0_train.txt",
                        "fold_0_val.txt",
                        "fold_0_test.txt"] # Read from files
    },
    SupportedDatasets.ADNI_CSR_large.name: {
        'RAW_DATA_DIR': '/mnt/data/ADNI_SEG/registered_output_mni152/FS72/' if (
            _run_docker
        ) else "/mnt/nas/Data_Neuro/ADNI_SEG/registered_output_mni152/FS72/",
        'FIXED_SPLIT': ["ADNI_large_train_qc_pass.txt",
                        "ADNI_large_val_qc_pass.txt",
                        "ADNI_large_test_qc_pass.txt"], # Read from files
    },
}
