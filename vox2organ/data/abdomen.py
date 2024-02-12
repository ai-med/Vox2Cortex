
""" Abdomen datasets """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
from typing import Union, Sequence
from enum import IntEnum

from data.image_and_mesh_dataset import ImageAndMeshDataset
from utils.modes import DataModes
from utils.utils import global_clip_and_zscore_norm
from logger import logger

log = logger.get_std_logger(__name__)


class AbdomenCTLabels(IntEnum):
    liver = 1
    kidney = 2
    spleen = 3
    pancreas = 4


class AbdomenMRILabels(IntEnum):
    liver = 1
    spleen = 2
    kidney_right = 3
    kidney_left = 4
    pancreas = 5


class AbdomenDataset(ImageAndMeshDataset):
    """ Abdomen dataset

    This dataset contains images and meshes and has additional functionality
    specifically for abdominal data.

    :param structure_type: A description of the structure(s) to segement, e.g.,
    'abdomen-all'
    :param kwargs: Parameters for ImageAndMeshDataset
    """

    def __init__(
        self,
        structure_type: Union[str, Sequence[str]],
        **kwargs
    ):

        # Map structure type to (file-)names
        (self.voxel_label_names,
         self.mesh_label_names) = self.__class__._get_seg_and_mesh_label_names(
             structure_type
         )

        super().__init__(
            image_file_name=self.image_file_name,
            mesh_file_names=list(self.mesh_label_names.values()),
            seg_file_name=self.seg_file_name,
            **kwargs
        )


class AbdomenMRIDataset(AbdomenDataset):
    """ Abdomen MRI dataset """

    image_file_name = "registered_mri.nii.gz"
    seg_file_name = "registered_label.nii.gz"

    LabelMap = AbdomenMRILabels

    @classmethod
    def _get_seg_and_mesh_label_names(cls, structure_type):
        if structure_type == "abdomen-all":
            seg_label_names = {
                "liver": ("liver",),
                "kidney": ("kidney_left", "kidney_right",),
                "spleen": ("spleen",),
                "pancreas": ("pancreas",)
            }
            mesh_label_names = {
                "liver": "registered_liver",
                "kidney_left": "registered_kidney_left",
                "kidney_right": "registered_kidney_right",
                "spleen": "registered_spleen",
                "pancreas": "registered_pancreas"
            }
        elif structure_type == "liver-only":
            seg_label_names = {
                "liver": ("liver",),
            }
            mesh_label_names = {
                "liver": "registered_liver",
            }
        elif structure_type == "pancreas-only":
            seg_label_names = {
                "pancreas": ("pancreas",),
            }
            mesh_label_names = {
                "pancreas": "registered_pancreas",
            }
        else:
            raise NotImplementedError()

        return seg_label_names, mesh_label_names

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AbdomenCTDataset(AbdomenDataset):
    """ Abdomen CT dataset """

    # Stats about image intensities
    INTENSITY_STATS_FILE = "intensity_stats.csv"
    # Padding input images with this value (representing background)
    PAD_VALUE = -1024.0
    image_file_name = "registered_img.nii.gz"
    seg_file_name = "registered_label.nii.gz"

    LabelMap = AbdomenCTLabels

    @classmethod
    def _get_seg_and_mesh_label_names(cls, structure_type):
        if structure_type == "abdomen-all":
            seg_label_names = {
                "liver": ("liver",),
                "kidney": ("kidney",),
                "spleen": ("spleen",),
                "pancreas": ("pancreas",)
            }
            mesh_label_names = {
                "liver": "registered_liver",
                "kidney_left": "registered_kidney_left",
                "kidney_right": "registered_kidney_right",
                "spleen": "registered_spleen",
                "pancreas": "registered_pancreas"
            }

        elif structure_type == "liver-only":
            seg_label_names = {
                "liver": ("liver",),
            }
            mesh_label_names = {
                "liver": "registered_liver",
            }
        elif structure_type == "pancreas-only":
            seg_label_names = {
                "pancreas": ("pancreas",),
            }
            mesh_label_names = {
                "pancreas": "registered_pancreas",
            }
        else:
            raise NotImplementedError()

        return seg_label_names, mesh_label_names

    def __init__(self, **kwargs):
        self.img_norm = self.global_clip_and_zscore_norm
        self.img_intensity_stats_file = os.path.join(
            kwargs['raw_data_dir'], AbdomenCTDataset.INTENSITY_STATS_FILE
        )
        self._intensity_stats = None

        super().__init__(**kwargs)

    def global_clip_and_zscore_norm(self, img):
        try:
            return global_clip_and_zscore_norm(img, **self._intensity_stats)
        except TypeError:
            # Read intensity stats once
            i_stats = [
                line.rstrip('\n') for line in open(
                    self.img_intensity_stats_file, 'r'
                ).readlines()
            ]

            # Stats file for the image normalization should have been created from
            # the training split
            if self.mode == DataModes.TRAIN and not logger.debug():
                used_ids_for_stats = i_stats[3].split(",")
                check_equal = lambda x, y: (
                    len(x) == len(y) and sorted(x) == sorted(y)
                )
                if not check_equal(self.ids, used_ids_for_stats):
                    raise ValueError(
                        "The ids used for the computation of intensity stats"
                        " is not equal to the training ids!"
                    )

            i_stats = dict(zip(i_stats[0].split(","), i_stats[1].split(",")))
            self._intensity_stats = {
                'cli_lower': float(i_stats["perc005"]),
                'cli_upper': float(i_stats["perc995"]),
                'mean': float(i_stats["mean"]),
                'std': float(i_stats["std"])
            }

        return global_clip_and_zscore_norm(img, **self._intensity_stats)
